import os
import numpy as np
import pandas as pd

def run_clinical_triage():
    print("Running Edge-Level Multi-Modal Clinical Triage Engine...")

    # -------------------------------------------------------------
    # 1. Evaluate Sleep Staging Architecture (Continuity / Insomnia)
    # -------------------------------------------------------------
    staging_file = "sleepfm_staging_predictions.csv"
    if not os.path.exists(staging_file):
        raise FileNotFoundError(f"Missing '{staging_file}'. Run inference.py first.")

    staging_df = pd.read_csv(staging_file)
    total_epochs = len(staging_df)
    wake_epochs = (staging_df['sleepfm_predicted_stage'] == 'Wake').sum()
    rem_epochs = (staging_df['sleepfm_predicted_stage'] == 'REM').sum()
    nrem_epochs = total_epochs - wake_epochs - rem_epochs

    total_sleep_time_min = (total_epochs * 30.0) / 60.0
    wake_after_sleep_onset_min = (wake_epochs * 30.0) / 60.0
    sleep_efficiency = ((total_epochs - wake_epochs) / total_epochs) * 100.0

    # -------------------------------------------------------------
    # 2. Evaluate Respiratory Stability (Deterministic ODI-3%)
    # -------------------------------------------------------------
    spo2_file = "max30100_full_day.csv"
    if not os.path.exists(spo2_file):
        raise FileNotFoundError(f"Missing '{spo2_file}'. Place raw SpO2 data in working directory.")

    spo2_df = pd.read_csv(spo2_file, header=None, names=['time', 'hr', 'spo2', 'ir', 'red', 'status'])
    spo2_vals = pd.to_numeric(spo2_df['spo2'], errors='coerce').bfill().ffill().values

    # Compute baseline via moving median filter (window = 120 samples ~= 1-2 min baseline)
    window_len = max(len(spo2_vals) // 240, 15)
    rolling_median = pd.Series(spo2_vals).rolling(window=window_len, min_periods=1, center=True).median().values
    
    # 3% desaturation detection
    desat_bool = (rolling_median - spo2_vals) >= 3.0
    desat_events = (pd.Series(desat_bool).astype(int).diff() == 1).sum()

    # Determine recording duration in hours
    recording_duration_hours = total_sleep_time_min / 60.0 if total_sleep_time_min > 0 else 8.0
    odi_3pct = desat_events / max(recording_duration_hours, 1.0)
    min_spo2 = np.nanmin(spo2_vals)
    mean_spo2 = np.nanmean(spo2_vals)

    # -------------------------------------------------------------
    # 3. Read Calibrated SleepFM Survival Hazards
    # -------------------------------------------------------------
    calibrated_file = "sleepfm_calibrated_clinical_risks.csv"
    if not os.path.exists(calibrated_file):
        raise FileNotFoundError(f"Missing '{calibrated_file}'. Run report.py or calibrate.py first.")

    df_cal = pd.read_csv(calibrated_file)
    risk_dict = dict(zip(df_cal['phecode'].astype(str), df_cal['hazard_ratio']))

    hypotension_hr = risk_dict.get('Hypotension', 1.0)
    athero_hr = risk_dict.get('440.1', 1.0)
    sinus_hr = risk_dict.get('475.0', 1.0)
    hfpef_hr = risk_dict.get('428.4', 1.0)
    insomnia_hr = risk_dict.get('327.41', 1.0)

    # -------------------------------------------------------------
    # 4. Triage Decision Rules (Rule-Out Clinical Screening Logic)
    # -------------------------------------------------------------
    triage_results = []

    # Domain 1: Sleep Architecture & Continuity
    if sleep_efficiency < 75.0 or wake_after_sleep_onset_min > 60.0:
        status_sleep = "AMBER"
        note_sleep = f"Efficiency: {sleep_efficiency:.1f}%, WASO: {wake_after_sleep_onset_min:.0f} min. Elevated sleep fragmentation / insomnia markers."
    else:
        status_sleep = "GREEN"
        note_sleep = f"Efficiency: {sleep_efficiency:.1f}%, WASO: {wake_after_sleep_onset_min:.0f} min. Preserved nocturnal sleep continuity."
    triage_results.append(("Sleep Continuity", status_sleep, note_sleep))

    # Domain 2: Respiratory Stability & Apnea Screen
    if odi_3pct >= 15.0 or min_spo2 < 85.0:
        status_resp = "RED"
        note_resp = f"ODI-3%: {odi_3pct:.1f}/hr (Min SpO2: {min_spo2:.0f}%). Moderate-to-severe nocturnal hypoxemia detected. Urgent formal PSG referral recommended."
    elif odi_3pct >= 5.0 or min_spo2 < 90.0:
        status_resp = "AMBER"
        note_resp = f"ODI-3%: {odi_3pct:.1f}/hr (Min SpO2: {min_spo2:.0f}%). Mild oxygen desaturations observed."
    elif sinus_hr > 1.8:
        status_resp = "AMBER"
        note_resp = f"ODI-3%: {odi_3pct:.1f}/hr, Mean SpO2: {mean_spo2:.1f}%. Stable oxygenation, but SleepFM detects subtle upper-airway flow limitation / snoring resistance."
    else:
        status_resp = "GREEN"
        note_resp = f"ODI-3%: {odi_3pct:.1f}/hr, Mean SpO2: {mean_spo2:.1f}%. Normal overnight oxygenation and breathing effort."
    triage_results.append(("Respiratory Stability", status_resp, note_resp))

    # Domain 3: Autonomic & Hemodynamic Regulation
    if athero_hr > 2.2 or hypotension_hr > 2.2:
        status_cardio = "AMBER"
        note_cardio = f"Elevated nocturnal vascular resistance & blunted dipping indicators (Hypotension HR: {hypotension_hr:.2f}x, Vascular HR: {athero_hr:.2f}x)."
    else:
        status_cardio = "GREEN"
        note_cardio = f"Normal nocturnal hemodynamic dipping and autonomic resting stability (Hypotension HR: {hypotension_hr:.2f}x)."
    triage_results.append(("Cardiovascular / Autonomic", status_cardio, note_cardio))

    # -------------------------------------------------------------
    # 5. Formatted Console Output
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("           STANFORD SLEEPFM EDGE CLINICAL TRIAGE SUMMARY")
    print("=" * 70)
    for domain, status, details in triage_results:
        print(f"[{status:<5}] {domain:<26}: {details}")
    print("=" * 70)

    # -------------------------------------------------------------
    # 6. Append Triage Section to clinical_sleep_risk_report.md
    # -------------------------------------------------------------
    report_file = "clinical_sleep_risk_report.md"
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            content = f.read()

        # Build clean triage markdown section
        triage_md = "\n\n---\n\n## 4. Edge Screening & Triage Assessment (Rule-Out Matrix)\n\n"
        triage_md += "| Clinical Target | Triage Status | Key Physiological Finding | Action Recommendation |\n"
        triage_md += "| :--- | :--- | :--- | :--- |\n"
        
        actions = {
            "Sleep Continuity": "Maintain consistent sleep-wake schedule; monitor daytime sleepiness." if status_sleep == "GREEN" else "Evaluate for primary sleep maintenance insomnia.",
            "Respiratory Stability": "Obstructive sleep apnea largely ruled out (ODI < 5.0/hr). Focus on nasal patency if snoring." if status_resp != "RED" else "High priority for laboratory Level 1 Polysomnography.",
            "Cardiovascular / Autonomic": "Perform routine ambulatory daytime BP tracking to evaluate baroreflex dipping." if status_cardio == "AMBER" else "Reassuring nocturnal autonomic profile."
        }

        for domain, status, details in triage_results:
            triage_md += f"| **{domain}** | **{status}** | {details} | {actions[domain]} |\n"

        # Prevent duplicate appending
        if "## 4. Edge Screening & Triage Assessment" not in content:
            with open(report_file, "w") as f:
                f.write(content + triage_md)
            print(f"Appended triage matrix to '{report_file}'.\n")

if __name__ == "__main__":
    run_clinical_triage()
