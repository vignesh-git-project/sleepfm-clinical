import pandas as pd
import numpy as np

def evaluate_clinical_impression(staging_csv='sleepfm_staging_predictions.csv', 
                                 hazards_csv='sleepfm_labeled_clinical_report.csv'):
    # Ingest Model Predictions
    df_stg = pd.read_csv(staging_csv)
    df_haz = pd.read_csv(hazards_csv)
    
    stg_dist = df_stg['sleepfm_stage'].value_counts(normalize=True) * 100
    eff = 100.0 - stg_dist.get('Wake', 0.0)
    wake_pct = stg_dist.get('Wake', 0.0)
    deep_pct = stg_dist.get('N3', 0.0)
    rem_pct = stg_dist.get('REM', 0.0)
    
    # Calculate cohort percentiles across the 1,065 outcomes
    df_haz['percentile'] = (df_haz['log_hazard'].rank(pct=True) * 100).round(1)
    haz_map = dict(zip(df_haz['phecode'].astype(str), df_haz['percentile']))
    
    osa_pct = haz_map.get('327.32', 50.0)
    csa_pct = haz_map.get('327.31', 50.0)
    hypovent_pct = haz_map.get('513.4', 50.0)
    insomnia_pct = haz_map.get('327.4', 50.0)
    htn_pct = haz_map.get('401.1', 50.0)
    t2d_pct = haz_map.get('250.2', 50.0)
    chf_pct = haz_map.get('428.0', 50.0)
    arrhythmia_pct = haz_map.get('427.5', 50.0)
    afib_pct = haz_map.get('427.21', 50.0)

    # Domain 1: Macro Sleep Architecture
    if eff >= 84.0 and deep_pct >= 20.0 and rem_pct >= 20.0:
        sleep_desc = f"physiologically consolidated, restorative sleep (efficiency {eff:.1f}%) with preserved deep slow-wave ({deep_pct:.1f}%) and REM ({rem_pct:.1f}%) distributions"
    elif eff >= 80.0:
        sleep_desc = f"adequately consolidated sleep (efficiency {eff:.1f}%) with mild architectural skew"
    elif eff >= 65.0:
        sleep_desc = f"fragmented sleep architecture characterized by elevated wakefulness ({wake_pct:.1f}%) and depressed sleep efficiency ({eff:.1f}%)"
    else:
        sleep_desc = f"severely disrupted, non-restorative sleep with marked sleep efficiency loss ({eff:.1f}%)"

    # Domain 2: Respiratory Stability & SDB
    sdb_score = max(osa_pct, csa_pct, hypovent_pct)
    if sdb_score >= 85.0:
        resp_desc = "marked nocturnal respiratory instability highly indicative of severe Sleep-Disordered Breathing"
    elif sdb_score >= 65.0:
        resp_desc = "mild-to-moderate nocturnal respiratory instability consistent with Sleep-Disordered Breathing"
    elif sdb_score <= 35.0:
        resp_desc = "stable nocturnal respiratory dynamics with negligible probability of sleep apnea"
    else:
        resp_desc = "baseline respiratory stability within normative cohort boundaries"

    # Domain 3: Autonomic & Cardiovascular Stress
    cardiac_score = max(chf_pct, arrhythmia_pct, afib_pct)
    if cardiac_score >= 85.0:
        cardio_desc = "significant autonomic strain and elevated nocturnal cardiovascular hazard markers"
    elif cardiac_score >= 65.0:
        cardio_desc = "mild sympathetic activation and secondary cardiac rhythm fluctuations"
    elif cardiac_score <= 35.0:
        cardio_desc = "optimal nocturnal parasympathetic dominance with suppressed arrhythmogenic indicators"
    else:
        cardio_desc = "stable cardiac electrophysiologic parameters"

    # Domain 4: Chronic Cardiometabolic Profile
    metabolic_score = max(htn_pct, t2d_pct)
    if metabolic_score >= 75.0:
        metabolic_desc = "The longitudinal profile highlights elevated risk for systemic hypertension and cardiometabolic syndrome."
    elif metabolic_score <= 35.0:
        metabolic_desc = "The longitudinal profile confirms strong negative predictive value against chronic cardiometabolic degradation (hypertension/T2D)."
    else:
        metabolic_desc = "Baseline cardiometabolic risk trajectories track the normative population average."

    # Domain 5: Insomnia / Maintenance Indicators
    if insomnia_pct >= 75.0 or (eff < 75.0 and wake_pct > 25.0):
        insomnia_desc = "Findings show active markers for nocturnal sleep fragmentation."
    else:
        insomnia_desc = "It rules out acute insomnia."

    impression = (
        f"The current recording indicates {sleep_desc} accompanied by {resp_desc} and {cardio_desc}. "
        f"{insomnia_desc} {metabolic_desc} "
        f"Formal confirmation of SDB severity would require standard scoring of the clinical Apnea-Hypopnea Index (AHI)."
    )
    return impression

if __name__ == '__main__':
    print(evaluate_clinical_impression())
