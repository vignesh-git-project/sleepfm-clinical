import base64
import os
import pandas as pd
import numpy as np
from generate_clinical_impression import evaluate_clinical_impression

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_f:
        return f"data:image/png;base64,{base64.b64encode(img_f.read()).decode('utf-8')}"

def build_dashboard(output_file="clinical_dashboard.html"):
    df_stg = pd.read_csv("sleepfm_staging_predictions.csv")
    df_haz = pd.read_csv("sleepfm_labeled_clinical_report.csv")

    stg_dist = df_stg["sleepfm_stage"].value_counts(normalize=True) * 100
    wake_pct = stg_dist.get("Wake", 0.0)
    n1_pct = stg_dist.get("N1", 0.0)
    n2_pct = stg_dist.get("N2", 0.0)
    n3_pct = stg_dist.get("N3", 0.0)
    rem_pct = stg_dist.get("REM", 0.0)
    eff = 100.0 - wake_pct

    # Calculate percentiles across all 1,065 conditions
    df_haz["percentile"] = (df_haz["log_hazard"].rank(pct=True) * 100).round(1)

    def assign_badge(pct):
        if pct < 35.0:
            return "Low / Below Average", "badge-low"
        elif pct <= 65.0:
            return "Normal / Baseline", "badge-normal"
        elif pct <= 85.0:
            return "Moderate Risk Elevation", "badge-warn"
        else:
            return "High Risk Elevation", "badge-high"

    clinical_domains = {
        "Respiratory & Sleep-Disordered Breathing": ["327.32", "327.31", "513.31", "327.4", "327.6"],
        "Cardiovascular & Autonomic Rhythms": ["428.0", "427.5", "401.3", "415.21", "401.1", "427.21"],
        "Cardiometabolic & Systemic Baseline": ["250.2", "250.1", "249.0", "401.2", "402.0"]
    }

    domain_tables_html = ""
    for domain, codes in clinical_domains.items():
        sub = df_haz[df_haz["phecode"].astype(str).isin(codes)].sort_values(by="percentile", ascending=False)
        
        rows = ""
        for _, r in sub.iterrows():
            tier, cls = assign_badge(r['percentile'])
            rows += f"""
            <tr>
                <td><code>{r['phecode']}</code></td>
                <td style="font-weight: 500;">{r['phenotype']}</td>
                <td>{r['log_hazard']:+.2f}</td>
                <td><strong>{r['percentile']:.1f}%</strong></td>
                <td><span class="badge {cls}">{tier}</span></td>
            </tr>"""

        domain_tables_html += f"""
        <div class="section-title">{domain}</div>
        <table>
            <thead>
                <tr>
                    <th style="width: 110px;">PheCode</th>
                    <th>Phenotype Description</th>
                    <th style="width: 160px;">Relative Log-Score (&beta;X)</th>
                    <th style="width: 140px;">Cohort Percentile</th>
                    <th style="width: 190px;">Clinical Stratification</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>"""

    clinical_impression_text = evaluate_clinical_impression()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SleepFM Clinical Diagnostic Dashboard</title>
    <style>
        :root {{
            --bg: #0f172a; --surface: #1e293b; --text: #f8fafc;
            --text-muted: #94a3b8; --primary: #38bdf8; --danger: #f87171;
            --warning: #fbbf24; --success: #34d399; --border: #334155;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); padding: 30px; margin: 0; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 25px; }}
        h1 {{ margin: 0; font-size: 24px; color: var(--primary); }}
        .meta {{ color: var(--text-muted); font-size: 14px; line-height: 1.5; }}
        .card-impression {{ background: var(--surface); border: 1px solid var(--border); border-left: 5px solid var(--primary); border-radius: 8px; padding: 20px; margin-bottom: 25px; }}
        .grid-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }}
        .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 18px; }}
        .card-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }}
        .card-val {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
        .section-title {{ font-size: 16px; font-weight: 600; margin: 25px 0 10px 0; color: var(--text); }}
        .hypno-img {{ width: 100%; border-radius: 8px; background: #fff; display: block; }}
        table {{ width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); margin-bottom: 20px; }}
        th, td {{ padding: 11px 16px; text-align: left; border-bottom: 1px solid var(--border); font-size: 14px; }}
        th {{ background: #111827; color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
        code {{ background: rgba(255,255,255,0.1); padding: 2px 5px; border-radius: 4px; color: var(--primary); font-family: monospace; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; display: inline-block; }}
        .badge-high {{ background: rgba(248, 113, 113, 0.2); color: var(--danger); border: 1px solid var(--danger); }}
        .badge-warn {{ background: rgba(251, 191, 36, 0.2); color: var(--warning); border: 1px solid var(--warning); }}
        .badge-normal {{ background: rgba(56, 189, 248, 0.2); color: var(--primary); border: 1px solid var(--primary); }}
        .badge-low {{ background: rgba(52, 211, 153, 0.2); color: var(--success); border: 1px solid var(--success); }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <div>
            <h1>SleepFM Clinical Diagnostic Dashboard</h1>
            <div class="meta">Multi-Modal Wearable Sensor Fusion (MPU6050 + MAX30100 + AD8232)</div>
        </div>
        <div class="meta" style="text-align: right;">
            <div>Demographic Prior: <strong>Age 28, Male</strong></div>
            <div>Recording Duration: <strong>8.0 Hours (960 Epochs)</strong></div>
        </div>
    </header>

    <div class="card-impression">
        <div class="card-label" style="color: var(--primary); font-weight: 700;">Automated Clinical Diagnostic Impression</div>
        <div style="font-size: 15px; line-height: 1.6; margin-top: 8px;">{clinical_impression_text}</div>
    </div>

    <div class="grid-cards">
        <div class="card"><div class="card-label">Sleep Efficiency</div><div class="card-val" style="color: var(--success);">{eff:.1f}%</div></div>
        <div class="card"><div class="card-label">Wake</div><div class="card-val">{wake_pct:.1f}%</div></div>
        <div class="card"><div class="card-label">Light (N1+N2)</div><div class="card-val">{n1_pct + n2_pct:.1f}%</div></div>
        <div class="card"><div class="card-label">Deep (N3)</div><div class="card-val" style="color: var(--primary);">{n3_pct:.1f}%</div></div>
        <div class="card"><div class="card-label">REM Sleep</div><div class="card-val">{rem_pct:.1f}%</div></div>
    </div>

    <div class="section-title">AASM 5-Stage Hypnogram Profile (960 Epochs / 30-sec windows)</div>
    <div class="card" style="padding: 8px;"><img class="hypno-img" src="{get_base64_image('hypnogram.png')}" alt="Hypnogram"></div>

    {domain_tables_html}
</div>
</body>
</html>
"""
    with open(output_file, "w") as f:
        f.write(html)
    print(f"[*] Standalone clinical dashboard exported: {output_file}")

if __name__ == "__main__":
    build_dashboard()
