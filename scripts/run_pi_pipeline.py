import os
import shutil
import subprocess
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default=str(REPO_ROOT / "data" / "csv"))
parser.add_argument("--output_dir", default=str(REPO_ROOT / "outputs" / "pi_device"))
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Copy CSV inputs to root
for fname in ["ecg.csv", "max30100_full_day.csv", "mpu6050_full_day.csv"]:
    src = Path(args.input_dir) / fname
    dst = REPO_ROOT / fname
    if src.exists():
        shutil.copy(src, dst)
    else:
        raise FileNotFoundError(f"Missing input CSV: {src}")

# 1. Staging pipeline (generates sleepfm_staging_predictions.csv and hypnogram.png)
print("Step 1: Running core staging pipeline...")
subprocess.run(["python", "run_full_pipeline.py"], cwd=REPO_ROOT, check=True)

# 2. Disease risk / clinical hazard labeling (generates sleepfm_labeled_clinical_report.csv)
print("Step 2: Generating clinical risk hazards and impressions...")
if (REPO_ROOT / "run_diagnosis.py").exists():
    subprocess.run(["python", "run_diagnosis.py"], cwd=REPO_ROOT, check=True)
elif (REPO_ROOT / "generate_clinical_impression.py").exists():
    subprocess.run(["python", "generate_clinical_impression.py"], cwd=REPO_ROOT, check=True)

# 3. HTML Dashboard export
print("Step 3: Exporting HTML dashboard...")
if (REPO_ROOT / "export_dashboard.py").exists():
    subprocess.run(["python", "export_dashboard.py"], cwd=REPO_ROOT, check=True)

# 4. Collect generated artifacts
artifacts = [
    "clinical_dashboard.html",
    "sleepfm_staging_predictions.csv",
    "sleepfm_labeled_clinical_report.csv",
    "hypnogram.png"
]

for item in artifacts:
    src = REPO_ROOT / item
    if src.exists():
        shutil.copy(src, Path(args.output_dir) / item)
        print(f"Collected artifact -> {item}")
    else:
        print(f"Warning: {item} not found in root")
