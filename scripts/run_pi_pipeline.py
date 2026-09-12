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

# 1. Run main staging pipeline
print("Running core staging pipeline...")
subprocess.run(["python", "run_full_pipeline.py"], cwd=REPO_ROOT, check=True)

# 2. Explicitly trigger dashboard export if present
if (REPO_ROOT / "export_dashboard.py").exists():
    print("Generating HTML dashboard...")
    subprocess.run(["python", "export_dashboard.py"], cwd=REPO_ROOT, check=True)

# 3. Collect all output artifacts
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
