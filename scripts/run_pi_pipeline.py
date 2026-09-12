import os
import shutil
import subprocess
import argparse
from pathlib import Path

# Identify repository root (parent of the scripts/ directory)
REPO_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default=str(REPO_ROOT / "data" / "csv"))
parser.add_argument("--output_dir", default=str(REPO_ROOT / "outputs" / "pi_device"))
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Copy CSVs to repo root where the core pipeline looks for them
for fname in ["ecg.csv", "max30100_full_day.csv", "mpu6050_full_day.csv"]:
    src = Path(args.input_dir) / fname
    dst = REPO_ROOT / fname
    if src.exists():
        shutil.copy(src, dst)
    else:
        raise FileNotFoundError(f"Missing required file: {src}")

# Execute the pipeline with cwd set strictly to REPO_ROOT
subprocess.run(["python", "run_full_pipeline.py"], cwd=REPO_ROOT, check=True)

# Collect generated artifacts into the output folder
artifacts = [
    "sleepfm_staging_predictions.csv",
    "clinical_dashboard.html",
    "sleepfm_labeled_clinical_report.csv",
    "hypnogram.png"
]

for item in artifacts:
    src = REPO_ROOT / item
    if src.exists():
        shutil.copy(src, Path(args.output_dir) / item)
