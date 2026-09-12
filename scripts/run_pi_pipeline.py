import os
import shutil
import subprocess
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="Path to input data folder")
parser.add_argument("--output_dir", required=True, help="Destination folder for artifacts")
args = parser.parse_args()

input_path = Path(args.input_dir)
if not input_path.is_absolute():
    input_path = REPO_ROOT / input_path

output_path = Path(args.output_dir)
if not output_path.is_absolute():
    output_path = REPO_ROOT / output_path

os.makedirs(output_path, exist_ok=True)

def stage_compressed_file(base_name, search_dir):
    candidates = [
        search_dir / f"{base_name}.csv.Z",
        search_dir / f"{base_name}.Z",
        search_dir / f"{base_name}.csv",
    ]
    
    target_file = None
    for cand in candidates:
        if cand.exists():
            target_file = cand
            break
            
    if not target_file:
        raise FileNotFoundError(f"Missing input log for '{base_name}' in {search_dir}")

    dest_csv = REPO_ROOT / f"{base_name}.csv"

    if target_file.name.endswith(".Z"):
        print(f"Decompressing {target_file.name} -> {dest_csv.name}...")
        with open(dest_csv, "wb") as f_out:
            subprocess.run(["gzip", "-dc", str(target_file)], stdout=f_out, check=True)
    else:
        print(f"Staging {target_file.name} -> {dest_csv.name}...")
        shutil.copy(target_file, dest_csv)

# Decompress and stage inputs
required_logs = ["ecg", "max30100_full_day", "mpu6050_full_day"]
for log_prefix in required_logs:
    stage_compressed_file(log_prefix, input_path)

# Run staging and clinical risk analysis
print("\nStep 1: Running pipeline CLI...")
subprocess.run(["python", "run_pipeline_cli.py"], cwd=REPO_ROOT, check=True)

# Export HTML dashboard
print("\nStep 2: Exporting HTML dashboard...")
subprocess.run(["python", "export_dashboard.py"], cwd=REPO_ROOT, check=True)

# Gather generated artifacts
artifacts = [
    "clinical_dashboard.html",
    "sleepfm_staging_predictions.csv",
    "sleepfm_labeled_clinical_report.csv",
    "hypnogram.png"
]

for item in artifacts:
    src = REPO_ROOT / item
    if src.exists():
        shutil.copy(src, output_path / item)
        print(f"Archived artifact -> {output_path / item}")
    else:
        print(f"Warning: {item} not found in root")

print(f"\nExecution complete. Output artifacts preserved in {output_path}")
