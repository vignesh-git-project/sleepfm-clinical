import os
import gzip
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

def stage_or_decompress(base_name, target_dir):
    """
    Looks for base_name.csv, base_name.csv.z, base_name.z, or base_name.csv.gz,
    and decompresses/copies it to target_dir/base_name.csv.
    """
    candidates = [
        target_dir / f"{base_name}.csv",
        target_dir / f"{base_name}.csv.z",
        target_dir / f"{base_name}.csv.Z",
        target_dir / f"{base_name}.z",
        target_dir / f"{base_name}.Z",
        target_dir / f"{base_name}.csv.gz"
    ]
    
    found_file = None
    for cand in candidates:
        if cand.exists():
            found_file = cand
            break
            
    if not found_file:
        raise FileNotFoundError(f"Could not find {base_name} (.csv, .z, or .gz) in {target_dir}")
        
    dest_file = REPO_ROOT / f"{base_name}.csv"
    
    # 1. Standard uncompressed CSV
    if found_file.suffix.lower() == ".csv":
        shutil.copy(found_file, dest_file)
        print(f"Staged uncompressed: {found_file.name} -> {dest_file.name}")
        return

    # 2. Compressed file (.z / .gz)
    print(f"Decompressing {found_file.name} -> {dest_file.name}...")
    try:
        # Try gzip/zlib stream first (standard for modern .z/.gz exports)
        with gzip.open(found_file, 'rb') as f_in, open(dest_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    except gzip.BadGzipFile:
        # Fallback to Unix uncompress / gzip CLI tool if standard LZW .Z
        subprocess.run(["gzip", "-dc", str(found_file)], stdout=open(dest_file, "wb"), check=True)

# Process all 3 sensor logs
required_bases = ["ecg", "max30100_full_day", "mpu6050_full_day"]
for base in required_bases:
    stage_or_decompress(base, input_path)

# Step 2: Run pipeline CLI
print("\nStep 1: Running pipeline CLI...")
subprocess.run(["python", "run_pipeline_cli.py"], cwd=REPO_ROOT, check=True)

# Step 3: Export HTML dashboard
print("\nStep 2: Exporting HTML dashboard...")
subprocess.run(["python", "export_dashboard.py"], cwd=REPO_ROOT, check=True)

# Step 4: Collect generated artifacts
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

print(f"\nCompleted! Artifacts stored in {output_path}")
