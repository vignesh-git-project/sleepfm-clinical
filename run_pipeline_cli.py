import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sleepfm.models.models import (
    SetTransformer,
    SleepEventLSTMClassifier,
    DiagnosisFinetuneFullLSTMCOXPHWithDemo
)
from generate_clinical_impression import evaluate_clinical_impression
from export_dashboard import build_dashboard

def parse_args():
    parser = argparse.ArgumentParser(description="SleepFM Multi-Modal Clinical Inference Engine CLI")
    parser.add_argument("--imu", type=str, default="mpu6050_full_day.csv", help="Path to MPU6050 actigraphy CSV")
    parser.add_argument("--ppg", type=str, default="max30100_full_day.csv", help="Path to MAX30100 PPG optical CSV")
    parser.add_argument("--ecg", type=str, default="ecg.csv", help="Path to AD8232 single-lead ECG CSV")
    parser.add_argument("--age", type=float, default=28.0, help="Subject age in years")
    parser.add_argument("--sex", type=int, default=1, choices=[0, 1], help="Biological sex: 0=Female, 1=Male")
    parser.add_argument("--output_html", type=str, default="clinical_dashboard.html", help="Dashboard output path")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[*] Starting SleepFM Pipeline on device: {device}")
    print(f"[*] Demographics Prior: Age {args.age}, Sex {'Male' if args.sex == 1 else 'Female'}")

    # 1. Load Foundation Backbone
    with open("sleepfm/checkpoints/model_base/config.json") as f:
        base_cfg = json.load(f)
    base_model = SetTransformer(
        in_channels=base_cfg.get("in_channels", 1),
        patch_size=base_cfg.get("patch_size", 640),
        embed_dim=base_cfg.get("embed_dim", 128),
        num_heads=base_cfg.get("num_heads", 8),
        num_layers=base_cfg.get("num_layers", 6),
        pooling_head=base_cfg.get("pooling_head", 8),
        dropout=base_cfg.get("dropout", 0.3)
    ).to(device)
    base_ckpt = torch.load("sleepfm/checkpoints/model_base/best.pt", map_location=device)
    base_sd = base_ckpt.get("state_dict", base_ckpt.get("model_state_dict", base_ckpt))
    base_model.load_state_dict({k.replace("module.", ""): v for k, v in base_sd.items()}, strict=False)
    base_model.eval()

    # 2. Load Staging Head
    with open("sleepfm/checkpoints/model_sleep_staging/config.json") as f:
        stg_cfg = json.load(f)
    stg_p = stg_cfg["model_params"]
    stg_model = SleepEventLSTMClassifier(
        embed_dim=stg_p["embed_dim"],
        num_heads=stg_p["num_heads"],
        num_layers=stg_p["num_layers"],
        num_classes=stg_p["num_classes"],
        pooling_head=stg_p["pooling_head"],
        dropout=stg_p["dropout"],
        max_seq_length=stg_p["max_seq_length"]
    ).to(device)
    stg_ckpt = torch.load("sleepfm/checkpoints/model_sleep_staging/best.pth", map_location=device)
    stg_sd = stg_ckpt.get("state_dict", stg_ckpt.get("model_state_dict", stg_ckpt))
    stg_model.load_state_dict({k.replace("module.", ""): v for k, v in stg_sd.items()}, strict=False)
    stg_model.eval()

    # 3. Load Diagnostic Head
    with open("sleepfm/checkpoints/model_diagnosis/config.json") as f:
        diag_cfg = json.load(f)
    diag_p = diag_cfg["model_params"]
    diag_model = DiagnosisFinetuneFullLSTMCOXPHWithDemo(
        embed_dim=diag_p["embed_dim"],
        num_heads=diag_p["num_heads"],
        num_layers=diag_p["num_layers"],
        num_classes=diag_p["num_classes"],
        pooling_head=diag_p["pooling_head"],
        dropout=diag_p["dropout"],
        max_seq_length=diag_p["max_seq_length"]
    ).to(device)
    diag_ckpt = torch.load("sleepfm/checkpoints/model_diagnosis/best.pth", map_location=device)
    diag_sd = diag_ckpt.get("state_dict", diag_ckpt.get("model_state_dict", diag_ckpt))
    diag_model.load_state_dict({k.replace("module.", ""): v for k, v in diag_sd.items()}, strict=False)
    diag_model.eval()

    # 4. Sensor Conditioning
    CHUNK_LEN = 38400
    N_CHUNKS = 96
    TOTAL_SAMPLES = N_CHUNKS * CHUNK_LEN
    N_TOKENS = N_CHUNKS * 60
    t_target = np.linspace(0, 1, TOTAL_SAMPLES)

    print("[*] Resampling sensor channels to uniform 128 Hz timeline...")
    # BAS
    mpu_df = pd.read_csv(args.imu)
    mag = pd.to_numeric(mpu_df['magnitude'], errors='coerce').bfill().ffill().values
    act = np.abs(mag - np.median(mag))
    bas = np.interp(t_target, np.linspace(0, 1, len(act)), (act - np.mean(act)) / (np.std(act) + 1e-6)).astype(np.float32)

    # RESP
    spo2_df = pd.read_csv(args.ppg, header=None, names=['time', 'hr', 'spo2', 'ir', 'red', 'status'])
    ir = pd.to_numeric(spo2_df['ir'], errors='coerce').bfill().ffill().values
    w = max(int(len(ir) / 28800 * 2.5), 5)
    resp_slow = pd.Series(ir).rolling(w, min_periods=1, center=True).mean().values
    resp_b = resp_slow - pd.Series(resp_slow).rolling(w * 10, min_periods=1, center=True).mean().values
    resp = np.interp(t_target, np.linspace(0, 1, len(resp_b)), (resp_b - np.mean(resp_b)) / (np.std(resp_b) + 1e-6)).astype(np.float32)

    # EKG
    ecg_df = pd.read_csv(args.ecg)
    ecg = pd.to_numeric(ecg_df['filtered_v'], errors='coerce').bfill().ffill().values
    ekg = np.interp(t_target, np.linspace(0, 1, len(ecg)), (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-6)).astype(np.float32)

    # 5. Extract Embeddings
    print("[*] Generating contextual patch embeddings via SetTransformer...")
    channel_embs = []
    mask_chunk = torch.zeros((N_CHUNKS, 1), dtype=torch.bool).to(device)
    with torch.no_grad():
        for sig in [bas, resp, ekg]:
            chunks = torch.tensor(sig.reshape(N_CHUNKS, 1, CHUNK_LEN), dtype=torch.float32).to(device)
            out_base = base_model(chunks, mask_chunk)
            channel_embs.append(out_base[1].reshape(N_TOKENS, 128))
    channel_embs.append(torch.zeros((N_TOKENS, 128), dtype=torch.float32).to(device))

    x_4ch = torch.stack(channel_embs, dim=0).unsqueeze(0).to(device)
    mask_4ch = torch.zeros((1, 4, N_TOKENS), dtype=torch.bool).to(device)
    mask_4ch[:, 3, :] = True

    # 6. Sleep Staging
    print("[*] Running Stage 2A Sleep Staging...")
    with torch.no_grad():
        stg_logits, _ = stg_model(x_4ch, mask_4ch)
        raw_logits = stg_logits.squeeze(0).cpu().numpy()

    n_epochs = N_TOKENS // 6
    ep_logits = raw_logits.reshape(n_epochs, 6, 5).mean(axis=1)
    calibrated_logits = ep_logits - ep_logits.mean(axis=0)
    calibrated_probs = np.exp(calibrated_logits) / np.sum(np.exp(calibrated_logits), axis=-1, keepdims=True)
    preds = np.argmax(calibrated_logits, axis=-1)

    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    pred_stages = [stage_names[i] for i in preds]

    df_staging = pd.DataFrame({
        'epoch_id': range(n_epochs),
        'time_hours': np.round(np.arange(n_epochs) * 30.0 / 3600.0, 4),
        'sleepfm_stage': pred_stages,
        'p_wake': np.round(calibrated_probs[:, 0], 4),
        'p_n1': np.round(calibrated_probs[:, 1], 4),
        'p_n2': np.round(calibrated_probs[:, 2], 4),
        'p_n3': np.round(calibrated_probs[:, 3], 4),
        'p_rem': np.round(calibrated_probs[:, 4], 4)
    })
    df_staging.to_csv('sleepfm_staging_predictions.csv', index=False)

    stage_map = {'Wake': 4, 'REM': 3, 'N1': 2, 'N2': 1, 'N3': 0}
    plt.figure(figsize=(14, 4))
    plt.step(df_staging['time_hours'], [stage_map[s] for s in pred_stages], where='post', color='#1f77b4', lw=1.2)
    plt.yticks([0, 1, 2, 3, 4], ['N3 (Deep)', 'N2', 'N1', 'REM', 'Wake'])
    plt.xlabel('Time (Hours)')
    plt.ylabel('Sleep Stage')
    plt.title('SleepFM Multi-Modal Clinical Hypnogram')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('hypnogram.png', dpi=300)
    plt.close()

    # 7. Disease Hazards
    print("[*] Running Stage 2B Multi-Condition Hazard Heads...")
    norm_age = (args.age - 60.0) / 12.0
    demo_tensor = torch.tensor([[norm_age, float(args.sex)]], dtype=torch.float32).to(device)
    with torch.no_grad():
        hazards = diag_model(x_4ch, mask_4ch, demo_tensor)
        hazard_ratios = torch.exp(hazards).squeeze(0).cpu().numpy()

    haz_df = pd.DataFrame({
        'disease_index': range(len(hazard_ratios)),
        'log_hazard': np.round(hazards.squeeze(0).cpu().numpy(), 4),
        'relative_hazard_ratio': np.round(hazard_ratios, 4)
    })
    
    labels = pd.read_csv('sleepfm/configs/label_mapping.csv')
    report_df = haz_df.copy()
    for c in labels.columns:
        report_df[c] = labels[c].values[:len(report_df)]
    report_df.to_csv('sleepfm_labeled_clinical_report.csv', index=False)

    # 8. Diagnostic Impression & Dashboard Export
    impression = evaluate_clinical_impression()
    build_dashboard(args.output_html)

    print("\n" + "=" * 70)
    print("                    PIPELINE INFERENCE COMPLETE")
    print("=" * 70)
    print("  CLINICAL IMPRESSION:")
    print(f"  {impression}")
    print("-" * 70)
    print(f"  Exported Artifacts:")
    print(f"    - Predictions : sleepfm_staging_predictions.csv")
    print(f"    - Hazards     : sleepfm_labeled_clinical_report.csv")
    print(f"    - Hypnogram   : hypnogram.png")
    print(f"    - Dashboard   : {args.output_html}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
