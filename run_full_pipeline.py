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

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[*] Initializing SleepFM Clinical Engine on device: {device}")

    # =========================================================================
    # 1. Load Stage 1 Foundation Backbone (SetTransformer)
    # =========================================================================
    print("[1/4] Loading Stage 1 SetTransformer Backbone...")
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

    # =========================================================================
    # 2. Load Stage 2A Staging Model (SleepEventLSTMClassifier)
    # =========================================================================
    print("[2/4] Loading Stage 2A Sleep Staging Classifier...")
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

    # =========================================================================
    # 3. Load Stage 2B Diagnostic Head (DiagnosisFinetuneFullLSTMCOXPHWithDemo)
    # =========================================================================
    print("[3/4] Loading Stage 2B Cox-PH Diagnostic Risk Head...")
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

    # =========================================================================
    # 4. Ingest, Filter, and Resample Multi-Modal Sensors
    # =========================================================================
    CHUNK_LEN = 38400   # 5 min * 60 s * 128 Hz
    N_CHUNKS = 96       # 8-hour total timeline = 96 chunks
    TOTAL_SAMPLES = N_CHUNKS * CHUNK_LEN
    N_TOKENS = N_CHUNKS * 60  # 5,760 tokens
    t_target = np.linspace(0, 1, TOTAL_SAMPLES)

    print(f"[*] Conditioning 8-hour sensor recording ({TOTAL_SAMPLES:,} samples)...")

    # Modality 0: BAS (Actigraphy from MPU6050)
    mpu_df = pd.read_csv('mpu6050_full_day.csv')
    mag = pd.to_numeric(mpu_df['magnitude'], errors='coerce').bfill().ffill().values
    act = np.abs(mag - np.median(mag))
    bas_res = np.interp(t_target, np.linspace(0, 1, len(act)), (act - np.mean(act)) / (np.std(act) + 1e-6)).astype(np.float32)

    # Modality 1: RESP (Respiratory envelope from MAX30100 PPG)
    spo2_df = pd.read_csv('max30100_full_day.csv', header=None, names=['time', 'hr', 'spo2', 'ir', 'red', 'status'])
    ir = pd.to_numeric(spo2_df['ir'], errors='coerce').bfill().ffill().values
    w = max(int(len(ir) / 28800 * 2.5), 5)
    resp_slow = pd.Series(ir).rolling(w, min_periods=1, center=True).mean().values
    resp_b = resp_slow - pd.Series(resp_slow).rolling(w * 10, min_periods=1, center=True).mean().values
    resp_res = np.interp(t_target, np.linspace(0, 1, len(resp_b)), (resp_b - np.mean(resp_b)) / (np.std(resp_b) + 1e-6)).astype(np.float32)

    # Modality 2: EKG (Single-lead filtered ECG)
    ecg_df = pd.read_csv('ecg.csv')
    ecg = pd.to_numeric(ecg_df['filtered_v'], errors='coerce').bfill().ffill().values
    ekg_res = np.interp(t_target, np.linspace(0, 1, len(ecg)), (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-6)).astype(np.float32)

    # =========================================================================
    # 5. Stage 1: Generate Contextual Embeddings
    # =========================================================================
    print("[*] Running Stage 1 SetTransformer inference across 5-minute contextual windows...")
    channel_embs = []
    mask_chunk = torch.zeros((N_CHUNKS, 1), dtype=torch.bool).to(device)

    with torch.no_grad():
        for mod_sig in [bas_res, resp_res, ekg_res]:
            chunks = torch.tensor(mod_sig.reshape(N_CHUNKS, 1, CHUNK_LEN), dtype=torch.float32).to(device)
            out_base = base_model(chunks, mask_chunk)
            channel_embs.append(out_base[1].reshape(N_TOKENS, 128))

    # Masked dummy channel for absent chin EMG
    channel_embs.append(torch.zeros((N_TOKENS, 128), dtype=torch.float32).to(device))

    # Input tensor: (B=1, C=4, S=5760, E=128)
    x_4ch = torch.stack(channel_embs, dim=0).unsqueeze(0).to(device)

    # Mask: False for active channels (BAS, RESP, EKG), True for padded channel (EMG)
    mask_4ch = torch.zeros((1, 4, N_TOKENS), dtype=torch.bool).to(device)
    mask_4ch[:, 3, :] = True

    # =========================================================================
    # 6. Stage 2A: Temporal Sleep Staging Execution
    # =========================================================================
    print("[*] Running Stage 2A Sleep Staging Classifier...")
    with torch.no_grad():
        stg_logits, _ = stg_model(x_4ch, mask_4ch)
        raw_logits = stg_logits.squeeze(0).cpu().numpy()

    tokens_per_epoch = 6
    n_epochs = N_TOKENS // tokens_per_epoch
    ep_logits = raw_logits.reshape(n_epochs, tokens_per_epoch, 5).mean(axis=1)

    # Center logits against absent chin EMG baseline prior
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
    print("  -> Exported: sleepfm_staging_predictions.csv")

    # Generate Clinical Hypnogram Plot
    stage_map = {'Wake': 4, 'REM': 3, 'N1': 2, 'N2': 1, 'N3': 0}
    y_vals = [stage_map[s] for s in pred_stages]

    plt.figure(figsize=(14, 4))
    plt.step(df_staging['time_hours'], y_vals, where='post', color='#1f77b4', lw=1.2)
    plt.yticks([0, 1, 2, 3, 4], ['N3 (Deep)', 'N2', 'N1', 'REM', 'Wake'])
    plt.xlabel('Time (Hours)')
    plt.ylabel('Sleep Stage')
    plt.title('SleepFM Multi-Modal Clinical Hypnogram (BAS + RESP + EKG)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('hypnogram.png', dpi=300)
    print("  -> Exported: hypnogram.png")

    # =========================================================================
    # 7. Stage 2B: Multi-Condition Diagnostic Risk Profiling
    # =========================================================================
    print("[*] Running Stage 2B Cox-PH Multi-Phenotype Risk Evaluation...")
    # Demographic priors: Normalized Age = (age - 60) / 12, Sex Binary = (0=Female, 1=Male)
    norm_age = (28.0 - 60.0) / 12.0
    sex_binary = 1.0
    demo_tensor = torch.tensor([[norm_age, sex_binary]], dtype=torch.float32).to(device)

    with torch.no_grad():
        hazards = diag_model(x_4ch, mask_4ch, demo_tensor)
        hazard_ratios = torch.exp(hazards).squeeze(0).cpu().numpy()

    df_hazards = pd.DataFrame({
        'disease_index': range(len(hazard_ratios)),
        'log_hazard': np.round(hazards.squeeze(0).cpu().numpy(), 4),
        'relative_hazard_ratio': np.round(hazard_ratios, 4)
    })
    df_hazards.to_csv('sleepfm_clinical_hazards_1065.csv', index=False)
    print("  -> Exported: sleepfm_clinical_hazards_1065.csv")

    # =========================================================================
    # 8. Clinical Summary Dashboard
    # =========================================================================
    stg_dist = df_staging['sleepfm_stage'].value_counts(normalize=True) * 100
    eff = 100.0 - stg_dist.get('Wake', 0.0)

    print("\n" + "=" * 65)
    print("           SLEEPFM MULTI-MODAL CLINICAL SUMMARY")
    print("=" * 65)
    print("  AASM Sleep Architecture:")
    for stg in stage_names:
        print(f"    {stg:<6}: {stg_dist.get(stg, 0.0):6.2f}%")
    print(f"  Overall Sleep Efficiency: {eff:6.2f}%")
    print("-" * 65)
    print("  Longitudinal Diagnostic Projection (1,065 Outcomes):")
    print(f"    Mean Hazard Ratio      : {hazard_ratios.mean():.3f}")
    print(f"    Hazard Ratio Std Dev   : {hazard_ratios.std():.3f}")
    print(f"    Min / Max Hazard Ratio : {hazard_ratios.min():.3f} / {hazard_ratios.max():.3f}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
