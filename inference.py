import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sleepfm.models.models import SetTransformer, SleepEventLSTMClassifier

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using compute device: {device}")

    # 1. Initialize Stage 1 Foundation Backbone (SetTransformer)
    base_cfg_path = "sleepfm/checkpoints/model_base/config.json"
    base_ckpt_path = "sleepfm/checkpoints/model_base/best.pt"

    with open(base_cfg_path) as f:
        base_cfg = json.load(f)

    print("Initializing Stage 1 SetTransformer...")
    base_model = SetTransformer(
        in_channels=base_cfg.get("in_channels", 1),
        patch_size=base_cfg.get("patch_size", 640),
        embed_dim=base_cfg.get("embed_dim", 128),
        num_heads=base_cfg.get("num_heads", 8),
        num_layers=base_cfg.get("num_layers", 6),
        pooling_head=base_cfg.get("pooling_head", 8),
        dropout=base_cfg.get("dropout", 0.3)
    ).to(device)

    base_ckpt = torch.load(base_ckpt_path, map_location=device)
    raw_dict = base_ckpt.get("state_dict", base_ckpt.get("model_state_dict", base_ckpt))
    base_model.load_state_dict({k.replace("module.", ""): v for k, v in raw_dict.items()}, strict=False)
    base_model.eval()

    # 2. Initialize Stage 2 Staging Classifier
    stg_cfg_path = "sleepfm/checkpoints/model_sleep_staging/config.json"
    stg_ckpt_path = "sleepfm/checkpoints/model_sleep_staging/best.pth"

    with open(stg_cfg_path) as f:
        stg_cfg = json.load(f)

    params = stg_cfg["model_params"]
    print("Initializing Stage 2 SleepEventLSTMClassifier...")
    stg_model = SleepEventLSTMClassifier(
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        num_classes=params["num_classes"],
        pooling_head=params["pooling_head"],
        dropout=params["dropout"],
        max_seq_length=params["max_seq_length"]
    ).to(device)

    stg_ckpt = torch.load(stg_ckpt_path, map_location=device)
    stg_sd = stg_ckpt.get("state_dict", stg_ckpt.get("model_state_dict", stg_ckpt))
    stg_model.load_state_dict({k.replace("module.", ""): v for k, v in stg_sd.items()}, strict=False)
    stg_model.eval()

    # 3. Standardize and Segment Signals into 5-Minute Chunks
    CHUNK_LEN = 38400   # 5 mins * 60 sec * 128 Hz
    N_CHUNKS = 96       # 8 hours = 96 chunks
    TOTAL_SAMPLES = N_CHUNKS * CHUNK_LEN
    N_TOKENS = N_CHUNKS * 60  # 5,760 tokens
    t_target = np.linspace(0, 1, TOTAL_SAMPLES)

    # BAS (IMU Actigraphy)
    mpu_df = pd.read_csv('mpu6050_full_day.csv')
    mag = pd.to_numeric(mpu_df['magnitude'], errors='coerce').bfill().ffill().values
    act = np.abs(mag - np.median(mag))
    bas_res = np.interp(t_target, np.linspace(0, 1, len(act)), (act - np.mean(act)) / (np.std(act) + 1e-6)).astype(np.float32)

    # RESP (PPG/IR Respiratory Envelope)
    spo2_df = pd.read_csv('max30100_full_day.csv', header=None, names=['time', 'hr', 'spo2', 'ir', 'red', 'status'])
    ir = pd.to_numeric(spo2_df['ir'], errors='coerce').bfill().ffill().values
    w = max(int(len(ir) / 28800 * 2.5), 5)
    resp_slow = pd.Series(ir).rolling(w, min_periods=1, center=True).mean().values
    resp_b = resp_slow - pd.Series(resp_slow).rolling(w * 10, min_periods=1, center=True).mean().values
    resp_res = np.interp(t_target, np.linspace(0, 1, len(resp_b)), (resp_b - np.mean(resp_b)) / (np.std(resp_b) + 1e-6)).astype(np.float32)

    # EKG (Filtered ECG)
    ecg_df = pd.read_csv('ecg.csv')
    ecg = pd.to_numeric(ecg_df['filtered_v'], errors='coerce').bfill().ffill().values
    ekg_res = np.interp(t_target, np.linspace(0, 1, len(ecg)), (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-6)).astype(np.float32)

    # 4. Extract 5-Minute Contextual Embeddings
    print("Generating contextual embeddings across 5-minute attention windows...")
    channel_embs = []
    with torch.no_grad():
        for sig in [bas_res, resp_res, ekg_res]:
            chunks = torch.tensor(sig.reshape(N_CHUNKS, 1, CHUNK_LEN), dtype=torch.float32).to(device)
            out = base_model(chunks, torch.zeros((N_CHUNKS, 1), dtype=torch.bool).to(device))
            channel_embs.append(out[1].reshape(N_TOKENS, 128))

    # Masked dummy channel 3 for missing EMG
    channel_embs.append(torch.zeros((N_TOKENS, 128), dtype=torch.float32).to(device))

    x_4ch = torch.stack(channel_embs, dim=0).unsqueeze(0).to(device)
    mask_4ch = torch.zeros((1, 4, N_TOKENS), dtype=torch.bool).to(device)
    mask_4ch[:, 3, :] = True

    # 5. Temporal Staging Classification
    print("Running 4-channel masked classification...")
    with torch.no_grad():
        logits, _ = stg_model(x_4ch, mask_4ch)
        raw_logits = logits.squeeze(0).cpu().numpy()

    # Aggregate into 30-Second Clinical Epochs
    tokens_per_epoch = 6
    n_epochs = N_TOKENS // tokens_per_epoch
    ep_logits = raw_logits.reshape(n_epochs, tokens_per_epoch, 5).mean(axis=1)

    # Calibrate logits against channel-bias shift
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
    print("Saved: sleepfm_staging_predictions.csv")

    # 6. Plot Hypnogram
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
    print("Saved: hypnogram.png")

    # 7. Summary Breakdown
    stg_dist = df_staging['sleepfm_stage'].value_counts(normalize=True) * 100
    print("\n" + "=" * 55)
    print("        AASM 5-STAGE SLEEP ARCHITECTURE")
    print("=" * 55)
    for stg in stage_names:
        print(f"  {stg:<6}: {stg_dist.get(stg, 0.0):6.2f}%")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
