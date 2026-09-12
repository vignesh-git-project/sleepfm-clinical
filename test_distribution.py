# test_distribution.py
import json
import torch
import numpy as np
import pandas as pd
from sleepfm.models.models import Tokenizer, SleepEventLSTMClassifier

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

with open("sleepfm/checkpoints/model_sleep_staging/config.json") as f:
    config = json.load(f)

params = config["model_params"]
SAMPLING_FREQ = 128
INPUT_SIZE = 640
EMBED_DIM = params["embed_dim"]

tokenizer = Tokenizer(input_size=INPUT_SIZE, output_size=EMBED_DIM).to(device)
classifier = SleepEventLSTMClassifier(
    embed_dim=EMBED_DIM,
    num_heads=params["num_heads"],
    num_layers=params["num_layers"],
    num_classes=params["num_classes"],
    pooling_head=params["pooling_head"],
    dropout=params["dropout"],
    max_seq_length=params["max_seq_length"]
).to(device)

state_dict = torch.load("sleepfm/checkpoints/model_sleep_staging/best.pth", map_location=device)
if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]
tok_w = {k.replace("tokenizer.", ""): v for k, v in state_dict.items() if k.startswith("tokenizer.")}
cls_w = {k.replace("classifier.", ""): v for k, v in state_dict.items() if not k.startswith("tokenizer.")}
if tok_w: tokenizer.load_state_dict(tok_w, strict=False)
classifier.load_state_dict(cls_w if cls_w else state_dict, strict=False)
tokenizer.eval()
classifier.eval()

# Ingest raw signals
target_len = (28800 * SAMPLING_FREQ // INPUT_SIZE) * INPUT_SIZE
n_tokens = target_len // INPUT_SIZE

spo2_df = pd.read_csv('max30100_full_day.csv', header=None, names=['time', 'hr', 'spo2', 'ir', 'red', 'status'])
# USE CONTINUOUS IR OPTICAL PLETHYSMOGRAM INSTEAD OF DISCRETE INTEGER %
ir_arr = pd.to_numeric(spo2_df['ir'], errors='coerce').bfill().ffill().values
# Bandpass / detrend IR: remove DC baseline offset
ir_ac = ir_arr - pd.Series(ir_arr).rolling(128, min_periods=1, center=True).mean().values

mpu_df = pd.read_csv('mpu6050_full_day.csv')
mag_arr = pd.to_numeric(mpu_df['magnitude'], errors='coerce').bfill().ffill().values
# Motion: only deviations from resting baseline (subtract median resting gravity ~9.8 or baseline)
mag_motion = np.abs(mag_arr - np.median(mag_arr))

ecg_df = pd.read_csv('ecg.csv')
ecg_raw = pd.to_numeric(ecg_df['filtered_v'], errors='coerce').bfill().ffill().values

# Standardize AC components
ir_norm = (ir_ac - np.mean(ir_ac)) / (np.std(ir_ac) + 1e-6)
act_norm = mag_motion / (np.std(mag_motion) + 1e-6)
ecg_norm = (ecg_raw - np.mean(ecg_raw)) / (np.std(ecg_raw) + 1e-6)

t_target = np.linspace(0, 1, target_len)
resp_res = np.interp(t_target, np.linspace(0, 1, len(ir_norm)), ir_norm)
act_res = np.interp(t_target, np.linspace(0, 1, len(act_norm)), act_norm)
ecg_res = np.interp(t_target, np.linspace(0, 1, len(ecg_norm)), ecg_norm)
emg_dummy = np.zeros(target_len, dtype=np.float32)

x_multi = np.stack([act_res, resp_res, ecg_res, emg_dummy], axis=0)
x_tensor = torch.tensor(x_multi, dtype=torch.float32).unsqueeze(0).to(device)

mask_tensor = torch.zeros((1, 4, n_tokens), dtype=torch.bool).to(device)
mask_tensor[:, 3, :] = True

with torch.no_grad():
    tokens = tokenizer(x_tensor)
    out = classifier(tokens, mask_tensor)
    logits = out[0] if isinstance(out, tuple) else out
    raw_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

# 6-token 30-sec epoch pooling
tokens_per_epoch = 6
n_epochs = n_tokens // tokens_per_epoch
epoch_probs = raw_probs[:n_epochs * tokens_per_epoch].reshape(n_epochs, tokens_per_epoch, 5).mean(axis=1)
epoch_pred = np.argmax(epoch_probs, axis=-1)

stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
s_series = pd.Series([stage_names[i] for i in epoch_pred])
print("\n=== STAGE BREAKDOWN WITH IR OPTICAL WAVEFORM ===")
print((s_series.value_counts(normalize=True) * 100).round(2))
