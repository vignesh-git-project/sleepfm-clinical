import json
import torch
import numpy as np
import pandas as pd
from sleepfm.models.models import Tokenizer, DiagnosisFinetuneFullLSTMCOXPHWithDemo

# 1. Device configuration (MPS for M1 Mac)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using compute device: {device}")

# 2. Load Diagnosis Config
config_path = "sleepfm/checkpoints/model_diagnosis/config.json"
with open(config_path) as f:
    config = json.load(f)

params = config["model_params"]
SAMPLING_FREQ = config.get("sampling_freq", 128)
INPUT_SIZE = 640
EMBED_DIM = params["embed_dim"]

# 3. Instantiate Diagnosis Architecture
print("Building SleepFM Diagnosis Architecture...")
tokenizer = Tokenizer(input_size=INPUT_SIZE, output_size=EMBED_DIM).to(device)
diag_model = DiagnosisFinetuneFullLSTMCOXPHWithDemo(
    embed_dim=EMBED_DIM,
    num_heads=params["num_heads"],
    num_layers=params["num_layers"],
    num_classes=params["num_classes"],  # 1065 clinical outcomes
    pooling_head=params["pooling_head"],
    dropout=params["dropout"],
    max_seq_length=params["max_seq_length"]
).to(device)

# 4. Load Weights
checkpoint_path = "sleepfm/checkpoints/model_diagnosis/best.pth"
print(f"Loading weights from {checkpoint_path}...")
state_dict = torch.load(checkpoint_path, map_location=device)

if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]
elif "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

tok_weights = {k.replace("tokenizer.", ""): v for k, v in state_dict.items() if k.startswith("tokenizer.")}
model_weights = {k.replace("model.", ""): v for k, v in state_dict.items() if not k.startswith("tokenizer.")}

if tok_weights:
    tokenizer.load_state_dict(tok_weights, strict=False)
diag_model.load_state_dict(model_weights if model_weights else state_dict, strict=False)

tokenizer.eval()
diag_model.eval()

# 5. Prepare Resampled Night Sensor Data
target_len = 28800 * SAMPLING_FREQ
n_tokens = target_len // INPUT_SIZE
target_len = n_tokens * INPUT_SIZE

spo2_df = pd.read_csv('max30100_full_day.csv', header=None, names=['time', 'hr', 'spo2', 'ir', 'red', 'status'])
spo2_arr = pd.to_numeric(spo2_df['spo2'], errors='coerce').bfill().ffill().values
mpu_df = pd.read_csv('mpu6050_full_day.csv')
mag_arr = mpu_df['magnitude'].values
ecg_df = pd.read_csv('ecg.csv')
ecg_raw = ecg_df['filtered_v'].values

# Standardize signals (Z-score normalization)
spo2_norm = (spo2_arr - np.mean(spo2_arr)) / (np.std(spo2_arr) + 1e-6)
mag_norm = (mag_arr - np.mean(mag_arr)) / (np.std(mag_arr) + 1e-6)
ecg_norm = (ecg_raw - np.mean(ecg_raw)) / (np.std(ecg_raw) + 1e-6)

t_target = np.linspace(0, 1, target_len)
resp_resampled = np.interp(t_target, np.linspace(0, 1, len(spo2_norm)), spo2_norm)
act_resampled = np.interp(t_target, np.linspace(0, 1, len(mag_norm)), mag_norm)
ecg_resampled = np.interp(t_target, np.linspace(0, 1, len(ecg_norm)), ecg_norm)
emg_dummy = np.zeros(target_len, dtype=np.float32)

x_multi = np.stack([act_resampled, resp_resampled, ecg_resampled, emg_dummy], axis=0)
x_tensor = torch.tensor(x_multi, dtype=torch.float32).unsqueeze(0).to(device)

# Modality & Sequence Mask: Shape [B=1, C=4, S=n_tokens]
# 0 (False) = Valid data, 1 (True) = Padded/Masked out
mask_tensor = torch.zeros((1, 4, n_tokens), dtype=torch.bool).to(device)
mask_tensor[:, 3, :] = True  # EMG is dummy/padded

# Demographic vector: [Age, Sex] -> e.g. [30.0, 1.0] (1=Male, 0=Female)
demo_tensor = torch.tensor([[30.0, 1.0]], dtype=torch.float32).to(device)

# 6. Execute Forward Pass
print("Computing 1,065 clinical condition hazard scores...")
with torch.no_grad():
    tokens = tokenizer(x_tensor)
    output = diag_model(tokens, mask_tensor, demo_tensor)
    logits = output[0] if isinstance(output, tuple) else output
    risk_scores = torch.sigmoid(logits).squeeze().cpu().numpy()

# 7. Export Diagnosis Report
diag_results = pd.DataFrame({
    'disease_outcome_id': range(len(risk_scores)),
    'predicted_risk_index': np.round(risk_scores, 4)
}).sort_values(by='predicted_risk_index', ascending=False)

diag_results.to_csv('sleepfm_disease_risk_report.csv', index=False)
print("Saved comprehensive risk report to 'sleepfm_disease_risk_report.csv'.")
print("\nTop 10 Elevated Condition Indices:")
print(diag_results.head(10))
