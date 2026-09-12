import pandas as pd
import numpy as np

df = pd.read_csv('sleepfm_labeled_disease_risk_report.csv')

# Compute logit from raw sigmoid or rank-order percentiles
# Standardize risk scores to Z-scores across all 1,065 outcomes
mean_r = df['predicted_risk_index'].mean()
std_r = df['predicted_risk_index'].std()
df['relative_z_score'] = (df['predicted_risk_index'] - mean_r) / (std_r + 1e-9)

# Filter for physiologically plausible cardiopulmonary & sleep phenotypes
cardio_respiratory_mask = df['phecode'].str.startswith(('327', '411', '415', '427', '428', '440', '475', '496', '785', '786', 'Hypotension', 'Chest Pain'), na=False)

calibrated_report = df[cardio_respiratory_mask].sort_values(by='predicted_risk_index', ascending=False)
print("=== Calibrated Cardiopulmonary & Sleep Predictions ===")
print(calibrated_report[['phecode', 'phenotype', 'predicted_risk_index', 'relative_z_score']].head(10).to_string(index=False))
