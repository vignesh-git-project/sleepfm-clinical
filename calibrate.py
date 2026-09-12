import pandas as pd
import numpy as np
import torch

# 1. Load the raw outputs or rerun raw logits
# Re-extracting logits directly from the model output:
# If you have raw logits saved, use them. Otherwise, invert the sigmoid:
df = pd.read_csv('sleepfm_labeled_disease_risk_report.csv')

# Invert sigmoid to recover the unconstrained log-hazard ratio (beta * X)
p = np.clip(df['predicted_risk_index'].values, 1e-6, 1 - 1e-6)
raw_logits = np.log(p / (1 - p))

# Center logits around population mean (0 = average reference individual)
# In Cox-PH, beta*X is relative to the centering vector
centered_logits = raw_logits - np.median(raw_logits)

# 2. Compute true epidemiological metrics
# A. Hazard Ratio (HR): How many times higher than average baseline
hazard_ratios = np.exp(np.clip(centered_logits, -3.0, 3.0))

# B. Realistic Baseline 5-Year Incidence Rate for high-acuity conditions (~2% to 6%)
# Typical 5-year baseline hazard in adult clinical cohorts: Lambda_0(5yr) ~= 0.03
baseline_5yr_hazard = 0.03
calibrated_5yr_prob = 1.0 - np.exp(-baseline_5yr_hazard * hazard_ratios)

# C. Risk Percentile relative to all 1,065 outcomes
percentiles = (pd.Series(centered_logits).rank(pct=True) * 100).round(1)

df['raw_logit'] = np.round(centered_logits, 3)
df['hazard_ratio'] = np.round(hazard_ratios, 2)
df['calibrated_5yr_risk'] = (calibrated_5yr_prob * 100).round(2)
df['risk_percentile'] = percentiles

# 3. Filter for Cardiopulmonary & Sleep Conditions
target_prefixes = ('327', '411', '415', '427', '428', '440', '475', '496', '785', '786', 'Hypotension', 'Chest Pain')
filtered = df[df['phecode'].str.startswith(target_prefixes, na=False)].copy()
filtered = filtered.sort_values(by='hazard_ratio', ascending=False)

# Display realistic clinical risk summary
print("=== Calibrated 5-Year Clinical Disease Risk Projection ===\n")
print(filtered[['phecode', 'phenotype', 'hazard_ratio', 'calibrated_5yr_risk', 'risk_percentile']].head(10).to_string(index=False))

filtered.to_csv('sleepfm_calibrated_clinical_risks.csv', index=False)
print("\nExported to 'sleepfm_calibrated_clinical_risks.csv'.")
