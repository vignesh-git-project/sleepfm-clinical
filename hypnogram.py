import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sleepfm_staging_predictions.csv')

# Map stages to hypnogram heights
stage_map = {'Wake': 4, 'REM': 3, 'N1': 2, 'N2': 1, 'N3': 0}
df['stage_y'] = df['sleepfm_predicted_stage'].map(stage_map)

plt.figure(figsize=(12, 4))
plt.step(df['time_hours'], df['stage_y'], where='post', color='#1f77b4', lw=1.5)
plt.yticks([0, 1, 2, 3, 4], ['N3 (Deep)', 'N2', 'N1', 'REM', 'Wake'])
plt.xlabel('Time (Hours)')
plt.ylabel('Sleep Stage')
plt.title('Stanford SleepFM Hypnogram')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('hypnogram.png', dpi=300)
print("Hypnogram saved to 'hypnogram.png'.")
