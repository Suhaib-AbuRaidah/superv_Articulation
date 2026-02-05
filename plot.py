import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Load data

df = pd.read_csv(f"run-2025-10-16 15_17_10-tag-Val_Loss.csv")
df["Smoothed"] = gaussian_filter1d(df["Value"], sigma=1)
# Plot
plt.figure(figsize=(12,8))
color = "orange"
plt.plot(df["Step"], df["Smoothed"],linestyle='-',color=color, label='Loss')
plt.xlabel("Epoch",fontsize=20)
plt.ylabel("Loss Value",fontsize=20)
plt.title(f"Validation Loss over Epochs",fontsize=26)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"Validation_Loss_over_Epochs.svg")
plt.show()
