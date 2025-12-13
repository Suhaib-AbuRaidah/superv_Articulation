import matplotlib.pyplot as plt
import numpy as np

# Example structure
category_metrics = {
    "Robotic Arm": {
        "acc": 0.91, "prec": 0.95, "recall": 0.91, "f1": 0.93,
        "acc_": 1.0, "f1_": 1.0,
        "revolute_cosine": 0.91, "angle_err_deg_r": 11.03,
        "prismatic_cosine": 0, "angle_err_deg_p": 0
    },
    # # Add more categories to test color variation
    # "Cabinet": {
    #     "acc": 0.88, "prec": 0.82, "recall": 0.82, "f1": 0.82,
    #     "acc_": 0.89, "f1_": 0.84,
    #     "revolute_cosine": 0.91, "angle_err_deg_r": 12.5,
    #     "prismatic_cosine": 0.88, "angle_err_deg_p": 3.8
    # },
    # "Refrigerator": {
    #     "acc": 0.92, "prec": 0.90, "recall": 0.89, "f1": 0.89,
    #     "acc_": 0.9, "f1_": 0.85,
    #     "revolute_cosine": 0.93, "angle_err_deg_r": 8.3,
    #     "prismatic_cosine": 0.90, "angle_err_deg_p": 0.1
    # }
}

# Metrics to plot
metric_groups = {
    "Parts Connections": ["acc", "prec", "recall", "f1"],
    "Joint Type": ["acc_", "f1_"],
    "Revolute Axis": ["angle_err_deg_r"],
    "Prismatic Axis": ["angle_err_deg_p"]
}

# --- Plot ---
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
axs = axs.flatten()

colors = plt.cm.tab10.colors  # 10 distinct colors

for ax, (group_name, metric_names) in zip(axs, metric_groups.items()):
    categories = list(category_metrics.keys())
    x = np.arange(len(metric_names))  # metrics on x-axis
    width = 0.15 / len(categories)  # keep groups compact

    for i, category in enumerate(categories):
        if len(metric_names)==1:
            values = [category_metrics[category][m] for m in metric_names]
            ax.bar(0, values, 0.1, color=colors)
            ax.set_xlim(-0.5, 0.5)
        values = [category_metrics[category][m] for m in metric_names]
        ax.bar(x + i * width, values, width, color=colors)
        
    ax.set_title(group_name, fontsize=14)
    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=0, fontsize=12, fontweight="bold")
    ax.set_ylabel("degree" if "angle_err_deg" in metric_names[0] else "score", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.suptitle("Model Evaluation Metrics for The Robotic Arm Category", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("evaluation_metrics.svg")
plt.show()
