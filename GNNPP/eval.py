import matplotlib.pyplot as plt
import numpy as np

# Example structure
category_metrics = {
    "Cabinet": {
        "acc_after": 0.906, "acc_before": 0.91, "f1_after": 0.947, "f1_before": 0.935,
        "acc_after_": 1.0, "acc_before_": 1.0, "f1_after_": 1.0, "f1_before_": 1.0,
        "Rev_angle_err_deg_after": 11.6, "Rev_angle_err_deg_before": 17.6,
        "Pri_angle_err_deg_after": 0.0, "Pri_angle_err_deg_before": 0.0
    },
}

# Metrics to plot
metric_groups = {
    "Parts Connections": [
        ["acc_before", "acc_after"],
        ["f1_before", "f1_after"]
    ],
    "Joint Type": [
        ["acc_before_", "acc_after_"],
        ["f1_before_", "f1_after_"]
    ],
    "Revolute Axis Parameters": [
        ["Rev_angle_err_deg_before", "Rev_angle_err_deg_after"]
    ],
    "Prismatic Axis Parameters": [
        ["Pri_angle_err_deg_before", "Pri_angle_err_deg_after"]
    ]
}

# --- Plot ---
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
axs = axs.flatten()

colors = plt.cm.tab10.colors

for ax, (group_name, metric_pairs) in zip(axs, metric_groups.items()):
    categories = list(category_metrics.keys())
    n_categories = len(categories)
    n_pairs = len(metric_pairs)
    
    # Width settings
    bar_width = 0.35  # Width of each individual bar
    group_spacing = 1.0  # Space between groups of pairs
    gap_before_after = 0.1  # Gap between before/after bars
    
    # Calculate positions
    x = np.arange(n_pairs) * group_spacing
    
    for i, category in enumerate(categories):
        color = colors[i % len(colors)]
        
        for pair_idx, (metric_before, metric_after) in enumerate(metric_pairs):
            # Calculate positions for this category's bars
            # Offset each category by (i - n_categories/2) to center them
            offset = (i - (n_categories - 1)/2) * (2 * bar_width + gap_before_after)
            
            before_x = x[pair_idx] + offset - bar_width/2
            after_x = x[pair_idx] + offset + bar_width/2
            
            # Get values
            val_before = category_metrics[category][metric_before]
            val_after = category_metrics[category][metric_after]
            if metric_before.endswith("angle_err_deg_before"):
                before_x = -0.05
                after_x = 0.05
                # Plot bars
                ax.bar(before_x, val_before, 0.1, 
                    color=colors[0], alpha=1.0, edgecolor='black', linewidth=1,
                    label=f'{category} (Before)' if pair_idx == 0 else "")
                ax.bar(after_x, val_after, 0.1, 
                    color=colors[1], alpha=1.0, edgecolor='black', linewidth=1,
                    label=f'{category} (After)' if pair_idx == 0 else "")
                ax.set_xlim(-0.3, 0.3)
                ax.set_ylim(bottom=0, top=19)
            else:
                ax.bar(before_x, val_before, bar_width, 
                    color=colors[0], alpha=1.0, edgecolor='black', linewidth=1,
                    label=f'{category} (Before)' if pair_idx == 0 else "")
                ax.bar(after_x, val_after, bar_width, 
                    color=colors[1], alpha=1.0, edgecolor='black', linewidth=1,
                    label=f'{category} (After)' if pair_idx == 0 else "")
                ax.set_ylim(bottom=0, top=1.05)
    # Set x-ticks at the center of each pair group
    ax.set_xticks(x)
    
    # Create labels
    tick_labels = []
    for metric_before, metric_after in metric_pairs:
        metric_name = metric_before.replace('_before', '').replace('_after', '')
        if metric_name.endswith('_'):
            metric_name = metric_name[:-1]
        if metric_name.startswith('Rev_'):
            metric_name = 'Revolute Axis\nError (°)'
        elif metric_name.startswith('Pri_'):
            metric_name = 'Prismatic Axis\nError (°)'
        elif metric_name == 'acc':
            metric_name = 'Accuracy'
        elif metric_name == 'f1':
            metric_name = 'F1 Score'
        elif metric_name == 'acc_':
            metric_name = 'Accuracy'
        elif metric_name == 'f1_':
            metric_name = 'F1 Score'
        tick_labels.append(metric_name)
    
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=11, fontweight="bold")
    ax.set_ylabel("Degree" if "angle_err_deg" in metric_pairs[0][0] else "Score", fontsize=12)
    ax.set_title(group_name, fontsize=16, fontweight="bold")
    
    # Add grid
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Add legend
    if n_categories == 1:  # Single category
        handles = [
            plt.Rectangle((0,0),1,1, facecolor=colors[0], alpha=1.0, edgecolor='black', label='Before'),
            plt.Rectangle((0,0),1,1, facecolor=colors[1], alpha=1.0, edgecolor='black', label='After')
        ]
        ax.legend(handles=handles, fontsize=10, loc='upper right')
    else:
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l and l not in unique:
                unique[l] = h
        ax.legend(unique.values(), unique.keys(), fontsize=10, loc='upper right')

plt.suptitle("Model Evaluation Metrics for The Robotic Arm Category", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("evaluation_metrics.svg", dpi=300, bbox_inches='tight')
plt.show()