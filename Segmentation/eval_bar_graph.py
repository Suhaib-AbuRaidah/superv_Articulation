import matplotlib.pyplot as plt
import numpy as np

# Example structure
category_metrics_res = {
    "Cabinet": {"Overall Accuracy (OA)": 95.26, "Mean IoU (mIoU)": 77.00},

    "Laptop": {"Overall Accuracy (OA)": 96.61, "Mean IoU (mIoU)": 93.34},

    "Refrigerator": {"Overall Accuracy (OA)": 95.16, "Mean IoU (mIoU)": 85.86 },
    "Washing Machine": {"Overall Accuracy (OA)": 99.24, "Mean IoU (mIoU)": 97.13},

    "Robotic Arm": {"Overall Accuracy (OA)": 89.87, "Mean IoU (mIoU)": 77.49},
}

# --- Plotting code ---
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
colors = plt.cm.tab10.colors

axs = axs.flatten()
ax1 = axs[0]
ax2 = axs[1]

ax1.bar(category_metrics_res.keys(), 
        [category_metrics_res[cat]["Overall Accuracy (OA)"] for cat in category_metrics_res.keys()],
        color=colors[0], edgecolor='black')
ax1.set_ylim(0, 100)
ax1.set_ylabel("Overall Accuracy (OA) (%)", fontsize=12, fontweight="bold")
ax1.set_xticklabels(category_metrics_res.keys(), fontsize=10, fontweight="bold")
ax1.set_title("Overall Accuracy (OA) by Category", fontsize=14, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.5)

ax2.bar(category_metrics_res.keys(), 
        [category_metrics_res[cat]["Mean IoU (mIoU)"] for cat in category_metrics_res.keys()],
        color=colors[1], edgecolor='black')
ax2.set_ylim(0, 100)
ax2.set_ylabel("Mean IoU (mIoU) (%)", fontsize=12, fontweight="bold")
ax2.set_xticklabels(category_metrics_res.keys(), fontsize=10, fontweight="bold")
ax2.set_title("Mean IoU (mIoU) by Category", fontsize=14, fontweight="bold")
ax2.grid(axis="y", linestyle="--", alpha=0.5)


plt.suptitle("Segmentation Evaluation Metrics by Category", fontsize=20, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("segmentation_evaluation_metrics_by_category.png", dpi=300)
plt.show()
# # --- Plot ---
# fig, axs = plt.subplots(2, 2, figsize=(14, 8))
# axs = axs.flatten()

# colors = plt.cm.tab10.colors

# for ax, (group_name, metric_pairs) in zip(axs, metric_groups.items()):
#     categories = list(category_metrics.keys())
#     n_categories = len(categories)
#     n_pairs = len(metric_pairs)
    
#     # Width settings
#     bar_width = 0.35  # Width of each individual bar
#     group_spacing = 1.0  # Space between groups of pairs
#     gap_before_after = 0.1  # Gap between before/after bars
    
#     # Calculate positions
#     x = np.arange(n_pairs) * group_spacing
    
#     for i, category in enumerate(categories):
#         color = colors[i % len(colors)]
        
#         for pair_idx, (metric_before, metric_after) in enumerate(metric_pairs):
#             # Calculate positions for this category's bars
#             # Offset each category by (i - n_categories/2) to center them
#             offset = (i - (n_categories - 1)/2) * (2 * bar_width + gap_before_after)
            
#             before_x = x[pair_idx] + offset - bar_width/2
#             after_x = x[pair_idx] + offset + bar_width/2
            
#             # Get values
#             val_before = category_metrics[category][metric_before]
#             val_after = category_metrics[category][metric_after]
#             if metric_before.endswith("angle_err_deg_before"):
#                 before_x = -0.05
#                 after_x = 0.05
#                 # Plot bars
#                 ax.bar(before_x, val_before, 0.1, 
#                     color=colors[0], alpha=1.0, edgecolor='black', linewidth=1,
#                     label=f'{category} (Before)' if pair_idx == 0 else "")
#                 ax.bar(after_x, val_after, 0.1, 
#                     color=colors[1], alpha=1.0, edgecolor='black', linewidth=1,
#                     label=f'{category} (After)' if pair_idx == 0 else "")
#                 ax.set_xlim(-0.3, 0.3)
#                 ax.set_ylim(bottom=0, top=19)
#             else:
#                 ax.bar(before_x, val_before, bar_width, 
#                     color=colors[0], alpha=1.0, edgecolor='black', linewidth=1,
#                     label=f'{category} (Before)' if pair_idx == 0 else "")
#                 ax.bar(after_x, val_after, bar_width, 
#                     color=colors[1], alpha=1.0, edgecolor='black', linewidth=1,
#                     label=f'{category} (After)' if pair_idx == 0 else "")
#                 ax.set_ylim(bottom=0, top=1.05)
#     # Set x-ticks at the center of each pair group
#     ax.set_xticks(x)
    
#     # Create labels
#     tick_labels = []
#     for metric_before, metric_after in metric_pairs:
#         metric_name = metric_before.replace('_before', '').replace('_after', '')
#         if metric_name.endswith('_'):
#             metric_name = metric_name[:-1]
#         if metric_name.startswith('Rev_'):
#             metric_name = 'Revolute Axis\nError (°)'
#         elif metric_name.startswith('Pri_'):
#             metric_name = 'Prismatic Axis\nError (°)'
#         elif metric_name == 'acc':
#             metric_name = 'Accuracy'
#         elif metric_name == 'f1':
#             metric_name = 'F1 Score'
#         elif metric_name == 'acc_':
#             metric_name = 'Accuracy'
#         elif metric_name == 'f1_':
#             metric_name = 'F1 Score'
#         tick_labels.append(metric_name)
    
#     ax.set_xticklabels(tick_labels, rotation=0, fontsize=11, fontweight="bold")
#     ax.set_ylabel("Degree" if "angle_err_deg" in metric_pairs[0][0] else "Score", fontsize=12)
#     ax.set_title(group_name, fontsize=16, fontweight="bold")
    
#     # Add grid
#     ax.grid(axis="y", linestyle="--", alpha=0.5)
    
#     # Add legend
#     if n_categories == 1:  # Single category
#         handles = [
#             plt.Rectangle((0,0),1,1, facecolor=colors[0], alpha=1.0, edgecolor='black', label='Before'),
#             plt.Rectangle((0,0),1,1, facecolor=colors[1], alpha=1.0, edgecolor='black', label='After')
#         ]
#         ax.legend(handles=handles, fontsize=10, loc='upper right')
#     else:
#         handles, labels = ax.get_legend_handles_labels()
#         unique = {}
#         for h, l in zip(handles, labels):
#             if l and l not in unique:
#                 unique[l] = h
#         ax.legend(unique.values(), unique.keys(), fontsize=10, loc='upper right')

# plt.suptitle("Model Evaluation Metrics for The Robotic Arm Category", fontsize=16, fontweight="bold")
# plt.tight_layout(rect=[0, 0, 1, 0.97])
# plt.savefig("evaluation_metrics.svg", dpi=300, bbox_inches='tight')
# plt.show()