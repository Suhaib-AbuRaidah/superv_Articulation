import torch
import numpy as np
import sys
import tqdm

sys.path.append('/home/suhaib/superv_Articulation')

from utilis.dataset2 import Segmentation_Dataset
from PointNet2.models.pointnet2_part_seg_msg import get_model
from torch.utils.data import DataLoader

# -----------------------
# Config
# -----------------------
NUM_CLASSES = 10
BATCH_SIZE = 8
CHECKPOINT = "./Segmentation/pre_trained_models_seg/2026-01-13_21-53-12_cabinet/chkpt_best_model_train.pth"

DATA_PATH = "../Ditto/Articulated_object_simulation-main/data/Shape2Motion_gcn/cabinet/train/scenes/*.npz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load dataset
# -----------------------
dataset = Segmentation_Dataset(DATA_PATH, device)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# Load model
# -----------------------
model = get_model(num_classes=NUM_CLASSES, normal_channel=False).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

print(f"Loaded checkpoint: {CHECKPOINT}")

# -----------------------
# Metric accumulators
# -----------------------
total_correct = 0
total_seen = 0

class_seen = np.zeros(NUM_CLASSES)
class_correct = np.zeros(NUM_CLASSES)
class_union = np.zeros(NUM_CLASSES)

# -----------------------
# Evaluation loop
# -----------------------
with torch.no_grad():
    pbar = tqdm.tqdm(dataloader, desc="Evaluating", dynamic_ncols=True)
    for data in pbar:
        pc_start, seg_labels, _, _ = data

        pc_start = pc_start.to(device)            # (B, N, 3)
        seg_labels = seg_labels.to(device)        # (B, N)

        preds, _ = model(pc_start.transpose(2, 1), None)
        # preds: (B, N, C)

        pred_labels = preds.argmax(dim=-1)        # (B, N)
        # Flatten
        pred_labels = pred_labels.view(-1)
        seg_labels = seg_labels.view(-1)

        # Overall accuracy
        correct = (pred_labels == seg_labels).sum().item()
        total_correct += correct
        total_seen += seg_labels.numel()

        # Per-class stats
        for cls in range(NUM_CLASSES):
            pred_mask = (pred_labels == cls)
            gt_mask   = (seg_labels == cls)

            class_seen[cls] += gt_mask.sum().item()
            class_correct[cls] += (pred_mask & gt_mask).sum().item()
            class_union[cls] += (pred_mask | gt_mask).sum().item()

# -----------------------
# Final metrics
# -----------------------
overall_acc = total_correct / total_seen

class_acc = class_correct / (class_seen + 1e-6)
mean_class_acc = class_acc.mean()

class_iou = class_correct / (class_union + 1e-6)
mean_iou = class_iou.mean()

# -----------------------
# Print results
# -----------------------
print("\n========== Evaluation Results ==========")
print(f"Overall Accuracy (OA): {overall_acc:.4f}")
print(f"Mean Class Accuracy (mAcc): {mean_class_acc:.4f}")
print(f"Mean IoU (mIoU): {mean_iou:.4f}\n")

for cls in range(NUM_CLASSES):
    print(
        f"Class {cls:02d} | "
        f"Acc: {class_acc[cls]:.4f} | "
        f"IoU: {class_iou[cls]:.4f} | "
        f"Seen: {int(class_seen[cls])}"
    )
