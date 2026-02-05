import torch
import numpy as np
import open3d as o3d
import sys

sys.path.append('/home/suhaib/superv_Articulation')

from utilis.dataset2 import Segmentation_Dataset
from PointNet2.models.pointnet2_part_seg_msg import get_model

# -----------------------
# Config
# -----------------------
NUM_CLASSES = 10
CHECKPOINT = "./Segmentation/pre_trained_models_seg/2026-01-13_21-53-12_cabinet/chkpt_best_model_train.pth"
DATA_PATH = "../Ditto/Articulated_object_simulation-main/data/Shape2Motion_gcn/cabinet/train/scenes/*.npz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Color map (extend if needed)
# -----------------------

COLOR_MAP = np.array([
    [0.7, 0.7, 0.7],  # class 0 - gray
    [1.0, 0.0, 0.0],  # class 1 - red
    [0.0, 1.0, 0.0],  # class 2 - green
    [0.0, 0.0, 1.0],  # class 3 - blue
    [1.0, 1.0, 0.0],  # class 4 - yellow
    [1.0, 0.0, 1.0],  # class 5 - magenta
    [0.0, 1.0, 1.0],  # class 6 - cyan
    [0.5, 0.5, 0.0],  # class 7 - olive
    [0.5, 0.0, 0.5],  # class 8 - purple
    [0.0, 0.5, 0.5],  # class 9 - teal
])

# np.random.seed(0)
# lst = [np.random.rand(3) for n in range(NUM_CLASSES)]
# COLOR_MAP = np.array(lst)
# -----------------------
# Load model
# -----------------------
model = get_model(num_classes=NUM_CLASSES, normal_channel=False).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

print(f"Loaded checkpoint: {CHECKPOINT}")

# -----------------------
# Load dataset (single samples)
# -----------------------
dataset = Segmentation_Dataset(DATA_PATH, device)

# -----------------------
# Inference loop
# -----------------------
with torch.no_grad():
    for idx in range(len(dataset)):
        pc_start, gt_mask, _, _ = dataset[idx]

        # Ensure shape (1, N, 3)
        if pc_start.ndim == 2:
            pc_start = pc_start.unsqueeze(0)

        pc_start = pc_start.to(device)

        preds, _ = model(pc_start.transpose(2, 1), None)
        # preds: (1, N, C)

        pred_labels = preds.argmax(dim=-1).squeeze(0).cpu().numpy()
        points = pc_start.squeeze(0).cpu().numpy()

        # -----------------------
        # Visualization
        # -----------------------
        colors = COLOR_MAP[pred_labels % len(COLOR_MAP)]

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(points)
        pcd_pred.colors = o3d.utility.Vector3dVector(colors)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points)
        gt_colors = COLOR_MAP[gt_mask.cpu().numpy() % len(COLOR_MAP)]
        pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)

        print(f"Visualizing sample {idx} | Points: {points.shape[0]}")
        print("Prediction:")
        o3d.visualization.draw_geometries([pcd_pred])
        print("Ground Truth:")
        o3d.visualization.draw_geometries([pcd_gt])

        # Press Ctrl+C or close window to stop
