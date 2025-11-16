import numpy as np
import torch
import sys
sys.path.append('/home/suhaib/superv_Articulation')
from GNNPP.gnn_pointnet_network import parts_connection_mlp
import glob
import torch.nn.functional as F
from utilis.Inference_graph import visualize_articulated_graph
import open3d as o3d
import matplotlib.pyplot as plt
import copy

def downsample_pc_masks( points, masks_list1=None, num_points=1024):
    """
    Randomly downsample the point cloud to a fixed size.
    """
    N = points.shape[0]
    if N == 0:
        points = np.zeros((num_points, 3), dtype=np.float32)
        N= num_points

    if N >= num_points:
        np.random.seed(97)
        indices = np.random.choice(N, num_points, replace=False)
    else:
        np.random.seed(97)
        indices = np.random.choice(N, num_points, replace=True)  # pad if too small
    
    if masks_list1 is not None:
        masks_list = copy.deepcopy(masks_list1)
        for i in range(len(masks_list)):
            labels = masks_list[i]
            labels = labels[indices]
            masks_list[i] = labels
        return points[indices], masks_list
    else:
        return points[indices]
    
def pc_to_img(pcd, width=600, height=600, 
              fov=45.0, 
              camera_distance=1.5, 
              elevation_deg=10, 
              azimuth_deg=35):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("pcd", pcd, mat)

    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = np.max(bounds.get_extent())

    # Convert spherical coordinates to cartesian for camera placement
    elev_rad = np.deg2rad(elevation_deg)
    azim_rad = np.deg2rad(azimuth_deg)

    cam_pos = center + camera_distance * extent * np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.sin(elev_rad),
        np.cos(elev_rad) * np.sin(azim_rad)
    ])

    renderer.setup_camera(fov, center, cam_pos, [0, 0, 1])

    img_o3d = renderer.render_to_image()
    img = np.asarray(img_o3d)
    # plt.imshow(img)
    # plt.show()
    return img


def masked_pc(pc, masks_list):
    num_joints = len(masks_list)
    pcd_start = o3d.geometry.PointCloud()
    pcd_start.points = o3d.utility.Vector3dVector(pc)
    color = np.zeros_like(pc)
    # color = create_color_array_grouped(pc_start_list[0].shape[0])
    for i in range(num_joints):
        if i==0:
            color[masks_list[i]] = np.array([1, 0, 0])
        elif i==1:
            color[masks_list[i]] = np.array([0, 1, 0])
        elif i==2:
            color[masks_list[i]] = np.array([0, 0, 1])
        elif i==3:
            color[masks_list[i]] = np.array([1, 1, 0])
        elif i==4:
            color[masks_list[i]] = np.array([1, 0, 1])
        elif i==5:
            color[masks_list[i]] = np.array([0, 1, 1])
        elif i==6:
            color[masks_list[i]] = np.array([0.5, 0.5, 0.5])

    pcd_start.colors = o3d.utility.Vector3dVector(color)
    pcd_start.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([pcd_start])
    return pcd_start


def joint_pred_to_matrix(joint_type_pred, src, dst,num_joints):
    parts_conne = torch.zeros((num_joints, num_joints))
    for i in range(joint_type_pred.shape[0]):
        if joint_type_pred[i] > 0:
            parts_conne[src[i], dst[i]] = 1
            parts_conne[dst[i], src[i]] = 1
    return parts_conne


file_paths = "../Ditto/Articulated_object_simulation-main/data/Shape2Motion_gcn/robotic_arm/scenes/*.npz"
data_list = []
for f in glob.glob(file_paths):
        data = np.load(f, allow_pickle=True)
        mask_start_list = []
        joint_type_list = []
        screw_axis_list = []
        screw_point_list = []
        num_joints = int((len(data)-2)/18)

        pc_start = data[f'pc_start_0']
        adjacency_matrix = data['adj']
        parts_conne_gt = data['parts_conne_gt']

        object_path = str(data['object_path'])
        
        if object_path.split('/')[-3] == 'robotic_arm':
            n = adjacency_matrix.shape[0]
            i = np.arange(n - 1)
            adjacency_matrix[i, i + 1] = 1      

        for joint in range(num_joints):
            mask_start = data[f'pc_seg_start_{joint}']
            mask_start_list.append(mask_start)
            joint_type = data[f'joint_type_{joint}']
            joint_type_list.append(int(joint_type))
            screw_axis = data[f'screw_axis_{joint}']
            screw_axis_list.append(screw_axis)


        base_mask = data['pc_seg_start_base']
        mask_start_list.insert(0, base_mask)


        pc_start_ds, mask_start_list_ds = downsample_pc_masks(pc_start, mask_start_list)
        bound_max = pc_start_ds.max(0)
        bound_min = pc_start_ds.min(0)
        center = (bound_min + bound_max) / 2
        scale = (bound_max - bound_min).max()
        pc_start_ds = (pc_start_ds - center) / scale

        
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32).cuda()
        parts_conne_gt = torch.tensor(parts_conne_gt, dtype=torch.float32).cuda()

        parts_list = []
        for mask in range(num_joints+1):
            part = pc_start_ds[mask_start_list_ds[mask]]
            part = downsample_pc_masks(part)
            part = torch.tensor(part, dtype=torch.float32).cuda().unsqueeze(0)
            parts_list.append(part)

        pc_start_ds = torch.tensor(pc_start_ds, dtype=torch.float32).cuda()
        joints_type = torch.tensor(np.array(joint_type_list), dtype=torch.float32).cuda()
        joints_screw_axis = torch.abs(torch.tensor(np.array(screw_axis_list), dtype=torch.float32).cuda())
        data_tuple = (pc_start_ds, parts_list, adjacency_matrix, parts_conne_gt, joints_type, joints_screw_axis, pc_start,mask_start_list,screw_point_list)

        data_list.append(data_tuple)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Model setup
params = {
    "pointnet_dim": 1024,
    "nlayers": 4,
    "nhidden": 256,
    "out_dim": 128,
    "dropout": 0.3,
    "lamda": 0.5,
    "alpha": 0.1,
    "variant": True,
    "nhidden_mlp": 128,
    "n_class": 1,
}

weights_path = "/home/suhaib/superv_Articulation/pre_trained_models_gcnpp/2025-11-03 23:15:17_mix/chkpt_best_model_val.pth"
model = parts_connection_mlp(**params).cuda()
model.load_state_dict(torch.load(weights_path))
model.eval()

# Containers for metrics
conn_acc, conn_prec, conn_rec, conn_f1 = [], [], [], []
joint_acc, joint_f1 = [], []
revolute_cos, revolute_ang, prismatic_cos, prismatic_ang = [], [], [], []

for idx, data in enumerate(data_list):
    pc_start_ds, parts_list, adj, parts_conne_gt, joints_type, joints_screw_axis, pc_start, mask_start_list, joints_screw_point = data

    adj = adj.squeeze(0)
    parts_connections = parts_conne_gt.squeeze(0)

    with torch.no_grad():
        edges_conne_pred, joint_type_pred, revolute_para_pred, prismatic_para_pred, (src, dst) = model(parts_list, adj)

    # ----- Connection prediction -----
    edges_conne_pred_bin = (torch.sigmoid(edges_conne_pred) > 0.5).float()
    conn_true = parts_conne_gt[src, dst].float().cpu().numpy()
    conn_pred = edges_conne_pred_bin.cpu().numpy()

    # if len(np.unique(conn_true)) > 1:  # avoid metrics crash when all labels are same
    conn_acc.append(accuracy_score(conn_true, conn_pred))
    conn_prec.append(precision_score(conn_true, conn_pred))
    conn_rec.append(recall_score(conn_true, conn_pred))
    conn_f1.append(f1_score(conn_true, conn_pred))

    # ----- Joint type prediction -----
    joint_mask = parts_conne_gt[src, dst].float().unsqueeze(1) > 0
    joint_mask = joint_mask.squeeze()
    if joint_mask.sum() == 0:
        continue

    joint_type_pred_valid = (torch.sigmoid(joint_type_pred[joint_mask]) > 0.5).float()
    joint_true = joints_type.float().cpu().numpy()
    joint_pred = joint_type_pred_valid.cpu().numpy()

    # print(f"Joint True: {joint_true}, Joint Pred: {joint_pred}")
    # if len(np.unique(joint_true)) > 1:
    joint_acc.append(accuracy_score(joint_true, joint_pred))
    joint_f1.append(f1_score(joint_true, joint_pred, average='macro', zero_division=1))

    # ----- Revolute and prismatic parameters -----
    revolute_mask = (joint_type_pred_valid == 0).squeeze()
    prismatic_mask = (joint_type_pred_valid == 1).squeeze()

    if revolute_mask.sum() > 0:
        rev_pred = revolute_para_pred[:, :3][joint_mask][revolute_mask].view(-1, 3)
        rev_pred = F.normalize(rev_pred, dim=1)
        # print(f"Revolute Axis Prediction: {rev_pred}\n")
        rev_gt = joints_screw_axis[revolute_mask].view(-1, 3)
        # print(f"Revolute Axis GT: {rev_gt}\n")
        cos_sim = F.cosine_similarity(rev_pred, rev_gt, dim=1)
        ang_err = torch.acos(torch.clamp(cos_sim, -1, 1)) * 180 / torch.pi
        revolute_cos.append(cos_sim.mean().item())
        revolute_ang.append(ang_err.mean().item())

    if prismatic_mask.sum() > 0:
        pri_pred = prismatic_para_pred[joint_mask][prismatic_mask].view(-1, 3)
        pri_pred = F.normalize(pri_pred, dim=1)
        pri_gt = joints_screw_axis[prismatic_mask].view(-1, 3)
        cos_sim = F.cosine_similarity(pri_pred, pri_gt, dim=1)
        ang_err = torch.acos(torch.clamp(cos_sim, -1, 1)) * 180 / torch.pi
        prismatic_cos.append(cos_sim.mean().item())
        prismatic_ang.append(ang_err.mean().item())

# ----- Summary -----
def mean_or_zero(lst): return np.mean(lst) if len(lst) > 0 else 0

metrics = {
    "conn_acc": mean_or_zero(conn_acc),
    "conn_prec": mean_or_zero(conn_prec),
    "conn_rec": mean_or_zero(conn_rec),
    "conn_f1": mean_or_zero(conn_f1),
    "joint_type_acc": mean_or_zero(joint_acc),
    "joint_type_f1": mean_or_zero(joint_f1),
    "revolute_cosine": mean_or_zero(revolute_cos),
    "revolute_angle_err_deg": mean_or_zero(revolute_ang),
    "prismatic_cosine": mean_or_zero(prismatic_cos),
    "prismatic_angle_err_deg": mean_or_zero(prismatic_ang),
}

print("==== Evaluation Results ====")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")