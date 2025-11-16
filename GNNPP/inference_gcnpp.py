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
        
        for joint in range(num_joints):
            mask_start = data[f'pc_seg_start_{joint}']
            mask_start_list.append(mask_start)
            joint_type = data[f'joint_type_{joint}']
            joint_type_list.append(int(joint_type))
            screw_axis = data[f'screw_axis_{joint}']
            screw_axis_list.append(screw_axis)
            # screw_moment = data[f'screw_momemt_{joint}']
            # screw_point = np.cross(screw_axis, screw_moment)
            # screw_point_list.append(screw_point)


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
        joints_screw_axis = torch.tensor(np.array(screw_axis_list), dtype=torch.float32).cuda()
        data_tuple = (pc_start_ds, parts_list, adjacency_matrix, parts_conne_gt, joints_type, joints_screw_axis, pc_start,mask_start_list,screw_point_list)

        data_list.append(data_tuple)
np.random.seed()
index = np.random.randint(0, len(data_list))
index = 182
print(index)
data = data_list[index]
pc_start_ds = data[0]
parts_list = data[1]
adj = data[2]
parts_conne_gt = data[3]
joints_type = data[4]
joints_screw_axis = data[5]
pc_start = data[6]
mask_start_list = data[7]
joints_screw_point = data[8]


adj = adj.squeeze(0)
parts_connections = parts_conne_gt.squeeze(0)

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

with torch.no_grad():
    edges_conne_pred, joint_type_pred, revolute_para_pred, prismatic_para_pred, (src, dst) = model(parts_list, adj)

print(revolute_para_pred.shape)

edges_conne_pred = (torch.sigmoid(edges_conne_pred)>0.5).float()
joint_mask = parts_conne_gt[src, dst].float().unsqueeze(1) > 0 # boolean mask of edges that exist
joint_mask=joint_mask.squeeze()
joint_type_pred_valid = (torch.sigmoid(joint_type_pred[joint_mask])>0.5).float()
joint_type_list_gt = joints_type.reshape(-1,1)
revolute_mask = (joint_type_pred_valid == 0).squeeze()  # 0 = revolute
revolute_screw_axis_pred = revolute_para_pred[:,:3][joint_mask].squeeze().view(-1, 3)
print(revolute_screw_axis_pred.shape)
revolute_pivot_point_pred = revolute_para_pred[:,3:][joint_mask].squeeze().view(-1, 3)
prismatic_screw_axis_pred = prismatic_para_pred[joint_mask].squeeze().view(-1,3)
axes_pred = torch.cat([revolute_screw_axis_pred[revolute_mask], prismatic_screw_axis_pred[~revolute_mask]], dim=0).squeeze(0)

targets = parts_conne_gt[src, dst].float()  # [num_edges, 1]
torch.set_printoptions(precision=1)
print(src)
print(dst)
print(f"Edge connections pred: \n{edges_conne_pred}")
print(f"adj: \n{adj}")
print(f"Edge connections gt: \n{parts_conne_gt}")
print(f"Joints type pred: \n{joint_type_pred_valid}")
print(f"Joints type gt: \n{joint_type_list_gt}")
# print(f"Revolute screw axis pred: \n{revolute_screw_axis_pred}")
# print(f"Prismatic screw axis pred: \n{prismatic_screw_axis_pred}")
print(f"Revolute pivot point pred: \n{revolute_pivot_point_pred}")
# axes_pred[4,0]= 4.5e-2
# axes_pred[4,2] = 9.8e-1
axes_pred = F.normalize(axes_pred, dim=1)
print(f"Axes pred: \n{axes_pred}")
print(f"Screw axis gt: \n{joints_screw_axis}")

adj_pred = joint_pred_to_matrix(edges_conne_pred, src, dst, adj.shape[0])
print(adj_pred)

# joint_types_pred = torch.tensor([[1.],[1.],[1.]], device='cuda')
# screw_axes_pred = torch.tensor([[ 0.8638,  0.0046, -0.1133],
#                                 [ 0.9865,  0.0055, -0.1334],
#                                 [ 0.7931,  0.0043, -0.1032]], device='cuda')

pcd = masked_pc(pc_start, mask_start_list)
img = pc_to_img(pcd)
visualize_articulated_graph(adj_pred,adj,joint_type_pred_valid, axes_pred,img)