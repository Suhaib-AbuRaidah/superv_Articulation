import torch
from Network_archit import ArticulationNet
from utilis.joint_estimation import aggregate_dense_prediction_r
import numpy as np
import glob
import open3d as o3d


## function definitions 

def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor / ((tensor ** 2).sum(dim, keepdim=True).sqrt() + 1.0e-5)


def downsample_point_cloud(points, labels=None, num_points=1024):
    """
    Randomly downsample the point cloud to a fixed size.
    """
    N = points.shape[0]
    if N >= num_points:
        np.random.seed(97)
        indices = np.random.choice(N, int(num_points), replace=False)
    else:
        np.random.seed(97)
        indices = np.random.choice(N, int(num_points), replace=True)  # pad if too small

    if labels is None:
        return points[indices]
    
    else:
        labels = labels[indices]
        return points[indices], labels
    

    
def batch_perpendicular_line(
    x: np.ndarray, l: np.ndarray, pivot: np.ndarray
) -> np.ndarray:
    """
    x: B * 3
    l: 3
    pivot: 3
    p_l: B * 3
    """
    offset = x - pivot
    p_l = offset.dot(l)[:, np.newaxis] * l[np.newaxis] - offset
    dist = np.sqrt(np.sum(p_l ** 2, axis=-1))
    p_l = p_l / (dist[:, np.newaxis] + 1.0e-5)
    return p_l, dist


def create_axis_arrow(origin, direction, length=0.8, radius=0.0008, color=[0, 0, 1],tr= np.array([0,0,0])):
    """
    Create an Open3D arrow pointing in `direction` starting at `origin`.
    """
    from scipy.spatial.transform import Rotation as R
# Translate to origin
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    end = origin + direction * (length/2 + length/10)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius*10,
        cone_radius=radius*10,
        cylinder_height=length,
        cone_height=length/10
    )
    arrow.paint_uniform_color(color)

    # Rotate arrow from +Z (default) to desired direction
    default_dir = np.array([0, 0, 1])
    rot_axis = np.cross(default_dir, direction)
    if np.linalg.norm(rot_axis) < 1e-6:
        rot_matrix = np.eye(3) if np.dot(default_dir, direction) > 0 else -np.eye(3)
    else:
        rot_angle = np.arccos(np.clip(np.dot(default_dir, direction), -1, 1))
        rot_matrix = R.from_rotvec(rot_angle * rot_axis / np.linalg.norm(rot_axis)).as_matrix()
    arrow.rotate(rot_matrix, center=(0,0,0))

    # Translate to origin
    arrow.translate(end+tr, relative=False)
    return arrow

def mesh_from_pcd(pcd):
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.1, max_nn=30))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.05)

    mesh = mesh.filter_smooth_simple(number_of_iterations=10)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)

    return mesh


def inference(model, pc_start, pc_target,seg_mask_start,center, scale):


    model = model.eval().to(device)

    with torch.no_grad():
        joint_type_logits, joint_param_revolute, joint_param_prismatic = model(pc_start, pc_target)

    joint_type_prob = joint_type_logits.sigmoid().mean()


    print(f"Articulation Parameters:")
    if joint_type_prob.item()< 0.5:
        print(f"\nJoint Type: Revolute\n\n")
    else:
        print(f"Joint Type: Prismatic\n\n")

    if joint_type_prob.item()< 0.5:
        # axis voting
        joint_r_axis = (
            normalize(joint_param_revolute[:, :, :3], -1)[0].cpu().numpy()
        )
        joint_r_axis = joint_r_axis[seg_mask_start==1]
        joint_r_t = joint_param_revolute[:, :, 3][0].cpu().numpy()
        joint_r_t = joint_r_t[seg_mask_start==1]
        joint_r_p2l_vec = (
            normalize(joint_param_revolute[:, :, 4:7], -1)[0].cpu().numpy()
        )
        joint_r_p2l_dist = joint_param_revolute[:, :, 7][0].cpu().numpy()

        pc_start_mobile = pc_start[0][seg_mask_start==1]
        pc_start_mobile = pc_start_mobile[np.newaxis,:].cpu().numpy()

        joint_r_p2l_vec = joint_r_p2l_vec[seg_mask_start==1]

        joint_r_p2l_dist = joint_r_p2l_dist[seg_mask_start==1]

        pivot_point = pc_start_mobile + joint_r_p2l_vec * joint_r_p2l_dist[:, np.newaxis]
        pivot_point = pivot_point.squeeze(0)

        # print(pivot_point.shape)
        (
            joint_axis_pred,
            pivot_point_pred,
            config_pred,
        ) = aggregate_dense_prediction_r(
            joint_r_axis, pivot_point, joint_r_t, method="mean"
        )
        # print(pivot_point_pred.shape)
        # print(f"joint_axis_pred: {joint_axis_pred}\n\npivot_point_pred: {pivot_point_pred}\n\nconfig_pred: {config_pred}\n\n")
        
    # prismatic
    else:
        # axis voting
        joint_p_axis = (
            normalize(joint_param_prismatic[:, :, :3], -1)[0].cpu().numpy()
        )
        joint_axis_pred = joint_p_axis.mean(0)
        joint_p_t = joint_param_prismatic[:, :, 3][0].cpu().numpy()
        config_pred = joint_p_t.mean()
        pc_start_mobile = pc_start[0][seg_mask_start==1]
        pivot_point_pred = pc_start_mobile.mean(0).cpu().numpy()


    return joint_axis_pred, pivot_point_pred, config_pred

file_paths = "../Ditto/Articulated_object_simulation-main/data/syn_local/microwave_valid/scenes/*.npz"

# data load
data_list = []
for f in glob.glob(file_paths):
    data = np.load(f, allow_pickle=True)
    pc_start = data['pc_start']
    pc_target = data['pc_end']
    seg_mask_start = data['pc_seg_start']
    seg_mask_target = data['pc_seg_end']
    joint_type = data['joint_type']
    screw_axis = data['screw_axis']
    screw_moment = data['screw_moment']
    state_start = data['state_start']
    state_target = data['state_end']

    bound_max = np.maximum(pc_start.max(0), pc_target.max(0))
    bound_min = np.minimum(pc_start.min(0), pc_target.min(0))
    center = (bound_min + bound_max) / 2
    scale = (bound_max - bound_min).max()
    pc_start = (pc_start - center) / scale
    pc_target = (pc_target - center) / scale

    pc_start_unsampled = pc_start.copy()
    pc_target_unsampled = pc_target.copy()
    pc_start, seg_mask_start = downsample_point_cloud(pc_start, seg_mask_start)
    pc_target, seg_mask_target = downsample_point_cloud(pc_target, seg_mask_target)

    screw_point = np.cross(screw_axis, screw_moment)
    p2l_vec, p2l_dist = batch_perpendicular_line(pc_start, screw_axis, screw_point)

    data_tuple = (pc_start, pc_target, seg_mask_start, seg_mask_target,
                  center,scale,
                    joint_type, screw_axis, state_start, state_target,
                    screw_moment, p2l_vec, p2l_dist, pc_start_unsampled, pc_target_unsampled)

    data_list.append(data_tuple)

idx = np.random.randint(len(data_list))

pc_start, pc_target,seg_mask_start,_,center,scale,_,_,_,_,_,_,_,pc_start_unsampled,pc_target_unsampled = data_list[2]


pc_start = torch.from_numpy(pc_start).unsqueeze(0).cuda().float()
pc_target = torch.from_numpy(pc_target).unsqueeze(0).cuda().float()




model = ArticulationNet().cuda()

ckpt = torch.load('./pre_trained_models/2025-09-24 21:43:04/chkpt_30.pth')
device = torch.device('cuda')
model.load_state_dict(ckpt, strict=True)

joint_type_pred, pivot_point_pred, config_pred = inference(model, pc_start, pc_target, seg_mask_start,center,scale)


# pc_start_points = np.asarray(pc_start.squeeze(0).cpu())
# pc_target_points = np.asarray(pc_target.squeeze(0).cpu())

# pc_start_points = downsample_point_cloud(pc_start_points, num_points=1024)
# pc_target_points = downsample_point_cloud(pc_target_points, num_points=1024)

pc_start_points = pc_start_unsampled.copy()
pc_target_points = pc_target_unsampled.copy()
pc_start_points = downsample_point_cloud(pc_start_points, num_points=4096*1.5)
pc_target_points = downsample_point_cloud(pc_target_points, num_points=4096*1.5)
pcd_start = o3d.geometry.PointCloud()
pcd_start.points = o3d.utility.Vector3dVector(pc_start_points)
colors = np.zeros_like(pc_start_points)
colors+= np.array([1,0,0])
pcd_start.colors = o3d.utility.Vector3dVector(colors)

pcd_target = o3d.geometry.PointCloud()
pcd_target.points = o3d.utility.Vector3dVector(pc_target_points)
colors = np.zeros_like(pc_target_points)
colors+= np.array([0,1,0])
pcd_target.colors = o3d.utility.Vector3dVector(colors)


pcd_target1 = o3d.geometry.PointCloud()
pcd_target1.points = o3d.utility.Vector3dVector(pc_target_points)
colors = np.zeros_like(pc_target_points)
colors+= np.array([0,1,0])
pcd_target1.colors = o3d.utility.Vector3dVector(colors)
tr = np.array([0.0,-0.2,-0.00])
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
sphere.paint_uniform_color([0, 0, 0]) 
sphere.translate(np.asarray(pivot_point_pred)+tr,relative=False)

arrow = create_axis_arrow(pivot_point_pred, joint_type_pred, length=0.55,tr=tr)

# mesh = mesh_from_pcd(pcd_target1)

o3d.visualization.draw_geometries([pcd_target,pcd_start, arrow, sphere])





