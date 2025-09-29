from sklearn.linear_model import Ridge
import numpy as np
from utilis.Visualizer import VisualizerWrapper
import datetime
import os
from scipy.spatial.transform import Rotation as R


def tsdf_of_point1(points, tsdf_volume, grid_res=64):

    points_scaled = points * (grid_res - 1 - 1e-5)
    points_scaled = np.clip(points_scaled, 0, grid_res - 1 - 1e-5)
    # idx = points_scaled.long()  # [num_points, 3]
    # Make sure batch dimension aligns
    # batch_size = points.shape[0]  # batch=0
    sampled_tsdf = tsdf_volume[points_scaled[:,0], points_scaled[:,1], points_scaled[:,2]]  # [num_points]
    return sampled_tsdf

def rotate_about_pivot(points, axis, angle, pivot):
    """
    points: (N,3)
    axis: (3,)
    angle: scalar
    pivot: (3,)
    """
    rotvec = axis * angle
    R_mat = R.from_rotvec(rotvec)
    return ((points - pivot) @ R_mat.T) + pivot

def tsdf_of_point(points, tsdf_volume, grid_res=64):
    lst = []
    points = points*(grid_res-1)
    points = np.clip(points, 0, grid_res-1)
    for u in range(points.shape[0]):
        i, j, k = points[u].astype(int)
        lst.append(tsdf_volume[i, j, k])
    return np.array(lst)    

def rotation_matrix_to_axis_angle(R):
    # trace gives cos(theta)
    cos_theta = (np.trace(R) - 1)/2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    angle = np.arccos(cos_theta)
    if angle == 0:
        # No rotation
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
        ]) / (2*np.sin(angle))

    return axis, angle


def training_step(model,viz, kwargs):
    xyz = kwargs["xyz"]         # torch.Tensor, shape (1,3)
    pc_start = kwargs["pc_start"]         # torch.Tensor, shape (N,3)
    pc_target = kwargs["pc_target"]       # torch.Tensor, shape (N,3)
    masks_start = kwargs["masks_start"]   # list of numpy bool arrays
    masks_target = kwargs["masks_target"]
    tsdf_start = kwargs["tsdf_start"]     # torch.Tensor
    tsdf_target = kwargs["tsdf_target"]   # torch.Tensor


    pc_transformed = pc_start.clone()
    mobile_masks_start = masks_start[1:]
    mobile_masks_target = masks_target[1:]
    num_joints = len(mobile_masks_start)



    theta = model(point_stacked).view(3, 4)  # Expecting 12 outputs
    trans = torch.cat([theta, torch.tensor([[0,0,0,1]], device=theta.device)], dim=0)

    R = trans[:3, :3]   # rotation part
    t = trans[:3, 3]    # translation (pivot point)
    axis, angle = rotation_matrix_to_axis_angle(R)
    pivot_point = t   # translation vector
    # trans_cons = torch.tensor([0.5,torch.randn(1).item(),0.5,1]).to('cuda').unsqueeze(0)
    # trans[0,:] = trans_cons + (trans[0,:] - trans[0,:].detach())

    point_trans = trans @ point.T
    point_trans = point_trans.T
    action = [
    angle.item(),
    pivot_point[0].item(), pivot_point[1].item(), pivot_point[2].item(),
    axis[0].item(), axis[1].item(), axis[2].item()]
    viz.update(pc_transformed, pc_target,action = action, mask_start=mobile_masks_start,mask_target=mobile_masks_target)