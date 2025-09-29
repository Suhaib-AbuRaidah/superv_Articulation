import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as f
from utilis.dataset1 import CustomDataset
from torch.utils.data import DataLoader
from utilis.Visualizer import VisualizerWrapper
import datetime
import os
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import Ridge


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias= False)


    def forward(self, x):
        return self.linear(x)

def tsdf_of_point1(points, tsdf_volume, grid_res=64):

    points_scaled = points * (grid_res - 1 - 1e-5)
    points_scaled = torch.clamp(points_scaled, 0, grid_res - 1 - 1e-5)
    idx = points_scaled.long()  # [num_points, 3]
    # Make sure batch dimension aligns
    batch_size = points.shape[0]  # batch=0
    sampled_tsdf = tsdf_volume[batch_size-1, idx[:,0], idx[:,1], idx[:,2]]  # [num_points]
    return sampled_tsdf

def rotate_about_pivot(points, axis, angle, pivot):
    """
    points: (N,3)
    axis: (3,)
    angle: scalar
    pivot: (3,)
    """
    rotvec = axis * angle
    R_mat = torch.from_numpy(R.from_rotvec(rotvec.detach().cpu().numpy()).as_matrix()).to(points.device).float()
    return ((points - pivot) @ R_mat.T) + pivot

def tsdf_of_point(points, tsdf_volume, grid_res=64):
    """
    points: (B, 3) in [0,1] normalized coords
    tsdf_volume: (1, 1, D, H, W)  # add batch & channel dims
    """
    # Scale points to [-1, 1] for grid_sample
    points_scaled = points * (grid_res - 1 - 1e-5)
    points_scaled = torch.clamp(points_scaled, 0, grid_res - 1 - 1e-5)  
    coords = points_scaled.view(1, -1, 1, 1, 3)  # (N=1, B, 1, 1, 3)

    sampled = F.grid_sample(
        tsdf_volume.unsqueeze(0),  # (1,1,D,H,W)
        coords,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return sampled.view(-1)  # (B,)

def rotation_matrix_to_axis_angle(R):
    # trace gives cos(theta)
    cos_theta = (torch.trace(R) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # numerical safety
    angle = torch.acos(cos_theta)

    if torch.isclose(angle, torch.tensor(0.0, device=R.device)):
        # No rotation
        axis = torch.tensor([1.0, 0.0, 0.0], device=R.device)
    else:
        # (R - R^T) encodes the axis information
        axis = torch.tensor([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
        ], device=R.device) / (2*torch.sin(angle))

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

    batch_size = xyz.shape[0]
    ones = torch.ones(batch_size,1).cuda()
    point = torch.cat((xyz,ones),1)
    point_stacked = torch.cat((point, point, point),1)

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


    for i in range(num_joints):
        current_mask = mobile_masks_start[i]
        # Rotate current joint points
        points = pc_transformed[current_mask]
        pc_rotated = rotate_about_pivot(points, axis, angle, pivot_point)
        pc_transformed[current_mask] = pc_rotated

        if i < num_joints - 1:
            # Create combined mask of downstream joints
            downstream_mask = torch.zeros_like(mobile_masks_start[0], dtype=torch.bool)
            for j in range(i + 1, num_joints):
                downstream_mask |= mobile_masks_start[j]

            # Apply rotation to downstream points
            downstream_points = pc_transformed[downstream_mask]
            transformed_points = rotate_about_pivot(downstream_points, axis, angle, pivot_point)
            
            pc_transformed[downstream_mask] = transformed_points
        # for i in range(50):
    viz.update(pc_transformed, pc_target, action = action, mask_start=mobile_masks_start,mask_target=mobile_masks_target)

    tsdf_initial = tsdf_of_point(point[:,:3], tsdf_start)
    tsdf_rotated = tsdf_of_point(point_trans[:,:3], tsdf_target)

    loss = torch.mean((tsdf_initial - tsdf_rotated)**2) 
    return loss


dataset = CustomDataset("./Pairs")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

start_training_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
checkpoint_path = f"./pre_trained_models_regression/{start_training_time}"
os.makedirs(checkpoint_path, exist_ok=True)
viz = VisualizerWrapper()
# model = LinearRegression(12,12).cuda()
model = Ridge(1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
# print(model)
viz = VisualizerWrapper()
loss = torch.tensor(10)
epoch = 0
while loss.item() > 0.2:
    epoch += 1
    for xyz, pc_start, pc_target, masks_start, masks_target, sdf_start, sdf_target, corresp_start, corresp_target in dataloader:
        xyz = xyz.cuda().float()
        pc_start = pc_start.cuda().float()
        pc_target = pc_target.cuda().float()
        sdf_start = sdf_start.cuda().float()
        sdf_target = sdf_target.cuda().float()
        corresp_start = corresp_start.cuda().float()
        corresp_target = corresp_target.cuda().float()

        kwargs = {"xyz": xyz, "pc_start": pc_start, "pc_target": pc_target,
                   "masks_start": masks_start, "masks_target": masks_target,
                   "tsdf_start": sdf_start, "tsdf_target": sdf_target,
                    }
        
        optimizer.zero_grad() 
        loss = training_step(model, viz, kwargs)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    if epoch % 100 == 0:
        weights_path = os.path.join(checkpoint_path, f"chkpt_{epoch}.pth")
        torch.save(model.state_dict(), weights_path)