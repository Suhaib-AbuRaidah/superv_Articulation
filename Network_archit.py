import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from utilis.dataset2 import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
from encoder.Pointnetpp_attn import PointNetPlusPlusAttnFusion
from utilis.Visualizer import VisualizerWrapper
from encoder.encoder import LocalPoolPointnetPPFusion
from decoder.decoder import LocalDecoderV1, LocalDecoder




# --- 1. Model ---
class ArticulationNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        attn_kwargs1 = {"emb_dims": 256, "n_blocks": 4,"dropout": 0.1,"ff_dims": 512, "n_heads": 4, "residual": True,"return_score":True}
        self.encoder = LocalPoolPointnetPPFusion(
            c_dim=128,                  # latent feature dimension
            dim=3,                      # 3D point cloud input
            hidden_dim=128,             # hidden layer size
            scatter_type="max",         # can also try "mean"
            mlp_kwargs={"dims": [64, 128, 256],"bn": False},   # simple MLP config
            attn_kwargs=attn_kwargs1,  # attention fusion config
            unet=True,                  # enable 2D U-Net refinement
            unet_kwargs={"depth": 3, "num_channels": 128},
            unet3d=False,               # no 3D U-Net by default (expensive)
            unet3d_kwargs=None,
            unet_corr=False,            # no U-Net for correspondence features by default
            unet_kwargs_corr=None,
            unet3d_corr=False,
            unet3d_kwargs_corr=None,
            corr_aggregation="max",  # how to merge correspondence features (could be "sum")
            plane_resolution=64,        # resolution of planes (e.g., 64x64)
            grid_resolution=32,         # resolution of grid (e.g., 32^3)
            plane_type=["corr_xz", "corr_xy", "corr_yz"], # use all 3 canonical planes
            padding=0.1,                # add small padding around points
            n_blocks=5,                 # number of ResNet-style blocks
            feat_pos="corr",            # use attention-based fusion
            return_score=False          # no need for attention scores in output
        )
        self.decoder_joint_type = LocalDecoder()
        self.decoder_revolute = LocalDecoder(out_dim=8)
        self.decoder_prismatic = LocalDecoder(out_dim=4)
        # self.fc_pivot = nn.Linear(emb_dim, 3)
        # self.fc_axis = nn.Linear(emb_dim, 3)
        # self.fc_angle = nn.Linear(emb_dim, 1)

    def forward(self, pc_start, pc_target):
        feat_map = self.encoder(pc_start, pc_target)
        joint_type_logits = self.decoder_joint_type(pc_start, feat_map)
        joint_param_r = self.decoder_revolute(pc_start, feat_map)
        joint_param_p = self.decoder_prismatic(pc_start, feat_map)

        # return joint_type_logits.mean(dim=1), joint_param_r.mean(dim=1)
        return joint_type_logits, joint_param_r, joint_param_p

        # joint_param_r = self.decoder_revolute(p_seg, c, **kwargs)
        # joint_param_p = self.decoder_prismatic(p_seg, c, **kwargs)
        # pivot = self.fc_pivot(feat_global)
        # axis = F.normalize(self.fc_axis(feat_global), dim=-1)
        # angle = self.fc_angle(feat_global)

        # return pivot, axis, angle

# --- 2. Transform Points ---
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


# --- 3. TSDF Sampler ---
def tsdf_of_point(points, tsdf_volume, grid_res=64):

    points_scaled = points * (grid_res - 1 - 1e-5)
    points_scaled = torch.clamp(points_scaled, 0, grid_res - 1 - 1e-5)
    idx = points_scaled.long()  # [num_points, 3]
    # Make sure batch dimension aligns
    batch_size = points.shape[0]  # batch=0
    sampled_tsdf = tsdf_volume[batch_size-1, idx[:,:,0], idx[:,:,1], idx[:,:,2]]  # [num_points]
    return sampled_tsdf




def training_step(model, data_dict):
    pc_start = data_dict["pc_start"]         # torch.Tensor, shape (N,3)
    pc_target = data_dict["pc_target"]       # torch.Tensor, shape (M,3)
    joint_type_gt = data_dict['joint_type']
    screw_axis = data_dict['screw_axis']
    state_start = data_dict['state_start']
    state_target = data_dict['state_end']
    gt_p2l_vec = data_dict['p2l_vec']
    gt_p2l_dist = data_dict['p2l_dist']
    joint_state_gt = state_start-state_target
    joint_type_gt = joint_type_gt.unsqueeze(1).expand(-1, pc_start.size(1))
    joint_state_gt = joint_state_gt.unsqueeze(1).expand(-1, pc_start.size(1))
    screw_axis = F.normalize(screw_axis, dim=-1)
    screw_axis = screw_axis.unsqueeze(1).expand(-1, pc_start.size(1),3)

    joint_type_logits, joint_param_revolute, joint_param_prismatic = model(pc_start, pc_target)

    screw_axis = F.normalize(screw_axis, dim=-1)
    joint_type = joint_type_logits

    joint_r_axis = joint_param_revolute[:,:, :3]
    joint_r_axis = F.normalize(joint_r_axis, dim=-1)
    joint_r_state = joint_param_revolute[:,:,3]
    joint_r_p2l_vec = joint_param_revolute[:,:, 4:7]
    joint_r_p2l_dist = joint_param_revolute[:,:, 7]

    joint_p_axis = joint_param_prismatic[:, :, :3]
    joint_p_axis = F.normalize(joint_p_axis, dim=-1)
    joint_p_state = joint_param_prismatic[:, :, 3]

    joint_type = torch.clamp(joint_type, -20, 20)
    # print(f"Joint type: {joint_type.mean()}\n\nJoint axis: {joint_r_axis.mean()}\n\nJoint state: {joint_r_state.mean()}\n\nJoint p2l vec: {joint_r_p2l_vec.mean()}\n\nJoint p2l dist: {joint_r_p2l_dist.mean()}\n\n")

    
    joint_type_loss = F.binary_cross_entropy_with_logits(joint_type, joint_type_gt,reduction="mean")
    # joint_type_loss = joint_type_loss.mean(-1).mean()


    # Revolute mask: label == 0
    mask_r = (joint_type_gt == 0).float()  # shape [B, N]
    # Prismatic mask: label == 1
    mask_p = (joint_type_gt == 1).float()

    # Revolute losses
    dot_r = torch.einsum("bnm,bnm->bn", joint_r_axis, screw_axis)
    dot_r = torch.clamp(dot_r, -1+1e-5, 1-1e-5)
    loss_axis_r = torch.acos(dot_r).mean(-1)           # [B]
    loss_state_r = F.l1_loss(joint_r_state, joint_state_gt, reduction="none").mean(-1)  # [B]
    loss_p2l_ori_dot = -torch.einsum("bnm,bnm->bn", joint_r_p2l_vec, gt_p2l_vec)
    loss_p2l_ori_dot = torch.clamp(loss_p2l_ori_dot, -1+1e-5, 1-1e-5)
    loss_p2l_ori = torch.acos(loss_p2l_ori_dot).mean(-1)  # [B]
    loss_p2l_dist = F.l1_loss(joint_r_p2l_dist, gt_p2l_dist, reduction="none").mean(-1)  # [B]

    revolute_loss = loss_axis_r + loss_state_r + loss_p2l_ori + loss_p2l_dist

    # Prismatic losses
    dot_p = torch.einsum("bnm,bnm->bn", joint_p_axis, screw_axis)
    dot_p = torch.clamp(dot_p, -1+1e-5, 1-1e-5)
    loss_axis_p = torch.acos(dot_p).mean(-1)           # [B]
    loss_state_p = F.l1_loss(joint_p_state, joint_state_gt, reduction="none").mean(-1)  # [B]

    prismatic_loss = loss_axis_p + loss_state_p

    # Combine with masks
    per_sample_loss = mask_r[:,0] * revolute_loss + mask_p[:,0] * prismatic_loss
    per_sample_loss = per_sample_loss.mean()

    # Add classification loss
    total_loss = joint_type_loss + per_sample_loss

    return total_loss

# --- 4. Training Step ---

# def training_step(model,viz, kwargs):
#     pc_start = kwargs["pc_start"]         # torch.Tensor, shape (N,3)
#     pc_target = kwargs["pc_target"]       # torch.Tensor, shape (M,3)
#     masks_start = kwargs["masks_start"]   # list of numpy bool arrays
#     masks_target = kwargs["masks_target"] # list of numpy bool arrays
#     tsdf_start = kwargs["tsdf_start"]     # torch.Tensor
#     tsdf_target = kwargs["tsdf_target"]   # torch.Tensor

#     device = pc_start.device    

#     pc_transformed = pc_start.clone()
#     mobile_masks_start = masks_start[1:]
#     mobile_masks_target = masks_target[1:]
#     num_joints = len(mobile_masks_start)

#     min_dists = torch.zeros(num_joints, device=device)

#     pivot_point, axis, angle = model(pc_start, pc_target)
#     # pivot_point_constant = torch.tensor([0.5,0.55,0.5]).to(device).unsqueeze(0)
#     # pivot_point = pivot_point_constant + (pivot_point - pivot_point.detach())
#     # axis_constant = torch.tensor([0,0,1]).to(device).unsqueeze(0)
#     # axis = axis_constant + (axis - axis.detach())
#     # angle_constant = torch.tensor(np.pi/2).to(device).unsqueeze(0)
#     # angle = angle_constant + (angle - angle.detach())
#     action = [angle.item(), pivot_point[0,0].item(), pivot_point[0,1].item(), pivot_point[0,2].item(), axis[0,0].item(), axis[0,1].item(), axis[0,2].item()]
#     # for i in range(50):
#     viz.update(pc_transformed, pc_target,action = action, mask_start=mobile_masks_start,mask_target=mobile_masks_target)
#     for i in range(num_joints):

#         parent_mask = masks_start[i]
#         parent_points = pc_transformed[parent_mask]  # (P,3)
#         min_dist_parent = torch.min(torch.norm(parent_points - pivot_point, dim=1))

#         current_mask = mobile_masks_start[i]
#         current_points = pc_transformed[current_mask]  # (C,3)
#         min_dist_current = torch.min(torch.norm(current_points - pivot_point, dim=1))

#         min_dist = min_dist_parent + min_dist_current
#         min_dists[i] = min_dist

#         # Rotate current joint points
#         points = pc_transformed[current_mask]
#         pc_rotated = rotate_about_pivot(points, axis, angle, pivot_point)
#         pc_transformed[current_mask] = pc_rotated

#         if i < num_joints - 1:
#             # Create combined mask of downstream joints
#             downstream_mask = torch.zeros_like(mobile_masks_start[0], dtype=torch.bool)
#             for j in range(i + 1, num_joints):
#                 downstream_mask |= mobile_masks_start[j]

#             # Apply rotation to downstream points
#             downstream_points = pc_transformed[downstream_mask]
#             transformed_points = rotate_about_pivot(downstream_points, axis, angle, pivot_point)
            
#             pc_transformed[downstream_mask] = transformed_points
#         # for i in range(50):
#         viz.update(pc_transformed, pc_target, action = action, mask_start=mobile_masks_start,mask_target=mobile_masks_target)
#     # Final downstream mask (all mobile joints except root)
#     mobile_parts_mask = torch.zeros_like(mobile_masks_start[0], dtype=torch.bool)
#     for j in range(0, num_joints):
#         mobile_parts_mask |= mobile_masks_start[j]
#     # Randomly sample 100 downstream points
#     num_downstream = mobile_parts_mask.sum()
#     idxs = torch.randperm(num_downstream, device=device)[:100]
#     mobile_points_sdf_initial = pc_start[mobile_parts_mask][idxs].unsqueeze(0)
#     mobile_points_sdf_rotated = pc_transformed[mobile_parts_mask][idxs].unsqueeze(0)
#     tsdf_initial = tsdf_of_point(mobile_points_sdf_initial, tsdf_start)
#     tsdf_rotated = tsdf_of_point(mobile_points_sdf_rotated, tsdf_target)
#     # Loss = average absolute difference
#     loss = torch.mean(torch.abs(tsdf_initial - tsdf_rotated)) + torch.mean(min_dists)
#     return loss


