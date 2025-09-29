import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import random 
import open3d as o3d 
from utilis.sdf_gen import points_to_occ_grid, occ_to_tsdf, voxel_index_to_world, tsdf_of_point

class CustomDataset(Dataset):
    def __init__(self, pairs_root_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.pairs_root_dir = pairs_root_dir
        self.pairs_list = []
        files = sorted([f for f in os.listdir(self.pairs_root_dir) if f.endswith("0.npz")])
        for file in files:
            data_path = os.path.join(self.pairs_root_dir, file)
            data_npz = np.load(data_path)

            pc_start = data_npz["pc_start"]
            pc_target = data_npz["pc_target"]
            start_labels = (data_npz["pc_seg_start"])
            target_labels = (data_npz["pc_seg_target"])
            start_p_occ = data_npz['p_occ_start']
            occ_list_start = data_npz['occ_list_start']
            target_p_occ = data_npz['p_occ_target']
            occ_list_target = data_npz['occ_list_target']
            correspondences = data_npz["correspondences"]
            filtered_correspondences = self.filter_correspondences(correspondences, pc_start, pc_target, start_labels, target_labels)
            filtered_correspondences = self.filter_by_distance(filtered_correspondences)

            bound_max = np.maximum(start_p_occ.max(0), target_p_occ.max(0))
            bound_min = np.minimum(start_p_occ.min(0), target_p_occ.min(0))
            center = (bound_min + bound_max) / 2
            scale = (bound_max - bound_min).max()
            start_p_occ = (start_p_occ - bound_min) / scale
            target_p_occ = (target_p_occ - bound_min) / scale
            pc_start = (pc_start - bound_min) / scale
            pc_target = (pc_target - bound_min) / scale

            filtered_correspondences[:, :3] = (filtered_correspondences[:,:3] - bound_min) / scale
            filtered_correspondences[:, 3:] = (filtered_correspondences[:,3:] - bound_min) / scale
            
            start_masks = []

            for i in range (np.max(start_labels)+1):
                mask = np.asarray(start_labels == i, dtype=bool)
                start_masks.append(mask)

            target_masks = []
            for i in range (np.max(target_labels)+1):
                mask = np.asarray(target_labels == i, dtype=bool)
                target_masks.append(mask)
            pc_start, start_masks = self.downsample_point_cloud(pc_start, start_masks)
            pc_target, target_masks = self.downsample_point_cloud(pc_target, target_masks)
            occ_grid_start, occ_grid_target = points_to_occ_grid(start_p_occ, target_p_occ, occ_list_start, occ_list_target)
            sdf_start = occ_to_tsdf(occ_grid_start)
            sdf_target = occ_to_tsdf(occ_grid_target)

            mobile_masks_start = start_masks[1:]

            mobile_parts_mask = torch.zeros_like(torch.from_numpy(mobile_masks_start[0]), dtype=torch.bool)
            for j in range(0, 1):
                mobile_parts_mask |= mobile_masks_start[j]

            pc_mobile = pc_start[mobile_parts_mask]

            data_tuples = (pc_start, pc_target, start_masks, target_masks, sdf_start, sdf_target, filtered_correspondences,pc_mobile)
            self.pairs_list.append(data_tuples)

    def __len__(self):
        return self.pairs_list[0][0].shape[0]
    
    def __getitem__(self,idx):
        # pairs_of_pcs = random.choice(self.pairs_list)
        pairs_of_pcs = self.pairs_list[0]
        # idx1, idx2 = random.sample([0, 1], 2)
        idx1 = 0
        idx2 = 1
        pc_start = pairs_of_pcs[idx1]
        pc_target = pairs_of_pcs[idx2]
        if idx1 == 0:
            masks_start = pairs_of_pcs[2]
            masks_target = pairs_of_pcs[3]
            sdf_start = pairs_of_pcs[4]
            sdf_target = pairs_of_pcs[5]
            corresp_start = pairs_of_pcs[6][:, :3]
            corresp_target = pairs_of_pcs[6][:, 3:]
            pc_mobile = pairs_of_pcs[7]
        else:
            masks_start = pairs_of_pcs[3]
            masks_target = pairs_of_pcs[2]
            sdf_start = pairs_of_pcs[5]
            sdf_target = pairs_of_pcs[4]
            corresp_start = pairs_of_pcs[6][:, 3:]
            corresp_target = pairs_of_pcs[6][:, :3]
            pc_mobile = pairs_of_pcs[7]

        return pc_mobile[idx],pc_start, pc_target, masks_start, masks_target, sdf_start, sdf_target, corresp_start, corresp_target
  
    def filter_correspondences(self,correspondences, pc_start, pc_target, pc_seg_start, pc_seg_target):
        pcd_start = o3d.geometry.PointCloud()
        pcd_start.points = o3d.utility.Vector3dVector(pc_start)
        colors_start = np.zeros_like(pc_start)
        colors_start[pc_seg_start==0] = [0.5,0.5,0.5]
        colors_start[pc_seg_start==1] = [1,0,0]
        pcd_start.colors = o3d.utility.Vector3dVector(colors_start)

        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(pc_target)
        colors_target = np.zeros_like(pc_target)
        colors_target[pc_seg_target==0] = [0.5,0.5,0.5]
        colors_target[pc_seg_target==1] = [0,1,0]
        pcd_target.colors = o3d.utility.Vector3dVector(colors_target)

        pcd_start_mobile = o3d.geometry.PointCloud()
        pcd_start_mobile.points = o3d.utility.Vector3dVector(pc_start[pc_seg_start==1])
        pcd_target_mobile = o3d.geometry.PointCloud()
        pcd_target_mobile.points = o3d.utility.Vector3dVector(pc_target[pc_seg_target==1])


        # Get AABB for both clouds
        bbox1 = pcd_start_mobile.get_axis_aligned_bounding_box()
        bbox2 = pcd_target_mobile.get_axis_aligned_bounding_box()

        # Combine bounds
        min_bound = np.minimum(bbox1.get_min_bound(), bbox2.get_min_bound())
        max_bound = np.maximum(bbox1.get_max_bound(), bbox2.get_max_bound())

        # Create new combined bounding box
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        # Extract start and target points from correspondences
        start_pts = correspondences[:, :3]
        target_pts = correspondences[:, 3:]

        # Check which correspondences have BOTH points inside bbox
        mask_start_inside = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(start_pts))
        mask_target_inside = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(target_pts))

        # Convert to boolean mask (True if both start and target are inside)
        mask_start_bool = np.isin(np.arange(len(start_pts)), mask_start_inside)
        mask_target_bool = np.isin(np.arange(len(target_pts)), mask_target_inside)
        mask = mask_start_bool & mask_target_bool

        # Filter correspondences
        return correspondences[mask]
    
    def downsample_point_cloud(self, points, labels_list, num_points=1024):
        """
        Randomly downsample the point cloud to a fixed size.
        """
        N = points.shape[0]
        if N >= num_points:
            np.random.seed(97)
            indices = np.random.choice(N, num_points, replace=False)
        else:
            np.random.seed(97)
            indices = np.random.choice(N, num_points, replace=True)  # pad if too small
        for i in range(len(labels_list)):
            labels = labels_list[i]
            labels = labels[indices]
            labels_list[i] = labels
        return points[indices], labels_list

    def filter_by_distance(self, correspondences, threshold=0.05):
        diffs = correspondences[:, :3] - correspondences[:, 3:]
        dists = np.linalg.norm(diffs, axis=1)
        mask = dists < threshold
        return correspondences[mask]