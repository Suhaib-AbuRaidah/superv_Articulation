import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import glob

class CustomDataset(Dataset):
    def __init__(self, file_paths):
        # file_paths = "./data/syn_local/refrigerator/scenes/*.npz"
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
            pc_start, seg_mask_start = self.downsample_point_cloud(pc_start, seg_mask_start)
            pc_target, seg_mask_target = self.downsample_point_cloud(pc_target, seg_mask_target)
            bound_max = np.maximum(pc_start.max(0), pc_target.max(0))
            bound_min = np.minimum(pc_start.min(0), pc_target.min(0))
            center = (bound_min + bound_max) / 2
            scale = (bound_max - bound_min).max()
            pc_start = (pc_start - center) / scale
            pc_target = (pc_target - center) / scale
            screw_point = np.cross(screw_axis, screw_moment)
            p2l_vec, p2l_dist = batch_perpendicular_line(pc_start, screw_axis, screw_point)

            data_tuple = (pc_start, pc_target, seg_mask_start, seg_mask_target,
                           joint_type, screw_axis, state_start, state_target,
                           screw_moment, p2l_vec, p2l_dist)

            data_list.append(data_tuple)
        
        self.pairs_list = data_list


    def __len__(self):
        return len(self.pairs_list)
    
    def __getitem__(self,idx):
        pairs_of_pcs = self.pairs_list[idx]

        pc_start = pairs_of_pcs[0]
        pc_target = pairs_of_pcs[1]
        seg_mask_start = pairs_of_pcs[2]
        seg_mask_target = pairs_of_pcs[3]
        joint_type = pairs_of_pcs[4]
        screw_axis = pairs_of_pcs[5]
        state_start = pairs_of_pcs[6]
        state_target = pairs_of_pcs[7]
        screw_moment = pairs_of_pcs[8]
        p2l_vec = pairs_of_pcs[9]
        p2l_dist = pairs_of_pcs[10]


        return pc_start, pc_target, seg_mask_start, seg_mask_target, joint_type, screw_axis, state_start, state_target, screw_moment, p2l_vec, p2l_dist
    
  
    
    def downsample_point_cloud(self, points, labels, num_points=1024):
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
        labels = labels[indices]
        # for i in range(len(labels_list)):
        #     labels = labels_list[i]
        #     labels = labels[indices]
        #     labels_list[i] = labels
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


from torch.utils.data import Dataset
import glob
import numpy as np
import torch

class Segmentation_Dataset(Dataset):
    def __init__(self, file_paths, device):
        self.device = device
        self.data_list = []

        for f in sorted(glob.glob(file_paths)):
            data = np.load(f, allow_pickle=True)

            pc_start = data['pc_start']
            pc_end = data['pc_end']
            seg_mask_start = data['pc_seg_start'].item()
            seg_mask_end = data['pc_seg_end'].item()

            pc_start, mask_start_list = self.downsample_pc_masks(pc_start, seg_mask_start)
            pc_end, mask_end_list = self.downsample_pc_masks(pc_end, seg_mask_end)

            # --- convert masks → per-point labels ---
            seg_labels_start = self.masks_to_labels(mask_start_list)
            seg_labels_end = self.masks_to_labels(mask_end_list)

            # normalize
            bound_max = pc_start.max(0)
            bound_min = pc_start.min(0)
            center = (bound_min + bound_max) / 2
            scale = (bound_max - bound_min).max()

            pc_start = (pc_start - center) / scale
            pc_end = (pc_end - center) / scale

            pc_start = torch.tensor(pc_start, dtype=torch.float32)
            pc_end = torch.tensor(pc_end, dtype=torch.float32)
            seg_labels_start = torch.tensor(seg_labels_start, dtype=torch.long)
            seg_labels_end = torch.tensor(seg_labels_end, dtype=torch.long)

            self.data_list.append(
                (pc_start, seg_labels_start, pc_end, seg_labels_end)
            )

    def masks_to_labels(self, mask_dict):
        masks = []
        for k in sorted(mask_dict.keys()):
            masks.append(mask_dict[k])
        masks = np.stack(masks, axis=0)  # (num_parts, N)
        labels = np.argmax(masks, axis=0)
        return labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        pairs_of_pcs = self.data_list[idx]

        pc_start = pairs_of_pcs[0]
        seg_labels_start = pairs_of_pcs[1]
        pc_end = pairs_of_pcs[2]
        seg_labels_end = pairs_of_pcs[3]

        return pc_start, seg_labels_start, pc_end, seg_labels_end
    
    def downsample_pc_masks(self, points, masks_list=None, num_points=4096):
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
        if masks_list is not None:
            for i in range(len(masks_list)):
                labels = masks_list[i]
                labels = labels[indices]
                masks_list[i] = labels
            return points[indices], masks_list
        else:
            return points[indices]
        


class PartsGraphDataset2(Dataset):
    def __init__(self, file_paths,device):
        # file_paths = "./data/Shape2Motion_gcn/*/scenes/*.npz"
        data_list = []
        for f in sorted(glob.glob(file_paths)):
            data = np.load(f, allow_pickle=True)
            
            pc_start = data[f'pc_start']
            pc_end = data[f'pc_end']
            adjacency_matrix = data['adj']
            parts_conne_gt = data['parts_conne_gt']
            seg_mask_start = data['pc_seg_start'].item()
            seg_mask_end = data['pc_seg_end'].item()
            joint_type_list = data['joint_type']
            screw_axis_list = data['screw_axis']
            screw_moment_list = data['screw_moment']
            angles_list = data['state_diff']
            file_name = f.split('/')[-1]
            num_joints = len(seg_mask_start)-1

            pc_start, mask_start_list = self.downsample_pc_masks(pc_start, seg_mask_start)
            pc_end, mask_end_list = self.downsample_pc_masks(pc_end, seg_mask_end)

            bound_max = pc_start.max(0)
            bound_min = pc_start.min(0)
            center = (bound_min + bound_max) / 2
            scale = (bound_max - bound_min).max()
            pc_start = (pc_start - center) / scale
            pc_end = (pc_end - center) / scale

            screw_point_list = []
            for joint in range(num_joints):
                screw_axis = screw_axis_list[joint]
                screw_moment = screw_moment_list[joint]
                screw_point = np.cross(screw_axis, screw_moment)
                screw_point = (screw_point - center) / scale
                screw_point_list.append(screw_point)

            joints_type = torch.tensor(np.array(joint_type_list), dtype=torch.float32).to(device)
            joints_screw_axis = torch.tensor(np.array(screw_axis_list), dtype=torch.float32).to(device)
            joints_screw_point = torch.tensor(np.array(screw_point_list), dtype=torch.float32).to(device)
            adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32).to(device)
            parts_conne_gt = torch.tensor(parts_conne_gt, dtype=torch.float32).to(device)
            angles_list = torch.tensor(np.array(angles_list), dtype=torch.float32).to(device)

            parts_start_list = []
            for mask in range(num_joints+1):
                part = pc_start[mask_start_list[mask]]
                part = self.downsample_pc_masks(part, num_points=1024)
                part = torch.tensor(part, dtype=torch.float32).to(device)
                parts_start_list.append(part)
            
            parts_end_list = []
            for mask in range(num_joints+1):
                part = pc_end[mask_end_list[mask]]
                part = self.downsample_pc_masks(part, num_points=1024)
                part = torch.tensor(part, dtype=torch.float32).to(device)
                parts_end_list.append(part)

            pc_start = torch.tensor(pc_start, dtype=torch.float32).to(device)
            pc_end = torch.tensor(pc_end, dtype=torch.float32).to(device)

            data_tuple = (pc_start, parts_start_list, pc_end, parts_end_list, adjacency_matrix, parts_conne_gt, joints_type, joints_screw_axis, joints_screw_point, angles_list,file_name)

            data_list.append(data_tuple)
        
        self.pairs_list = data_list

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        pairs_of_pcs = self.pairs_list[idx]

        pc_start = pairs_of_pcs[0]
        parts_start_list = pairs_of_pcs[1]
        pc_end = pairs_of_pcs[2]
        parts_end_list = pairs_of_pcs[3]
        adjacency_matrix = pairs_of_pcs[4]
        parts_conne_gt = pairs_of_pcs[5]
        joint_type_list = pairs_of_pcs[6]
        screw_axis_list = pairs_of_pcs[7]
        screw_point_list = pairs_of_pcs[8]
        angles = pairs_of_pcs[9]
        file_name = pairs_of_pcs[10]


        return pc_start, parts_start_list, pc_end, parts_end_list, adjacency_matrix, parts_conne_gt, joint_type_list, screw_axis_list, screw_point_list, angles, file_name
    
    def downsample_pc_masks(self, points, masks_list=None, num_points=4096):
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
        if masks_list is not None:
            for i in range(len(masks_list)):
                labels = masks_list[i]
                labels = labels[indices]
                masks_list[i] = labels
            return points[indices], masks_list
        else:
            return points[indices]
    

def collate_graphs(batch):
    """
    Custom collate_fn for variable-sized graphs.
    Each element in batch = (pc_start, parts_list, adjacency_matrix).
    We return lists instead of stacking.
    """
    pc_starts = []
    parts_lists = []
    adjs = []
    parts_conne_gt_lst = []
    
    for pc_start, parts_list, adj, parts_conne_gt in batch:
        pc_starts.append(pc_start)         # [N, 3]
        parts_lists.append(parts_list)     # list of [P_i, 3]
        adjs.append(adj)                   # [num_parts, num_parts]
        parts_conne_gt_lst.append(parts_conne_gt)

    return {
        "pc_start": pc_starts,
        "parts_list": parts_lists,
        "adj": adjs,
        "parts_connections": parts_conne_gt_lst,
    }

import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class PartsGraphDataset3(Dataset):
    """
    Returns per-sample:
      pc_start:        (N,3) torch
      parts_start_list: list of (1024,3) torch, length = num_parts
      pc_end:          (N,3) torch
      parts_end_list:  list of (1024,3) torch, length = num_parts
      adj:             (num_parts,num_parts) torch
      parts_conne_gt:  (num_parts,num_parts) torch
      joint_type:      (num_joints,) torch  (assumed in upper-tri connected-edge order)
      screw_axis:      (num_joints,3) torch
      screw_point:     (num_joints,3) torch  (point on axis)
      angles:          (num_joints,) torch
      file_name:       str

      edge_src:        (E,) torch long  upper-tri edges
      edge_dst:        (E,) torch long
      edge_dir_gt:     (E,2048,3) torch  (both parts concatenated, zeros for non-edges)
      edge_dist_gt:    (E,2048,1) torch  (both parts concatenated, zeros for non-edges)

    ASSUMPTION:
      screw_axis_list / screw_point_list / joint_type_list / angles_list
      are stored in the SAME order as CONNECTED edges when scanning upper-triangular (i<j).
    """

    def __init__(self, file_paths, device, num_global_points=8192, num_part_points=1024, seed=97):
        self.device = device
        self.num_global_points = int(num_global_points)
        self.num_part_points = int(num_part_points)
        self.seed = int(seed)

        data_list = []
        for f in sorted(glob.glob(file_paths)):
            data = np.load(f, allow_pickle=True)

            pc_start = data["pc_start"].astype(np.float32)  # (N,3)
            pc_end = data["pc_end"].astype(np.float32)      # (N,3)

            adjacency_matrix = data["adj"].astype(np.float32)
            parts_conne_gt_np = data["parts_conne_gt"].astype(np.float32)

            seg_mask_start = data["pc_seg_start"].item()  # dict-like
            seg_mask_end = data["pc_seg_end"].item()

            joint_type_list = np.array(data["joint_type"], dtype=np.float32)
            screw_axis_list = np.array(data["screw_axis"], dtype=np.float32)
            screw_moment_list = np.array(data["screw_moment"], dtype=np.float32)
            angles_list = np.array(data["state_diff"], dtype=np.float32)

            file_name = f.split("/")[-1]

            # num parts = num_joints + 1
            num_parts = len(seg_mask_start)
            num_joints = num_parts - 1

            # Global downsample (keeps masks consistent)
            pc_start, mask_start_list = self.downsample_pc_masks(pc_start, seg_mask_start, num_points=self.num_global_points)
            pc_end, mask_end_list = self.downsample_pc_masks(pc_end, seg_mask_end, num_points=self.num_global_points)

            # Normalize by pc_start bounds
            bound_max = pc_start.max(0)
            bound_min = pc_start.min(0)
            center = (bound_min + bound_max) / 2.0
            scale = (bound_max - bound_min).max()
            scale = float(scale) if float(scale) > 1e-9 else 1.0

            pc_start = (pc_start - center) / scale
            pc_end = (pc_end - center) / scale

            # Build screw points on axis from (axis, moment): q = axis x moment
            screw_point_list = []
            for j in range(num_joints):
                axis = screw_axis_list[j]
                moment = screw_moment_list[j]
                q = np.cross(axis, moment).astype(np.float32)
                q = (q - center) / scale
                screw_point_list.append(q)
            screw_point_list = np.stack(screw_point_list, axis=0).astype(np.float32)  # (num_joints,3)

            # Build per-part point clouds (numpy + torch), each (1024,3)
            parts_start_np_list = []
            parts_start_list = []
            for k in range(num_parts):
                part = pc_start[mask_start_list[k]]
                part = self.downsample_points(part, num_points=self.num_part_points)
                part = part.astype(np.float32)
                parts_start_np_list.append(part)
                parts_start_list.append(torch.tensor(part, dtype=torch.float32, device=device))

            parts_end_list = []
            for k in range(num_parts):
                part = pc_end[mask_end_list[k]]
                part = self.downsample_points(part, num_points=self.num_part_points)
                parts_end_list.append(torch.tensor(part.astype(np.float32), dtype=torch.float32, device=device))

            # --- Precompute edge-wise GT projection (both parts concatenated) ---
            # Upper-triangular edges
            src_ut, dst_ut = np.triu_indices(num_parts, k=1)
            E = src_ut.shape[0]
            P = self.num_part_points

            edge_dir_gt = np.zeros((E, 2 * P, 3), dtype=np.float32)  # (E,2048,3)
            edge_dist_gt = np.zeros((E, 2 * P, 1), dtype=np.float32) # (E,2048,1)

            # Ensure axis normalization for projection computation
            screw_axis_list_norm = screw_axis_list.copy().astype(np.float32)
            axis_norms = np.linalg.norm(screw_axis_list_norm, axis=1, keepdims=True) + 1e-6
            screw_axis_list_norm = screw_axis_list_norm / axis_norms

            # Fill connected edges in upper-tri scan order using counter into joint arrays
            counter = 0
            for e in range(E):
                i = int(src_ut[e])
                j = int(dst_ut[e])
                if parts_conne_gt_np[i, j] == 1.0:
                    axis = screw_axis_list_norm[counter]
                    pivot = screw_point_list[counter]

                    pc_i = parts_start_np_list[i]  # (1024,3)
                    pc_j = parts_start_np_list[j]  # (1024,3)

                    dir_i, dist_i = self.compute_gt_proj_dir_dist(pc_i, axis, pivot)
                    dir_j, dist_j = self.compute_gt_proj_dir_dist(pc_j, axis, pivot)

                    edge_dir_gt[e, :P, :] = dir_i
                    edge_dir_gt[e, P:, :] = dir_j
                    edge_dist_gt[e, :P, :] = dist_i
                    edge_dist_gt[e, P:, :] = dist_j

                    counter += 1
                # else remains zeros (non-edge)

            # Torch conversions
            pc_start_t = torch.tensor(pc_start, dtype=torch.float32, device=device)
            pc_end_t = torch.tensor(pc_end, dtype=torch.float32, device=device)

            joints_type_t = torch.tensor(joint_type_list, dtype=torch.float32, device=device)
            joints_screw_axis_t = torch.tensor(screw_axis_list_norm, dtype=torch.float32, device=device)
            joints_screw_point_t = torch.tensor(screw_point_list, dtype=torch.float32, device=device)

            adjacency_t = torch.tensor(adjacency_matrix, dtype=torch.float32, device=device)
            parts_conne_gt_t = torch.tensor(parts_conne_gt_np, dtype=torch.float32, device=device)
            angles_t = torch.tensor(angles_list, dtype=torch.float32, device=device)

            edge_src_t = torch.tensor(src_ut, dtype=torch.long, device=device)
            edge_dst_t = torch.tensor(dst_ut, dtype=torch.long, device=device)
            edge_dir_t = torch.tensor(edge_dir_gt, dtype=torch.float32, device=device)
            edge_dist_t = torch.tensor(edge_dist_gt, dtype=torch.float32, device=device)

            data_tuple = (
                pc_start_t, parts_start_list,
                pc_end_t, parts_end_list,
                adjacency_t, parts_conne_gt_t,
                joints_type_t, joints_screw_axis_t, joints_screw_point_t,
                angles_t, file_name,
                edge_src_t, edge_dst_t, edge_dir_t, edge_dist_t
            )
            data_list.append(data_tuple)

        self.pairs_list = data_list

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        return self.pairs_list[idx]

    # ---------------- helpers ----------------

    def downsample_points(self, points, num_points):
        """points: (N,3) numpy -> (num_points,3) numpy"""
        N = points.shape[0]
        if N == 0:
            return np.zeros((num_points, 3), dtype=np.float32)

        rng = np.random.default_rng(self.seed)
        if N >= num_points:
            idx = rng.choice(N, num_points, replace=False)
        else:
            idx = rng.choice(N, num_points, replace=True)
        return points[idx]

    def downsample_pc_masks(self, pc, seg_mask_dict, num_points=8192):
        """
        pc: (N,3) numpy
        seg_mask_dict: dict-like, keys 0..num_parts-1, each value is boolean mask (N,) or indices.
        Returns:
          pc_ds: (num_points,3)
          mask_list_ds: list length num_parts, each boolean mask (num_points,)
        """
        N = pc.shape[0]
        if N == 0:
            num_parts = len(seg_mask_dict)
            pc_ds = np.zeros((num_points, 3), dtype=np.float32)
            mask_list_ds = [np.zeros((num_points,), dtype=bool) for _ in range(num_parts)]
            return pc_ds, mask_list_ds

        # Normalize masks to boolean arrays
        num_parts = len(seg_mask_dict)
        masks_bool = []
        for k in range(num_parts):
            m = seg_mask_dict[k]
            m = np.asarray(m)
            if m.dtype == bool:
                masks_bool.append(m)
            else:
                # treat as indices
                mb = np.zeros((N,), dtype=bool)
                mb[m.astype(np.int64)] = True
                masks_bool.append(mb)

        rng = np.random.default_rng(self.seed)
        if N >= num_points:
            idx = rng.choice(N, num_points, replace=False)
        else:
            idx = rng.choice(N, num_points, replace=True)

        pc_ds = pc[idx]
        mask_list_ds = [m[idx] for m in masks_bool]
        return pc_ds.astype(np.float32), mask_list_ds

    def compute_gt_proj_dir_dist(self, pc, axis, pivot):
        """
        pc:    (P,3) numpy
        axis:  (3,) numpy, normalized
        pivot: (3,) numpy

        returns:
          dir_gt:  (P,3)
          dist_gt: (P,1)
        """
        axis = axis.astype(np.float32)
        axis = axis / (np.linalg.norm(axis) + 1e-6)

        v = pivot[None, :] - pc  # (P,3)
        proj_len = (v * axis[None, :]).sum(axis=1, keepdims=True)  # (P,1)
        v_perp = v - proj_len * axis[None, :]  # (P,3)

        dist = np.linalg.norm(v_perp, axis=1, keepdims=True).astype(np.float32)  # (P,1)
        dir_gt = (v_perp / (dist + 1e-6)).astype(np.float32)  # (P,3)
        return dir_gt, dist
