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


class PartsGraphDataset(Dataset):
    def __init__(self, file_paths,device):
        # file_paths = "./data/Shape2Motion_gcn/*/scenes/*.npz"
        data_list = []
        for f in sorted(glob.glob(file_paths)):
            data = np.load(f, allow_pickle=True)
            mask_start_list = []
            joint_type_list = []
            screw_axis_list = []
            screw_point_list = []
            num_joints = int((len(data)-4)/18)

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
                screw_moment = data[f'screw_moment_{joint}']
                screw_point = np.cross(screw_axis, screw_moment)
                screw_axis_list.append(screw_axis)
                screw_point_list.append(screw_point)


            base_mask = data['pc_seg_start_base']
            mask_start_list.insert(0, base_mask)


            pc_start, mask_start_list = self.downsample_pc_masks(pc_start, mask_start_list)

            bound_max = pc_start.max(0)
            bound_min = pc_start.min(0)
            center = (bound_min + bound_max) / 2
            scale = (bound_max - bound_min).max()
            pc_start = (pc_start - center) / scale


            joints_type = torch.tensor(np.array(joint_type_list), dtype=torch.float32).to(device)
            joints_screw_axis = torch.tensor(np.array(screw_axis_list), dtype=torch.float32).to(device)
            joints_screw_point = torch.tensor(np.array(screw_point_list), dtype=torch.float32).to(device)
            adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32).to(device)
            parts_conne_gt = torch.tensor(parts_conne_gt, dtype=torch.float32).to(device)


            parts_list = []
            for mask in range(num_joints+1):
                part = pc_start[mask_start_list[mask]]
                part = self.downsample_pc_masks(part)
                part = torch.tensor(part, dtype=torch.float32).to(device)
                parts_list.append(part)

            pc_start = torch.tensor(pc_start, dtype=torch.float32).to(device)

            data_tuple = (pc_start, parts_list, adjacency_matrix, parts_conne_gt, joints_type, joints_screw_axis, joints_screw_point)

            data_list.append(data_tuple)
        
        self.pairs_list = data_list

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        pairs_of_pcs = self.pairs_list[idx]

        pc_start = pairs_of_pcs[0]
        parts_list = pairs_of_pcs[1]
        adjacency_matrix = pairs_of_pcs[2]
        parts_conne_gt = pairs_of_pcs[3]
        joint_type_list = pairs_of_pcs[4]
        screw_axis_list = pairs_of_pcs[5]
        screw_point_list = pairs_of_pcs[6]

        return pc_start, parts_list, adjacency_matrix, parts_conne_gt, joint_type_list, screw_axis_list, screw_point_list
    
    def downsample_pc_masks(self, points, masks_list=None, num_points=1024):
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