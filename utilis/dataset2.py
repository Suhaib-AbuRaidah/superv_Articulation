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