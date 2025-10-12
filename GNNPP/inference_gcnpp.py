import numpy as np
import torch
import sys
sys.path.append('/home/suhaib/superv_Articulation')
from GNNPP.gnn_pointnet_network import parts_connection_mlp
import glob
import torch.nn.functional as F



def downsample_pc_masks( points, masks_list=None, num_points=1024):
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

file_paths = "../Ditto/Articulated_object_simulation-main/data/Shape2Motion_local/cabinet_val/scenes/*.npz"
data_list = []
for f in glob.glob(file_paths):
        data = np.load(f, allow_pickle=True)
        mask_start_list = []
        num_joints = int((len(data)-2)/18)

        pc_start = data[f'pc_start_0']
        adjacency_matrix = data['adj']
        parts_conne_gt = data['parts_conne_gt']

        for joint in range(num_joints):
            mask_start = data[f'pc_seg_start_{joint}']
            mask_start_list.append(mask_start)

        base_mask = data['pc_seg_start_base']
        mask_start_list.insert(0, base_mask)


        pc_start, mask_start_list = downsample_pc_masks(pc_start, mask_start_list)

        bound_max = pc_start.max(0)
        bound_min = pc_start.min(0)
        center = (bound_min + bound_max) / 2
        scale = (bound_max - bound_min).max()
        pc_start = (pc_start - center) / scale

        
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32).cuda()
        parts_conne_gt = torch.tensor(parts_conne_gt, dtype=torch.float32).cuda()

        parts_list = []
        for mask in range(num_joints+1):
            part = pc_start[mask_start_list[mask]]
            part = downsample_pc_masks(part)
            part = torch.tensor(part, dtype=torch.float32).cuda().unsqueeze(0)
            parts_list.append(part)

        pc_start = torch.tensor(pc_start, dtype=torch.float32).cuda()

        data_tuple = (pc_start, parts_list, adjacency_matrix, parts_conne_gt)

        data_list.append(data_tuple)
np.random.seed()
index = np.random.randint(0, len(data_list))
print(index)
data = data_list[index]
pc_start = data[0]
parts_list = data[1]
adj = data[2]
parts_conne_gt = data[3]

adj = adj.squeeze(0)
parts_connections = parts_conne_gt.squeeze(0)

params = {
    "pointnet_dim": 1024,
    "nlayers": 4,
    "nhidden": 256,
    "out_dim": 64,
    "dropout": 0.3,
    "lamda": 0.5,
    "alpha": 0.1,
    "variant": True,
    "nhidden_mlp": 128,
    "n_class": 1,
}

weights_path = "/home/suhaib/superv_Articulation/pre_trained_models_gcnpp/2025-10-10 00:07:30/chkpt_130.pth"
model = parts_connection_mlp(**params).cuda()
model.load_state_dict(torch.load(weights_path))
model.eval()

with torch.no_grad():
    edges_conne_pred, (src, dst) = model(parts_list, adj)

targets = parts_conne_gt[src, dst].float()  # [num_edges, 1]
print(src)
print(dst)
print(f"Edge connections pred: \n{torch.sigmoid(edges_conne_pred) > 0.5}")
print(adj)
print(f"Edge connections gt: \n{parts_conne_gt}")