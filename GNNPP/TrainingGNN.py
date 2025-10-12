
import torch
import numpy as np
import os
import sys
sys.path.append('/home/suhaib/superv_Articulation')
from utilis.dataset2 import PartsGraphDataset, collate_graphs
from GNNPP.gnn_pointnet_network import parts_connection_mlp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utilis.Visualizer import VisualizerWrapper
import datetime
import tqdm



def training_step(model, data_dict):
    # pc_starts = data_dict["pc_start"]         # torch.Tensor, shape (N,3)
    # parts_lists = data_dict["parts_list"]
    # adjs = data_dict["adj"]
    # parts_connections = data_dict["parts_connections"]
    pc_starts, parts_list, adj, parts_connections, joint_type_list_gt = data_dict

    total_loss = 0.0

    adj = adj.squeeze(0)
    parts_connections = parts_connections.squeeze(0)
    edges_conne_pred, joint_type_pred, (src, dst) = model(parts_list, adj)

    targets = parts_connections[src, dst].float()  # [num_edges, 1]
    targets.unsqueeze_(1)
    joint_type_list_gt = joint_type_list_gt.float()
    joint_type_list_gt.unsqueeze_(1)

    loss_part_conn = F.binary_cross_entropy_with_logits(edges_conne_pred, targets, reduction='sum')
    loss_joint_type = F.binary_cross_entropy_with_logits(joint_type_pred, joint_type_list_gt, reduction='sum')
    loss = loss_part_conn + loss_joint_type
    
    total_loss += loss
    
    return total_loss

# --- 4. Training Step ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PartsGraphDataset("../Ditto/Articulated_object_simulation-main/data/Shape2Motion_local/cabinet/scenes/*.npz",device)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

start_training_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
checkpoint_path = f"./pre_trained_models_gcnpp/{start_training_time}"
os.makedirs(checkpoint_path, exist_ok=True)

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

model = parts_connection_mlp(**params).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

for epoch in tqdm.tqdm(range(200)):
    epoch_loss = 0.0
    step = 0
    for data in dataloader:
        step+=1
        # data_dict = {
        #     "pc_start": pc_start, "parts_list": parts_list,
        #     "Adjacency_matrix": Adjacency_matrix, 
        #     "parts_connections": parts_connections,
        #     }
        optimizer.zero_grad()
        loss = training_step(model, data)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if step % 10 == 0:
            print(f"Epoch {epoch}, Avg Loss: {epoch_loss/step:.4f}, step: {step}")
    
    if epoch+1 % 10 == 0:
        weights_path = os.path.join(checkpoint_path, f"chkpt_{epoch}.pth")
        torch.save(model.state_dict(), weights_path)
