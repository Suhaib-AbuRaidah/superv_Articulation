from Network_archit import ArticulationNet, training_step
import torch
import numpy as np
import os
from utilis.dataset2 import CustomDataset
from torch.utils.data import DataLoader
from utilis.Visualizer import VisualizerWrapper
import datetime
import tqdm

dataset = CustomDataset("../Ditto/Articulated_object_simulation-main/data/syn_local/mix/scenes/*.npz")
dataloader = DataLoader(dataset, batch_size=12, shuffle=False)

start_training_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
checkpoint_path = f"./pre_trained_models/{start_training_time}"
os.makedirs(checkpoint_path, exist_ok=True)

model = ArticulationNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
# print(model)
# viz = VisualizerWrapper()
# loss = torch.tensor(10)
for epoch in tqdm.tqdm(range(200)):
    epoch_loss = 0.0
    num_batches = 0

    for pc_start, pc_target, _, _, joint_type, screw_axis, state_start, state_target,screw_moment, p2l_vec, p2l_dist in dataloader:
        pc_start = pc_start.cuda().float()
        pc_target = pc_target.cuda().float()
        joint_type = joint_type.cuda().float()
        screw_axis = screw_axis.cuda().float()
        state_start = state_start.cuda().float()
        state_target = state_target.cuda().float()
        screw_moment = screw_moment.cuda().float()
        p2l_vec = p2l_vec.cuda().float()
        p2l_dist = p2l_dist.cuda().float()


        data_dict = {
            
            "pc_start": pc_start, "pc_target": pc_target,
            "joint_type": joint_type, "screw_axis": screw_axis,
            "state_start": state_start, "state_end": state_target,
            "screw_moment": screw_moment, "p2l_vec": p2l_vec,
            "p2l_dist": p2l_dist,
            
        }
        
        optimizer.zero_grad()
        loss = training_step(model, data_dict)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        avg_loss = epoch_loss / num_batches
        if num_batches % 12 == 0:
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
    
    if epoch % 10 == 0:
        weights_path = os.path.join(checkpoint_path, f"chkpt_{epoch}.pth")
        torch.save(model.state_dict(), weights_path)
