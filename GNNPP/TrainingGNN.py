
import torch
import numpy as np
import os
import sys
sys.path.append('/home/suhaib/superv_Articulation')
from utilis.dataset2 import PartsGraphDataset2, collate_graphs
from GNNPP.gnn_pointnet_network_v3 import parts_connection_mlp
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from utilis.Visualizer import VisualizerWrapper
from torch.utils.tensorboard import SummaryWriter

import datetime
import tqdm

# --- CONFIGURATION ---
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
EPOCHS = 20
NUM_WORKERS = 1



WEIGHT_DECAY = 1e-4
LR_DECAY_STEP = 10
LR_DECAY_GAMMA = 0.8

def training_step(model, data_dict):
    (
        pc_starts,
        parts_start_list,
        pc_ends,
        parts_end_list,
        adj,
        parts_connections_gt,
        joint_type_list_gt,
        screw_axis_list_gt,
        screw_point_list_gt,
        angles,
        file_name,
    ) = data_dict
    total_loss = 0.0

    adj = adj.squeeze()
    parts_connections_gt = parts_connections_gt.squeeze()
    screw_axis_list_gt = screw_axis_list_gt.squeeze().view(-1,3)
    screw_axis_list_gt = torch.abs(screw_axis_list_gt)
    screw_axis_list_gt = F.normalize(screw_axis_list_gt, dim=1)
    screw_point_list_gt = screw_point_list_gt.squeeze().view(-1,3)

    joint_type_list_gt = joint_type_list_gt.squeeze()
    angles = angles.squeeze().view(-1,1)
    # Forward pass
    edges_conne_pred, joint_type_pred, revolute_para_pred, prismatic_para_pred, z, (src, dst) = model(parts_start_list, parts_end_list, adj)
    conn_gt = parts_connections_gt[src, dst].float().unsqueeze(1)  # [num_edges, 1]
    loss_latent = F.mse_loss(z, angles, reduction='mean')
    # print(f"Adj: \n{adj}\n")
    # print(f"Parts conn gt: \n{conn_gt}\n")
    x = torch.sigmoid(edges_conne_pred)
    x = (x > 0.5).float()
    # print(f"Edges conn pred: \n{x}\n")
    loss_part_conn = F.binary_cross_entropy_with_logits(edges_conne_pred, conn_gt, reduction='sum')
    
    joint_mask = conn_gt.squeeze(1) > 0  # boolean mask of edges that exist
    if joint_mask.sum() > 0:
        joint_type_pred_valid = joint_type_pred[joint_mask]
        joint_type_list_gt = joint_type_list_gt.reshape(-1,1)
        loss_joint_type = F.binary_cross_entropy_with_logits(joint_type_pred_valid, joint_type_list_gt, reduction='sum')
    else:
        loss_joint_type = torch.tensor(0.0, device=adj.device)

    revolute_mask = (joint_type_list_gt == 0)  # 0 = revolute
    prismatic_mask = (joint_type_list_gt == 1)  # 1 = prismatic

    # Compute per-edge parameter losses (L2)
    revolute_axis_pred = revolute_para_pred[:,:,:3][joint_mask].squeeze()
    rev_weights = torch.sigmoid(revolute_para_pred[:,:,3:4][joint_mask])
    revolute_axis_pred = (revolute_axis_pred * rev_weights).sum(dim=1) / (rev_weights.sum(dim=1) + 1e-6)
    revolute_axis_pred = F.normalize(revolute_axis_pred, dim=1)
    
    prismatic_axis_pred = prismatic_para_pred[:,:,:3][joint_mask].squeeze()
    pri_weights = torch.sigmoid(prismatic_para_pred[:,:,3:4][joint_mask])
    prismatic_axis_pred = (prismatic_axis_pred * pri_weights).sum(dim=1) / (pri_weights.sum(dim=1) + 1e-6)
    prismatic_axis_pred = F.normalize(prismatic_axis_pred, dim=1)

    revolute_axis_loss = torch.sqrt(F.mse_loss(revolute_axis_pred, screw_axis_list_gt, reduction='none').clamp(min=1e-12)).mean(1)
    revolute_loss = revolute_axis_loss

    prismatic_loss = torch.sqrt(F.mse_loss(prismatic_axis_pred, screw_axis_list_gt, reduction='none').clamp(min=1e-12)).mean(1)

    # Apply masks

    revolute_loss = (revolute_loss * revolute_mask.float().view(-1))
    revolute_loss = revolute_loss.mean()
    prismatic_loss = (prismatic_loss * prismatic_mask.float().view(-1))
    prismatic_loss = prismatic_loss.mean()
    
    total_loss = loss_part_conn + loss_joint_type + revolute_loss + prismatic_loss + loss_latent

    return total_loss, loss_part_conn, loss_joint_type, revolute_loss, prismatic_loss, loss_latent



# --- 4. Training Step ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = PartsGraphDataset2("../Ditto/Articulated_object_simulation-main/data/Shape2Motion_gcn/robotic_arm2/train/scenes/*.npz",device)
val_dataset = PartsGraphDataset2("../Ditto/Articulated_object_simulation-main/data/Shape2Motion_gcn/robotic_arm2/val/scenes/*.npz",device)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


start_training_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
checkpoint_path = f"./pre_trained_models_gcnpp/{start_training_time}_robotic_arm"
os.makedirs(checkpoint_path, exist_ok=True)
writer = SummaryWriter(f'runs/{start_training_time}')
print(f"tensorboard --logdir '/home/suhaib/superv_Articulation/runs/{start_training_time}'")
print(f"Checkpoints will be saved to: {checkpoint_path}")
params = {
    "pointnet_dim": 1024,
    "nlayers": 4,
    "nhidden": 256,
    "out_dim": 128,
    "dropout": 0.3,
    "lamda": 0.5,
    "alpha": 0.1,
    "variant": True,
    "nhidden_mlp": 128,
    "n_class": 1,
    "latent_dim": 1,
    "decoder_out_dim": 128,
    "motion_decoder_out_dim": 256,
}

model = parts_connection_mlp(**params).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)
model.train()

global_step = 0
best_loss_train = float('inf')
best_loss_val = float('inf')


for epoch in range(200):
    current_lr = scheduler.get_last_lr()[0]
    print(f"\n=== Epoch {epoch+1} | LR: {current_lr:.6f} ===")
    epoch_loss = 0.0
    epoch_loss_part_conn = 0.0
    epoch_loss_joint_type = 0.0
    epoch_revolute_loss = 0.0
    epoch_prismatic_loss = 0.0
    epoch_loss_latent = 0.0

    pbar_train = tqdm.tqdm(train_dataloader, desc=f"Train Ep {epoch+1}", leave=True, dynamic_ncols=True)
    for step, data in enumerate(pbar_train, start=1):
        if epoch == 0 and step == 1:
            # Make sure 'data' is a tensor or tuple of tensors
            try:
                (
                    pc_starts,
                    parts_list,
                    adj,
                    parts_connections_gt,
                    joint_type_list_gt,
                    screw_axis_list_gt,
                    screw_point_list_gt,
                ) = data

                adj = adj.squeeze(0)
                parts_connections_gt = parts_connections_gt.squeeze(0)
                writer.add_graph(model, (parts_list,adj))
            except Exception as e:
                print("Skipping add_graph:", e)

        optimizer.zero_grad()
        total_loss, loss_part_conn, loss_joint_type, revolute_loss, prismatic_loss, loss_latent = training_step(model, data)
        total_loss.backward()
        optimizer.step()

        # Accumulate
        epoch_loss += total_loss.item()
        epoch_loss_part_conn += loss_part_conn.item()
        epoch_loss_joint_type += loss_joint_type.item()
        epoch_revolute_loss += revolute_loss.item()
        epoch_prismatic_loss += prismatic_loss.item()
        epoch_loss_latent += loss_latent.item()

        global_step += 1

        # Average losses up to this point
        avg_total_loss = epoch_loss / step
        avg_loss_part_conn = epoch_loss_part_conn / step
        avg_loss_joint_type = epoch_loss_joint_type / step
        avg_revolute_loss = epoch_revolute_loss / step
        avg_prismatic_loss = epoch_prismatic_loss / step
        avg_loss_latent = epoch_loss_latent / step

        pbar_train.set_postfix({
            "Total": f"{total_loss:.4f}",
            "conn": f"{loss_part_conn:.4f}",
            "type": f"{loss_joint_type:.4f}",
            "Rev": f"{revolute_loss:.4f}",
            "Pri": f"{prismatic_loss:.4f}",
            "Lat": f"{loss_latent:.4f}"
        })


    # TensorBoard
    writer.add_scalar('Loss/train', avg_total_loss, epoch)
    writer.add_scalar('Loss/part_conn', avg_loss_part_conn, epoch)
    writer.add_scalar('Loss/joint_type', avg_loss_joint_type, epoch)
    writer.add_scalar('Loss/revolute', avg_revolute_loss, epoch)
    writer.add_scalar('Loss/prismatic', avg_prismatic_loss, epoch)
    writer.add_scalar('Loss/latent', avg_loss_latent, epoch)


    model.eval()  # turn off dropout, etc.
    val_loss = 0.0
    val_loss_part_conn = 0.0
    val_loss_joint_type = 0.0
    val_revolute_loss = 0.0
    val_prismatic_loss = 0.0
    val_loss_latent = 0.0

    pbar_val = tqdm.tqdm(val_dataloader, desc=f"Val Ep {epoch+1}", leave=True, dynamic_ncols=True)
    with torch.no_grad():  # no gradient computation
        for val_step, val_data in enumerate(pbar_val, start=1):
            total_loss, loss_part_conn, loss_joint_type, revolute_loss, prismatic_loss,loss_latent = training_step(model, val_data)

            val_loss += total_loss.item()
            val_loss_part_conn += loss_part_conn.item()
            val_loss_joint_type += loss_joint_type.item()
            val_revolute_loss += revolute_loss.item()
            val_prismatic_loss += prismatic_loss.item()
            val_loss_latent += loss_latent.item()

            pbar_val.set_postfix({
                "Total": f"{val_loss:.4f}",
                "conn": f"{val_loss_part_conn:.4f}",
                "type": f"{val_loss_joint_type:.4f}",
                "Rev": f"{val_revolute_loss:.4f}",
                "Pri": f"{val_prismatic_loss:.4f}",
                "Lat": f"{val_loss_latent:.4f}"
            })

    val_loss /= len(val_dataloader)
    val_loss_part_conn /= len(val_dataloader)
    val_loss_joint_type /= len(val_dataloader)
    val_revolute_loss /= len(val_dataloader)
    val_prismatic_loss /= len(val_dataloader)
    val_loss_latent /= len(val_dataloader)

    # Log to TensorBoard
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/part_conn', val_loss_part_conn, epoch)
    writer.add_scalar('Val/joint_type', val_loss_joint_type, epoch)
    writer.add_scalar('Val/revolute', val_revolute_loss, epoch)
    writer.add_scalar('Val/prismatic', val_prismatic_loss, epoch)
    writer.add_scalar('Val/latent', val_loss_latent, epoch)


    model.train()
    scheduler.step()

    print(f"Summary Ep {epoch+1}:\n"
        f"Train Loss: {avg_total_loss:.4f} | "   
        f"Val Loss: {val_loss:.4f}")
    
    if avg_total_loss < best_loss_train:
        best_loss_train = avg_total_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f"chkpt_best_model_train.pth"))
    
    if val_loss < best_loss_val:
        best_loss_val = val_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f"chkpt_best_model_val.pth"))

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f"chkpt_{epoch}.pth"))