
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.expanduser('~/superv_Articulation'))
from utilis.dataset2 import PartsGraphDataset3, collate_graphs
from GNNPP.gnn_pointnet2_network_v3 import parts_connection_mlp
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from utilis.Visualizer import VisualizerWrapper
from torch.utils.tensorboard import SummaryWriter
# import open3d as o3d
import datetime
import tqdm

# --- CONFIGURATION ---
BATCH_SIZE = 1
LEARNING_RATE = 5e-4
EPOCHS = 20
NUM_WORKERS = 1



WEIGHT_DECAY = 0.0
LR_DECAY_STEP = 4
LR_DECAY_GAMMA = 0.8

def canonical_direction(z):
    threshold = 0.10
    mask = (
        (z[:, 2] >= threshold) |
        ((z[:, 2] <= threshold) & (z[:, 1] >= threshold)) |
        ((z[:, 2] <= threshold) & (z[:, 1] <= threshold) & (z[:, 0] >= threshold))
    )

    return torch.where(mask.unsqueeze(1), z, -z)

def point_to_axis_distance(point, axis_point, axis_dir):
    """
    point:      (N,3) predicted pivot
    axis_point: (N,3) GT pivot
    axis_dir:   (N,3) GT axis (normalized)
    """
    v = point - axis_point
    proj = torch.sum(v * axis_dir, dim=1, keepdim=True) * axis_dir
    perp = v - proj
    return torch.norm(perp, dim=1)
    
def training_step(model, data_dict):
    (
        pc_starts,
        parts_start_list,
        pc_ends,
        parts_end_list,
        pc_start_unsampled,
        pc_end_unsampled,
        seg_mask_start_unsampled, 
        seg_mask_end_unsampled,
        adj,
        parts_connections_gt,
        joint_type_list_gt,
        screw_axis_list_gt,
        screw_point_list_gt,
        angles,
        file_name,
        src, dst,
        edge_dir_gt, edge_dst_gt,
    ) = data_dict

    total_loss = 0.0

    adj = adj.squeeze()
    parts_connections_gt = parts_connections_gt.squeeze()
    screw_axis_list_gt = screw_axis_list_gt.squeeze().view(-1, 3)
    screw_axis_list_gt = F.normalize(screw_axis_list_gt, dim=1)

    screw_point_list_gt = screw_point_list_gt.squeeze().view(-1, 3)
    joint_type_list_gt = joint_type_list_gt.squeeze()

    # keep GT edge projection tensors as-is (do NOT reshape to (-1,3) / flatten)
    # expected (E, P_edge, 3) and (E, P_edge, 1) or (E, P_edge)
    edge_dir_gt = edge_dir_gt.squeeze(0)
    edge_dst_gt = edge_dst_gt.squeeze(0)

    angles = angles.squeeze().view(-1, 1)

    # Forward pass
    edges_conne_pred, joint_type_pred, revolute_para_pred, prismatic_para_pred, (src, dst) = model(parts_start_list, parts_end_list, adj)

    edges_conne_pred = edges_conne_pred.mean(dim=1)
    joint_type_pred = joint_type_pred.mean(dim=1)

    conn_gt = parts_connections_gt[src, dst].float().unsqueeze(1)  # [num_edges, 1]

    loss_part_conn = F.binary_cross_entropy_with_logits(edges_conne_pred, conn_gt, reduction='sum')
    # print(f"loss_part_conn:\n{loss_part_conn}")
    joint_mask = conn_gt.squeeze(1) > 0  # boolean mask of edges that exist
    if joint_mask.sum() > 0:
        joint_type_pred_valid = joint_type_pred[joint_mask]
        joint_type_list_gt = joint_type_list_gt.reshape(-1, 1)
        loss_joint_type = F.binary_cross_entropy_with_logits(joint_type_pred_valid, joint_type_list_gt, reduction='sum')
        # print(f"loss_joint_type:\n{loss_joint_type}")
    else:
        loss_joint_type = torch.tensor(0.0, device=adj.device)

    revolute_mask = (joint_type_list_gt == 0)  # 0 = revolute
    prismatic_mask = (joint_type_list_gt == 1)  # 1 = prismatic
    # Revolute axis prediction (unchanged)
    revolute_axis_pred = revolute_para_pred[:, :, :3][joint_mask].squeeze()
    rev_weights = torch.sigmoid(revolute_para_pred[:, :, 3:4][joint_mask])
    revolute_axis_pred = (revolute_axis_pred * rev_weights).sum(dim=1) / (rev_weights.sum(dim=1) + 1e-6)
    revolute_axis_pred = F.normalize(revolute_axis_pred, dim=1)
    # --- ADDED: pivot projection direction + distance losses (paper-style) ---
    # Predicted projection direction d_p and signed distance h_p (per-point)
    edge_dir_pred = revolute_para_pred[:, :, 4:7][joint_mask]              # (E_joint, P_edge, 3)
    edge_dir_pred = F.normalize(edge_dir_pred, dim=-1)

    edge_dst_pred = revolute_para_pred[:, :, 7:8][joint_mask]              # (E_joint, P_edge, 1)


    # GT for the same edges (upper-tri ordering). Keep only existing edges, Distribu
    edge_dir_gt_j = edge_dir_gt[joint_mask]                                # (E_joint, P_edge, 3)
    edge_dst_gt_j = edge_dst_gt[joint_mask]                                # (E_joint, P_edge, 1) or (E_joint, P_edge)

    if edge_dst_gt_j.dim() == 2:
        edge_dst_gt_j = edge_dst_gt_j.unsqueeze(-1)


    # parts_start_list: list of (1,P,3) or (P,3) tensors (as used in your codebase)
    pcs1 = torch.cat([pc.transpose(1, 2) for pc in parts_start_list], dim=0)        # (N,3,P)

    # build per-edge union points (E,2P,3)
    edge_points = torch.cat([pcs1[src], pcs1[dst]], dim=2).transpose(1, 2)          # (E,2P,3)
    edge_points = edge_points[joint_mask]                                           # (E_joint,2P,3)

    # predicted per-point footpoint on/near axis: q = p + h*d
    edge_pivot_per_point = edge_points + edge_dir_pred * edge_dst_pred              # (E_joint,2P,3)
    # ground truth per-point footpoint on/near axis: q_gt = p + h_gt*d_gt
    edge_pivot_per_point_gt = edge_points + edge_dir_gt_j * edge_dst_gt_j   # (E_joint,2P,3)


    # --- Approach 2: cross-part closest point (A -> B and B -> A) ---
    P = edge_points.shape[1] // 2
    pc_A, pc_B = edge_points[:, :P, :], edge_points[:, P:, :]                       # (E_joint,P,3)
    q_A, q_B   = edge_pivot_per_point[:, :P, :], edge_pivot_per_point[:, P:, :]     # (E_joint,P,3)

    D_A = torch.cdist(q_A, pc_B)                                                    # (E_joint,P,P)
    D_B = torch.cdist(q_B, pc_A)                                                    # (E_joint,P,P)

    min_dist_A, _ = D_A.min(dim=2)                                                  # (E_joint,P)
    min_dist_B, _ = D_B.min(dim=2)                                                  # (E_joint,P)

    # per-edge loss (E_joint,)
    cross_min_dist = 0.5 * (min_dist_A.mean(dim=1) + min_dist_B.mean(dim=1))
    # print(f"cross_min_dist:\n{cross_min_dist}")

    pivot_consistancy_loss = cross_min_dist

    rev_edge_pivots = edge_pivot_per_point.mean(dim=1)  # (E_joint, 3)
    rev_edge_pivots_gt = edge_pivot_per_point_gt.mean(dim=1)  # (E_joint, 3)

    pivot_hard_loss = torch.sqrt(F.mse_loss(rev_edge_pivots, rev_edge_pivots_gt, reduction='none')).mean(dim=1)  # (E_joint,)
    # print(f"pivot_hard_loss:\n{pivot_hard_loss}")

    # Direction loss: arccos(d · d̂)
    cos_sim = torch.sum(edge_dir_pred * edge_dir_gt_j, dim=-1)             # (E_joint, P_edge)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
    loss_pivot_dir = torch.acos(cos_sim).mean(dim=1)                       # (E_joint,)

    # print(f"loss_pivot_dir:\n{loss_pivot_dir}")
    # Distance loss: |h - ĥ|
    loss_pivot_dst = torch.abs(edge_dst_pred - edge_dst_gt_j).squeeze(-1)  # (E_joint, P_edge)
    loss_pivot_dst = loss_pivot_dst.mean(dim=1)                            # (E_joint,)
    # print(f"loss_pivot_dst:\n{loss_pivot_dst}")


    # revolute_pivot_loss = loss_pivot_dir + loss_pivot_dst + pivot_consistancy_loss + pivot_hard_loss    # (E_joint,)
    revolute_pivot_loss = loss_pivot_dir + loss_pivot_dst   # (E_joint,)
    # ----------------------------------------------------------------------

    # Prismatic axis prediction (unchanged)
    prismatic_axis_pred = prismatic_para_pred[:, :, :3][joint_mask].squeeze()
    pri_weights = torch.sigmoid(prismatic_para_pred[:, :, 3:4][joint_mask])
    prismatic_axis_pred = (prismatic_axis_pred * pri_weights).sum(dim=1) / (pri_weights.sum(dim=1) + 1e-6)
    prismatic_axis_pred = F.normalize(prismatic_axis_pred, dim=1)
    # print(f"revolute_axis_pred:\n{revolute_axis_pred}")
    # print(f"screw_axis_list_gt:\n{screw_axis_list_gt[revolute_mask.squeeze(1)]}\n")
    revolute_axis_loss = 1 - torch.abs(torch.sum(revolute_axis_pred * screw_axis_list_gt, dim=1)).mean()
    # print(f"revolute_axis_loss:\n{revolute_axis_loss}\n")
    revolute_loss = revolute_axis_loss + revolute_pivot_loss

    # print(f"prismatic_axis_pred:\n{prismatic_axis_pred}")
    # print(f"screw_axis_list_gt:\n{screw_axis_list_gt[prismatic_mask.squeeze(1)]}\n")
    prismatic_loss = 1 - torch.abs(torch.sum(prismatic_axis_pred * screw_axis_list_gt, dim=1)).mean()
    # print(f"prismatic_loss:\n{prismatic_loss}\n")
    # Apply masks (keep your structure; revolute_pivot_loss is per-edge now)
    revolute_loss = (revolute_loss * revolute_mask.float().view(-1))
    revolute_loss = revolute_loss.mean()

    revolute_axis_loss = (revolute_axis_loss * revolute_mask.float().view(-1))
    revolute_axis_loss = revolute_axis_loss.mean()

    revolute_pivot_loss = (revolute_pivot_loss * revolute_mask.float().view(-1))
    revolute_pivot_loss = revolute_pivot_loss.mean()

    prismatic_loss = (prismatic_loss * prismatic_mask.float().view(-1))
    prismatic_loss = prismatic_loss.mean()

    total_loss = loss_part_conn + loss_joint_type + revolute_loss + prismatic_loss
    return total_loss, loss_part_conn, loss_joint_type, revolute_loss, revolute_axis_loss, revolute_pivot_loss, prismatic_loss




# --- 4. Training Step ---

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = PartsGraphDataset3(os.path.expanduser("~/superv_Articulation/data/Shape2Motion_gcn/LRW/*/train/scenes/*.npz"),device)
val_dataset = PartsGraphDataset3(os.path.expanduser("~/superv_Articulation/data/Shape2Motion_gcn/LRW/*/val/scenes/*.npz"),device)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

start_training_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
checkpoint_path = f"~/superv_Articulation/pre_trained_models_gcnpp/{start_training_time}_LRW"
checkpoint_path = os.path.expanduser(checkpoint_path)
os.makedirs(checkpoint_path, exist_ok=True)
writer = SummaryWriter(f'runs/{start_training_time}')
print(f"tensorboard --logdir '~/superv_Articulation/runs/{start_training_time}'")
print(f"Checkpoints will be saved to: {checkpoint_path}")
params = {
    "pointnet_dim": 1024,
    "nlayers": 4,
    "nhidden": 512,
    "out_dim": 512,
    "dropout": 0.3,
    "lamda": 0.5,
    "alpha": 0.1,
    "variant": True,
    "nhidden_mlp": 512,
    "n_class": 1,
    "latent_dim": 1,
    "decoder_out_dim": 128,
    "motion_decoder_out_dim": 128,
}

model = parts_connection_mlp(**params).to(device)
state = torch.load(os.path.expanduser("~/superv_Articulation/pre_trained_models_gcnpp/2026-02-18 17:10:53_C/chkpt_best_model_train.pth"), map_location=device)
model.load_state_dict(state["model_state"])
# Count trainable parameters only
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {trainable_params:,}')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)
model.train()

global_step = 0
best_loss_train = float('inf')
best_loss_val = float('inf')


for epoch in range(EPOCHS):
    current_lr = scheduler.get_last_lr()[0]
    print(f"\n=== Epoch {epoch+1} | LR: {current_lr:.6f} ===")
    epoch_loss = 0.0
    epoch_loss_part_conn = 0.0
    epoch_loss_joint_type = 0.0
    epoch_revolute_axis_loss = 0.0
    epoch_revolute_pivot_loss = 0.0
    epoch_prismatic_loss = 0.0
    # epoch_loss_latent = 0.0

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
        total_loss, loss_part_conn, loss_joint_type, revolute_loss, revolute_axis_loss, revolute_pivot_loss, prismatic_loss = training_step(model, data)
        total_loss.backward()
        optimizer.step()

        # Accumulate
        epoch_loss += total_loss.item()
        epoch_loss_part_conn += loss_part_conn.item()
        epoch_loss_joint_type += loss_joint_type.item()
        epoch_revolute_axis_loss += revolute_axis_loss.item()
        epoch_revolute_pivot_loss += revolute_pivot_loss.item()
        epoch_prismatic_loss += prismatic_loss.item()

        # epoch_loss_latent += loss_latent.item()

        global_step += 1

        # Average losses up to this point
        avg_total_loss = epoch_loss / step
        avg_loss_part_conn = epoch_loss_part_conn / step
        avg_loss_joint_type = epoch_loss_joint_type / step
        avg_revolute_axis_loss = epoch_revolute_axis_loss / step
        avg_revolute_pivot_loss = epoch_revolute_pivot_loss / step
        avg_prismatic_loss = epoch_prismatic_loss / step
        # avg_loss_latent = epoch_loss_latent / step

        pbar_train.set_postfix({
            "Total": f"{total_loss:.4f}",
            "conn": f"{loss_part_conn:.4f}",
            "type": f"{loss_joint_type:.4f}",
            "Rev Axis": f"{revolute_axis_loss:.4f}",
            "Rev Pivot": f"{revolute_pivot_loss:.4f}",
            "Pri": f"{prismatic_loss:.4f}",
            # "Lat": f"{loss_latent:.4f}"
        })


    # TensorBoard
    writer.add_scalar('Loss/train', avg_total_loss, epoch)
    writer.add_scalar('Loss/part_conn', avg_loss_part_conn, epoch)
    writer.add_scalar('Loss/joint_type', avg_loss_joint_type, epoch)
    writer.add_scalar('Loss/revolute Axis', avg_revolute_axis_loss, epoch)
    writer.add_scalar('Loss/revolute Pivot', avg_revolute_pivot_loss, epoch)
    writer.add_scalar('Loss/prismatic', avg_prismatic_loss, epoch)
    # writer.add_scalar('Loss/latent', avg_loss_latent, epoch)


    model.eval()  # turn off dropout, etc.
    val_loss = 0.0
    val_loss_part_conn = 0.0
    val_loss_joint_type = 0.0
    val_revolute_axis_loss = 0.0
    val_revolute_pivot_loss = 0.0
    val_prismatic_loss = 0.0
    # val_loss_latent = 0.0

    pbar_val = tqdm.tqdm(val_dataloader, desc=f"Val Ep {epoch+1}", leave=True, dynamic_ncols=True)
    with torch.no_grad():  # no gradient computation
        for val_step, val_data in enumerate(pbar_val, start=1):
            total_loss, loss_part_conn, loss_joint_type, revolute_loss,revolute_axis_loss, revolute_pivot_loss, prismatic_loss = training_step(model, val_data)

            val_loss += total_loss.item()
            val_loss_part_conn += loss_part_conn.item()
            val_loss_joint_type += loss_joint_type.item()
            val_revolute_axis_loss += revolute_axis_loss.item()
            val_revolute_pivot_loss += revolute_pivot_loss.item()
            val_prismatic_loss += prismatic_loss.item()
            # val_loss_latent += loss_latent.item()

            pbar_val.set_postfix({
                "Total": f"{total_loss:.4f}",
                "conn": f"{loss_part_conn:.4f}",
                "type": f"{loss_joint_type:.4f}",
                "Rev Axis": f"{revolute_axis_loss:.4f}",
                "Rev Pivot": f"{revolute_pivot_loss:.4f}",
                "Pri": f"{prismatic_loss:.4f}",
                # "Lat": f"{loss_latent:.4f}"
            })

    val_loss /= len(val_dataloader)
    val_loss_part_conn /= len(val_dataloader)
    val_loss_joint_type /= len(val_dataloader)
    val_revolute_axis_loss /= len(val_dataloader)
    val_revolute_pivot_loss /= len(val_dataloader)
    val_prismatic_loss /= len(val_dataloader)
    # val_loss_latent /= len(val_dataloader)

    # Log to TensorBoard
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/part_conn', val_loss_part_conn, epoch)
    writer.add_scalar('Val/joint_type', val_loss_joint_type, epoch)
    writer.add_scalar('Val/revolute', val_revolute_axis_loss, epoch)
    writer.add_scalar('Val/revolute_pivot', val_revolute_pivot_loss, epoch)
    writer.add_scalar('Val/prismatic', val_prismatic_loss, epoch)
    # writer.add_scalar('Val/latent', val_loss_latent, epoch)


    model.train()
    scheduler.step()

    print(f"Summary Ep {epoch+1}:\n"
        f"Train Loss: {avg_total_loss:.4f} | Val Loss: {val_loss:.4f}\n"
        f"Part Conn Train Loss: {avg_loss_part_conn:.4f} | Part Conn Val Loss: {val_loss_part_conn:.4f}\n"
        f"Joint Type Train Loss: {avg_loss_joint_type:.4f} | Joint Type Val Loss: {val_loss_joint_type:.4f}\n"
        f"Revolute Axis Train Loss: {avg_revolute_axis_loss:.4f} | Revolute Axis Val Loss: {val_revolute_axis_loss:.4f}\n"
        f"Revolute Pivot Train Loss: {avg_revolute_pivot_loss:.4f} | Revolute Pivot Val Loss: {val_revolute_pivot_loss:.4f}\n"
        f"Prismatic Train Loss: {avg_prismatic_loss:.4f} | Prismatic Val Loss: {val_prismatic_loss:.4f}\n")
    
    if avg_total_loss < best_loss_train:
        best_loss_train = avg_total_loss
        check_dict_train = {
            "epoch": epoch + 1,
            "train_loss": float(best_loss_train),
            "val_loss": float(best_loss_val),
            "params": params,
            "model_state": model.state_dict(),
        }
        torch.save(check_dict_train, os.path.join(checkpoint_path, f"chkpt_best_model_train.pth"))
        print(f"\nNew best train model saved at epoch {epoch+1} with train loss {best_loss_train:.4f}")
    if val_loss < best_loss_val:
        best_loss_val = val_loss
        check_dict_val = {
            "epoch": epoch + 1,
            "train_loss": float(best_loss_train),
            "val_loss": float(best_loss_val),
            "params": params,
            "model_state": model.state_dict(),
        }
        torch.save(check_dict_val, os.path.join(checkpoint_path, f"chkpt_best_model_val.pth"))
        print(f"\nNew best validation model saved at epoch {epoch+1} with val loss {best_loss_val:.4f}")
    if (epoch + 1) % 5 == 0:
        check_dict_epoch = {
            "epoch": epoch + 1,
            "train_loss": float(best_loss_train),
            "val_loss": float(best_loss_val),
            "params": params,
            "model_state": model.state_dict(),
        }
        torch.save(check_dict_epoch, os.path.join(checkpoint_path, f"chkpt_{epoch + 1}.pth"))