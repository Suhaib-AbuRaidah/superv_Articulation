import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append('/home/suhaib/superv_Articulation')

from utilis.dataset2 import PartsGraphDataset2
from GNNPP.gnn_pointnet_network import parts_connection_mlp
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def mean_or_zero(x):
    if len(x) == 0:
        return 0.0
    return float(np.mean(x))

def binarize(logits, thr=0.5):
    return (torch.sigmoid(logits) > thr).float()

def cosine_similarity_vec(a, b):
    return torch.sum(a * b, dim=1)

def angular_err_deg(a, b):
    cos_sim = cosine_similarity_vec(a, b).clamp(-1+1e-7, 1-1e-7)
    return torch.rad2deg(torch.acos(cos_sim))

def _empty_metrics_dict():
    return {
        "conn_acc": 0.0,
        "conn_prec": 0.0,
        "conn_rec": 0.0,
        "conn_f1": 0.0,
        "joint_acc": 0.0,
        "joint_f1": 0.0,
        "revolute_cos": [],
        "revolute_ang": [],
        "prismatic_cos": [],
        "prismatic_ang": [],
    }

def _empty_metrics_with_conn(conn_acc, p, r, f1):
    m = _empty_metrics_dict()
    m["conn_acc"] = conn_acc
    m["conn_prec"] = p
    m["conn_rec"] = r
    m["conn_f1"] = f1
    return m

# ------------------------------------------------------------
# Evaluation step for a single sample
# ------------------------------------------------------------
def eval_step(model, data_dict, verbose=False, skip_invalid=True):
    """
    Returns dict as before. If an indexing problem occurs, either skip the sample
    (skip_invalid=True) and return zeros/empty lists, or raise an informative error.
    """
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

    adj = adj.squeeze()
    # print(f"\n\nAdjacency Matrix:\n{adj}\n\n")
    parts_connections_gt = parts_connections_gt.squeeze()
    # print(f"Parts Connections GT:\n{parts_connections_gt}\n\n")
    screw_axis_list_gt = screw_axis_list_gt.squeeze().view(-1,3)
    screw_axis_list_gt = torch.abs(screw_axis_list_gt)
    screw_axis_list_gt = F.normalize(screw_axis_list_gt, dim=1)
    screw_point_list_gt = screw_point_list_gt.squeeze().view(-1,3)

    joint_type_list_gt = joint_type_list_gt.squeeze()
    angles = angles.squeeze().view(-1,1)
    
    # ensure device consistent
    device = next(model.parameters()).device

    # move GT tensors to same device (they may already be)
    parts_connections_gt = parts_connections_gt.squeeze().to(device)
    joint_type_list_gt = joint_type_list_gt.squeeze().to(device)
    screw_axis_list_gt = screw_axis_list_gt.squeeze().to(device)

    # model forward (keep in eval mode; caller should have set it)
    edges_conne_pred, joint_type_pred, revolute_para_pred, prismatic_para_pred, z, (src, dst) = \
        model(parts_start_list, parts_end_list, adj)

    # ensure src/dst are LongTensor on correct device
    src = src.to(device).long()
    dst = dst.to(device).long()

    # Quick sanity checks
    if src.numel() != dst.numel():
        msg = f"src ({src.numel()}) != dst ({dst.numel()})"
        if verbose: print("INDEX ERROR:", msg, "file:", file_name)
        if skip_invalid:
            return _empty_metrics_dict()
        raise IndexError(msg)

    # shape expectations
    n_nodes = parts_connections_gt.shape[0]  # adjacency matrix is [N,N]
    max_idx = int(max(int(src.max().item()), int(dst.max().item()))) if src.numel() > 0 else -1
    min_idx = int(min(int(src.min().item()), int(dst.min().item()))) if src.numel() > 0 else 0

    # If indexing out of bounds, log and optionally skip
    if src.numel() == 0:
        if verbose: print("No edges predicted for file:", file_name)
        return _empty_metrics_dict()

    if min_idx < 0 or max_idx >= n_nodes:
        if verbose:
            print("Index out of bounds detected for file:", file_name)
            print(f"nodes in GT adjacency: {n_nodes}, src_min:{min_idx}, src_max:{max_idx}")
            print("src sample:", src)
            print("dst sample:", dst)
        if skip_invalid:
            return _empty_metrics_dict()
        raise IndexError(f"Index out of bounds: nodes={n_nodes}, src_max={max_idx}, src_min={min_idx}")

    # Now safe to index
    conn_gt = parts_connections_gt[src, dst].float().unsqueeze(1)  # [E_model,1]

    # connection metrics
    conn_pred_lbl = binarize(edges_conne_pred)
    tp = ((conn_pred_lbl == 1) & (conn_gt == 1)).sum().item()
    fp = ((conn_pred_lbl == 1) & (conn_gt == 0)).sum().item()
    fn = ((conn_pred_lbl == 0) & (conn_gt == 1)).sum().item()
    tn = ((conn_pred_lbl == 0) & (conn_gt == 0)).sum().item()

    conn_acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    conn_prec = tp / max(tp + fp, 1)
    conn_rec = tp / max(tp + fn, 1)
    conn_f1 = 2 * conn_prec * conn_rec / max(conn_prec + conn_rec, 1e-12)

    # joint mask (edges that actually exist in GT)
    joint_mask = conn_gt.squeeze(1) > 0

    # Align joint_type_list_gt by (src,dst) then mask
    jt_gt_edges = joint_type_list_gt[src, dst].view(-1, 1).to(device)  # [E_model,1]
    jt_pred_edges = joint_type_pred.view(-1, 1)  # logits aligned with model edges

    joint_acc = 0.0
    joint_f1 = 0.0
    if joint_mask.sum() > 0:
        jt_pred_valid = binarize(jt_pred_edges[joint_mask])
        jt_gt_valid = jt_gt_edges[joint_mask].float()

        tp2 = ((jt_pred_valid == 1) & (jt_gt_valid == 1)).sum().item()
        fp2 = ((jt_pred_valid == 1) & (jt_gt_valid == 0)).sum().item()
        fn2 = ((jt_pred_valid == 0) & (jt_gt_valid == 1)).sum().item()
        tn2 = ((jt_pred_valid == 0) & (jt_gt_valid == 0)).sum().item()

        joint_acc = (tp2 + tn2) / max(tp2 + tn2 + fp2 + fn2, 1)
        prec2 = tp2 / max(tp2 + fp2, 1)
        rec2 = tp2 / max(tp2 + fn2, 1)
        joint_f1 = 2 * prec2 * rec2 / max(prec2 + rec2, 1e-12)

    # Axis metrics
    # prepare GT axes aligned by edges
    # screw_axis_list_gt is [N_nodes, 3] or [E_gt,3]? If you used adjacency indexing, it's [N_nodes,3] and jt_gt_edges indexing will produce per-edge axis.
    try:
        axis_gt_edges = screw_axis_list_gt[src, dst].view(-1, 3).to(device)
    except Exception:
        # fallback: if screw_axis_list_gt is [E_gt,3] already, try indexing directly by edge index if sizes match
        if screw_axis_list_gt.shape[0] == jt_gt_edges.shape[0]:
            axis_gt_edges = screw_axis_list_gt.view(-1, 3).to(device)
        else:
            if verbose: print("Unable to align screw_axis_list_gt by (src,dst) for file:", file_name)
            return _empty_metrics_with_conn(conn_acc, conn_prec, conn_rec, conn_f1)

    axis_gt_edges = axis_gt_edges.abs()
    axis_gt_edges = F.normalize(axis_gt_edges, dim=1)

    # predicted axes for valid edges
    rev_pred_axes = F.normalize(revolute_para_pred.view(-1, 3)[joint_mask], dim=1) if revolute_para_pred.numel() > 0 else torch.empty((0,3), device=device)
    pri_pred_axes = F.normalize(prismatic_para_pred.view(-1, 3)[joint_mask], dim=1) if prismatic_para_pred.numel() > 0 else torch.empty((0,3), device=device)

    # determine joint type per-edge (GT) to split between revolute/prismatic
    jt_gt_edges_flat = jt_gt_edges.view(-1)
    revolute_mask = joint_mask & (jt_gt_edges_flat == 0)
    prismatic_mask = joint_mask & (jt_gt_edges_flat == 1)

    revolute_cos, revolute_ang, prismatic_cos, prismatic_ang = [], [], [], []

    # revolute: choose predicted axes from rev_pred_axes aligned with revolute_mask
    if revolute_mask.sum() > 0 and rev_pred_axes.numel() > 0:
        # Need to align indices: rev_pred_axes is revolute predictions only for joint_mask True
        # create index positions of joint_mask True
        valid_idx = torch.nonzero(joint_mask, as_tuple=False).view(-1)
        rev_idx_in_valid = torch.nonzero(revolute_mask, as_tuple=False).view(-1)
        # map to positions inside rev_pred_axes
        rev_positions = (revolute_mask.nonzero(as_tuple=False).view(-1) - valid_idx[0] + 0) if False else None
        # simpler: use boolean selection with same size
        gt_rev = axis_gt_edges[revolute_mask[joint_mask]]
        pred_rev = rev_pred_axes[revolute_mask[joint_mask]]
        if pred_rev.shape[0] == gt_rev.shape[0]:
            c = cosine_similarity_vec(pred_rev, gt_rev).cpu().numpy()
            a = angular_err_deg(pred_rev, gt_rev).cpu().numpy()
            revolute_cos.extend(c.tolist())
            revolute_ang.extend(a.tolist())
        else:
            # best effort: iterate pairs (safe)
            min_len = min(pred_rev.shape[0], gt_rev.shape[0])
            if min_len > 0:
                c = cosine_similarity_vec(pred_rev[:min_len], gt_rev[:min_len]).cpu().numpy()
                a = angular_err_deg(pred_rev[:min_len], gt_rev[:min_len]).cpu().numpy()
                revolute_cos.extend(c.tolist())
                revolute_ang.extend(a.tolist())

    # prismatic
    if prismatic_mask.sum() > 0 and pri_pred_axes.numel() > 0:
        gt_pri = axis_gt_edges[prismatic_mask[joint_mask]]
        pred_pri = pri_pred_axes[prismatic_mask[joint_mask]]
        if pred_pri.shape[0] == gt_pri.shape[0]:
            c = cosine_similarity_vec(pred_pri, gt_pri).cpu().numpy()
            a = angular_err_deg(pred_pri, gt_pri).cpu().numpy()
            prismatic_cos.extend(c.tolist())
            prismatic_ang.extend(a.tolist())
        else:
            min_len = min(pred_pri.shape[0], gt_pri.shape[0])
            if min_len > 0:
                c = cosine_similarity_vec(pred_pri[:min_len], gt_pri[:min_len]).cpu().numpy()
                a = angular_err_deg(pred_pri[:min_len], gt_pri[:min_len]).cpu().numpy()
                prismatic_cos.extend(c.tolist())
                prismatic_ang.extend(a.tolist())

    return {
        "conn_acc": conn_acc,
        "conn_prec": conn_prec,
        "conn_rec": conn_rec,
        "conn_f1": conn_f1,
        "joint_acc": joint_acc,
        "joint_f1": joint_f1,
        "revolute_cos": revolute_cos,
        "revolute_ang": revolute_ang,
        "prismatic_cos": prismatic_cos,
        "prismatic_ang": prismatic_ang,
    }


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def evaluate(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PartsGraphDataset2(
        "../Ditto/Articulated_object_simulation-main/data/Shape2Motion_gcn/robotic_arm_testing/scenes/*.npz",
        device
    )
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
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
    }

    model = parts_connection_mlp(**params).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Accumulators
    all_conn_acc = []
    all_conn_prec = []
    all_conn_rec = []
    all_conn_f1 = []
    all_joint_acc = []
    all_joint_f1 = []
    all_revolute_cos = []
    all_revolute_ang = []
    all_prismatic_cos = []
    all_prismatic_ang = []

    with torch.no_grad():
        for data in val_loader:
            metrics = eval_step(model, data)

            all_conn_acc.append(metrics["conn_acc"])
            all_conn_prec.append(metrics["conn_prec"])
            all_conn_rec.append(metrics["conn_rec"])
            all_conn_f1.append(metrics["conn_f1"])
            all_joint_acc.append(metrics["joint_acc"])
            all_joint_f1.append(metrics["joint_f1"])
            all_revolute_cos.extend(metrics["revolute_cos"])
            all_revolute_ang.extend(metrics["revolute_ang"])
            all_prismatic_cos.extend(metrics["prismatic_cos"])
            all_prismatic_ang.extend(metrics["prismatic_ang"])

    final_metrics = {
        "conn_acc": mean_or_zero(all_conn_acc),
        "conn_prec": mean_or_zero(all_conn_prec),
        "conn_rec": mean_or_zero(all_conn_rec),
        "conn_f1": mean_or_zero(all_conn_f1),
        "joint_type_acc": mean_or_zero(all_joint_acc),
        "joint_type_f1": mean_or_zero(all_joint_f1),
        "revolute_cosine": mean_or_zero(all_revolute_cos),
        "revolute_angle_err_deg": mean_or_zero(all_revolute_ang),
        "prismatic_cosine": mean_or_zero(all_prismatic_cos),
        "prismatic_angle_err_deg": mean_or_zero(all_prismatic_ang),
    }

    return final_metrics


if __name__ == "__main__":
    chk = "./pre_trained_models_gcnpp/2025-11-26 22:17:23_mix/chkpt_best_model_val.pth"
    metrics = evaluate(chk)
    print(metrics)
