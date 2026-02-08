import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append(os.path.expanduser('~/superv_Articulation'))

from utilis.dataset2 import PartsGraphDataset2
from GNNPP.gnn_pointnet2_network_v2 import parts_connection_mlp
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
    return torch.abs(torch.sum(a * b, dim=1))

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


def canonical_direction(z):
    threshold = 0.15
    mask = (
        (z[:, 2] >= threshold) |
        ((z[:, 2] <= threshold) & (z[:, 1] >= threshold)) |
        ((z[:, 2] <= threshold) & (z[:, 1] <= threshold) & (z[:, 0] >= threshold))
    )

    return torch.where(mask.unsqueeze(1), z, -z)
# ------------------------------------------------------------
# Evaluation step for a single sample
# ------------------------------------------------------------
def eval_step(model, data_dict, verbose=False, skip_invalid=True):
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

    device = next(model.parameters()).device

    adj = adj.squeeze().to(device)
    parts_connections_gt = parts_connections_gt.squeeze().to(device)
    joint_type_list_gt = joint_type_list_gt.view(-1).to(device)          # [E_gt]
    screw_axis_list_gt = screw_axis_list_gt.squeeze().view(-1, 3).to(device)
    screw_axis_list_gt = F.normalize(screw_axis_list_gt, dim=1)
    screw_point_list_gt = screw_point_list_gt.squeeze().view(-1, 3).to(device)
    # screw_axis_list_gt = canonical_direction(screw_axis_list_gt)

    # -------- model forward --------
    edges_conne_pred, joint_type_pred, revolute_para_pred, prismatic_para_pred, (src, dst) = \
        model(parts_start_list, parts_end_list, adj)
    
    edges_conne_pred = edges_conne_pred.mean(dim=1)
    joint_type_pred = joint_type_pred.mean(dim=1)
    
    src = src.long().to(device)
    dst = dst.long().to(device)

    if src.numel() == 0:
        return _empty_metrics_dict()

    # -------- model-edge GT connectivity --------
    conn_gt = parts_connections_gt[src, dst].float().unsqueeze(1)


    conn_pred_lbl = binarize(edges_conne_pred)
    print(f"conn_gt: \n{conn_gt}")
    print(f"conn_pred_lbl: \n{conn_pred_lbl}")
    tp = ((conn_pred_lbl == 1) & (conn_gt == 1)).sum().item()
    fp = ((conn_pred_lbl == 1) & (conn_gt == 0)).sum().item()
    fn = ((conn_pred_lbl == 0) & (conn_gt == 1)).sum().item()
    tn = ((conn_pred_lbl == 0) & (conn_gt == 0)).sum().item()

    conn_acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    conn_prec = tp / max(tp + fp, 1)
    conn_rec  = tp / max(tp + fn, 1)
    conn_f1   = 2 * conn_prec * conn_rec / max(conn_prec + conn_rec, 1e-12)
    # print(f"File: {file_name} | Conn F1: {conn_f1:.4f}")
    # -------- joint mask --------
    joint_mask = conn_gt.squeeze(1) > 0
    if joint_mask.sum() == 0:
        return {
            "conn_acc": conn_acc,
            "conn_prec": conn_prec,
            "conn_rec": conn_rec,
            "conn_f1": conn_f1,
            "joint_acc": 0.0,
            "joint_f1": 0.0,
            "revolute_cos": [],
            "revolute_ang": [],
            "prismatic_cos": [],
            "prismatic_ang": [],
        }

    # -------- build GT edge map --------
    gt_src, gt_dst = torch.where(parts_connections_gt > 0)

    edge2idx = {
        (int(s.item()), int(d.item())): i
        for i, (s, d) in enumerate(zip(gt_src, gt_dst))
    }

    # # align GT joint types to model edges
    # jt_gt_edges = torch.tensor(
    #     [joint_type_list_gt[edge2idx[(int(s), int(d))]]
    #      for s, d in zip(src[joint_mask], dst[joint_mask])],
    #     device=device
    # ).view(-1, 1)
    jt_gt_edges = joint_type_list_gt.view(-1, 1)
    jt_pred_edges = joint_type_pred[joint_mask].view(-1, 1)

    jt_pred_lbl = binarize(jt_pred_edges)
    print(f"jt_gt_edges: \n{jt_gt_edges}")
    print(f"jt_pred_lbl: \n{jt_pred_lbl}")
    tp2 = ((jt_pred_lbl == 1) & (jt_gt_edges == 1)).sum().item()
    fp2 = ((jt_pred_lbl == 1) & (jt_gt_edges == 0)).sum().item()
    fn2 = ((jt_pred_lbl == 0) & (jt_gt_edges == 1)).sum().item()
    tn2 = ((jt_pred_lbl == 0) & (jt_gt_edges == 0)).sum().item()

    joint_acc = (tp2 + tn2) / max(tp2 + tn2 + fp2 + fn2, 1)
    prec2 = tp2 / max(tp2 + fp2, 1)
    rec2  = tp2 / max(tp2 + fn2, 1)
    joint_f1 = 2 * prec2 * rec2 / max(prec2 + rec2, 1e-12)

    # -------- axis evaluation --------
    # axis_gt = torch.stack([
    #     screw_axis_list_gt[edge2idx[(int(s), int(d))]]
    #     for s, d in zip(src[joint_mask], dst[joint_mask])
    # ])
    axis_gt = screw_axis_list_gt
    pivot_gt = screw_point_list_gt
    rev_mask = jt_gt_edges.view(-1) == 0
    pri_mask = jt_gt_edges.view(-1) == 1

    revolute_cos, revolute_ang, pivot_dist = [], [], []
    prismatic_cos, prismatic_ang = [], []

    if rev_mask.sum() > 0:
            # Compute per-edge parameter losses (L2)
        revolute_axis_pred = revolute_para_pred[:,:,:3][joint_mask].squeeze()
        rev_weights = torch.sigmoid(revolute_para_pred[:,:,3:4][joint_mask])
        revolute_axis_pred = (revolute_axis_pred * rev_weights).sum(dim=1) / (rev_weights.sum(dim=1) + 1e-6)
        pred_rev = F.normalize(revolute_axis_pred, dim=1)[rev_mask]
        gt_rev   = axis_gt[rev_mask]
        print(f"pred_rev: \n{pred_rev}\ngt_rev: \n{gt_rev}")
        revolute_cos = cosine_similarity_vec(pred_rev, gt_rev).cpu().tolist()
        revolute_ang = angular_err_deg(pred_rev, gt_rev).cpu().tolist()

        pivot_gt = pivot_gt[rev_mask]
        revolute_pivot_pred = revolute_para_pred[:,:,4:7][joint_mask].squeeze()
        rev_piv_weights = torch.sigmoid(revolute_para_pred[:,:,7:8][joint_mask])
        revolute_pivot_pred = (revolute_pivot_pred * rev_piv_weights).sum(dim=1) / (rev_piv_weights.sum(dim=1) + 1e-6)
        # Could also evaluate pivot point error here if desired
        pivot_dist = point_to_axis_distance(
        revolute_pivot_pred[rev_mask],
        pivot_gt,
        gt_rev).cpu().tolist()


    if pri_mask.sum() > 0:
        prismatic_axis_pred = prismatic_para_pred[:,:,:3][joint_mask].squeeze()
        pri_weights = torch.sigmoid(prismatic_para_pred[:,:,3:4][joint_mask])
        prismatic_axis_pred = (prismatic_axis_pred * pri_weights).sum(dim=1) / (pri_weights.sum(dim=1) + 1e-6)
        pred_pri = F.normalize(prismatic_axis_pred, dim=1)[pri_mask]
        gt_pri   = axis_gt[pri_mask]
        print(f"pred_pri: \n{pred_pri}\ngt_pri: \n{gt_pri}")
        prismatic_cos = cosine_similarity_vec(pred_pri, gt_pri).cpu().tolist()
        prismatic_ang = angular_err_deg(pred_pri, gt_pri).cpu().tolist()

    return {
        "conn_acc": conn_acc,
        "conn_prec": conn_prec,
        "conn_rec": conn_rec,
        "conn_f1": conn_f1,
        "joint_acc": joint_acc,
        "joint_f1": joint_f1,
        "revolute_cos": revolute_cos,
        "revolute_ang": revolute_ang,
        "revolute_pivot_dist": pivot_dist,
        "prismatic_cos": prismatic_cos,
        "prismatic_ang": prismatic_ang,
    }



# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def evaluate(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PartsGraphDataset2(
        os.path.expanduser("~/superv_Articulation/data/Shape2Motion_gcn/cabinet_simp/val/scenes/*.npz"),
        device
    )
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    params = {
        "pointnet_dim": 1024,
        "nlayers": 4,
        "nhidden": 512,
        "out_dim": 512,
        "dropout": 0.3,
        "lamda": 0.5,
        "alpha": 0.1,
        "variant": True,
        "nhidden_mlp": 256,
        "n_class": 1,
        "latent_dim": 1,
        "decoder_out_dim": 128,
        "motion_decoder_out_dim": 512,
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
    all_revolute_pivot_dist = []
    all_prismatic_cos = []
    all_prismatic_ang = []
    
    with torch.no_grad():
        for index, data in enumerate(val_loader):
            print(f"Evaluating sample {index}/{len(val_loader)}")
            metrics = eval_step(model, data)
            print(f"Metrics: {metrics}")
            all_conn_acc.append(metrics["conn_acc"])
            all_conn_prec.append(metrics["conn_prec"])
            all_conn_rec.append(metrics["conn_rec"])
            all_conn_f1.append(metrics["conn_f1"])
            all_joint_acc.append(metrics["joint_acc"])
            all_joint_f1.append(metrics["joint_f1"])
            all_revolute_cos.extend(metrics["revolute_cos"])
            all_revolute_ang.extend(metrics["revolute_ang"])
            all_revolute_pivot_dist.extend(metrics["revolute_pivot_dist"])
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
        "revolute_pivot_dist": mean_or_zero(all_revolute_pivot_dist),
        "prismatic_cosine": mean_or_zero(all_prismatic_cos),
        "prismatic_angle_err_deg": mean_or_zero(all_prismatic_ang),
    }

    return final_metrics


if __name__ == "__main__":
    chk = os.path.expanduser("~/superv_Articulation/pre_trained_models_gcnpp/2026-02-04 20:21:38_C/chkpt_best_model_train.pth")
    metrics = evaluate(chk)
    print(f"\n\n{metrics}")
