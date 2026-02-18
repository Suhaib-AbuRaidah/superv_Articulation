import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm

sys.path.append(os.path.expanduser('~/superv_Articulation'))

from utilis.dataset2 import PartsGraphDataset3
from GNNPP.gnn_pointnet2_network_v3 import parts_connection_mlp
from utilis.contact_points import find_contact_points


# --------------------------- helpers ---------------------------

def _cosine_abs(a, b, eps=1e-8):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    c = (a * b).sum(dim=1)
    return c.abs().clamp(eps, 1.0)

def _angular_err_deg_from_cos(cosv):
    cosv = cosv.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.rad2deg(torch.acos(cosv))

def _point_to_axis_distance(point, axis_point, axis_dir):
    """
    point:      (N,3) predicted pivot
    axis_point: (N,3) GT pivot
    axis_dir:   (N,3) GT axis (normalized)
    """
    v = point - axis_point
    proj = torch.sum(v * axis_dir, dim=1, keepdim=True) * axis_dir
    perp = v - proj
    return torch.norm(perp, dim=1)

def _as_edge_logits(x):
    """
    Make edge logits shape [E] from common shapes like [E,1], [E,P,1] (if not already reduced).
    Assumes you already did mean(dim=1) for per-point outputs when needed.
    """
    if x.dim() == 2 and x.shape[-1] == 1:
        return x.view(-1)
    if x.dim() == 1:
        return x
    raise ValueError(f"Unexpected edge logit shape: {tuple(x.shape)}")

def binarize_from_logits(edge_logits, thr=0.5):
    # edge_logits: [E]
    return (torch.sigmoid(edge_logits) > thr).to(edge_logits.dtype)  # [E]


# --------------------------- evaluation ---------------------------

@torch.no_grad()
def evaluation(model, data_dict, snap_pivot_to_contact=True):
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
        src_unused, dst_unused,
        edge_dir_gt, edge_dst_gt,
    ) = data_dict

    device = next(model.parameters()).device

    # ------------------ move inputs to device ------------------
    adj = adj.squeeze().to(device)
    print(f"adj:\n{adj}\n")
    # parts_start_list = [x.to(device, non_blocking=True) for x in parts_start_list]
    # parts_end_list   = [x.to(device, non_blocking=True) for x in parts_end_list]

    # ------------------ forward ------------------
    edges_conne_pred, joint_type_pred, revolute_para_pred, prismatic_para_pred, (src, dst) = \
        model(parts_start_list, parts_end_list, adj)


    # Aggregate per-point logits -> per-edge logits (same as your inference)
    edges_conne_pred = edges_conne_pred.mean(dim=1)  # [E_total,1] or [E_total]
    joint_type_pred  = joint_type_pred.mean(dim=1)   # [E_total,1] or [E_total]
    print(f"joint_type_pred:\n {joint_type_pred}\n")
    edge_logits = _as_edge_logits(edges_conne_pred)  # [E_total]
    jt_logits   = _as_edge_logits(joint_type_pred)   # [E_total]

    src = src.long().to(device)
    dst = dst.long().to(device)

    if src.numel() == 0:
        return None

    # ------------------ GT to device ------------------
    parts_connections_gt = parts_connections_gt.squeeze().to(device)          # [N,N]
    joint_type_list_gt   = joint_type_list_gt.view(-1).to(device)            # [E_gt]
    print(f"joint_type_list_gt:\n{joint_type_list_gt}\n")
    screw_axis_list_gt   = screw_axis_list_gt.squeeze().view(-1, 3).to(device)
    screw_axis_list_gt   = F.normalize(screw_axis_list_gt, dim=1)
    screw_point_list_gt  = screw_point_list_gt.squeeze().view(-1, 3).to(device)

    # ------------------ connection metrics over candidate edges ------------------
    conn_pred_lbl = binarize_from_logits(edge_logits, thr=0.5)   # [E_total] in {0,1}
    joint_mask_pred = (conn_pred_lbl > 0).view(-1)               # [E_total] bool
    print(f"joint mask pred:\n {joint_mask_pred}\n")
    conn_gt = parts_connections_gt[src, dst].float().view(-1)    # [E_total]
    print(f"conn_gt:\n{conn_gt}\n")
    conn_pred = conn_pred_lbl.float().view(-1)                   # [E_total]
    print(f"conn_pred:\n{conn_pred}\n")

    tp = ((conn_pred == 1) & (conn_gt == 1)).sum().item()
    fp = ((conn_pred == 1) & (conn_gt == 0)).sum().item()
    fn = ((conn_pred == 0) & (conn_gt == 1)).sum().item()
    tn = ((conn_pred == 0) & (conn_gt == 0)).sum().item()

    conn_acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    conn_prec = tp / max(tp + fp, 1)
    conn_rec  = tp / max(tp + fn, 1)
    conn_f1   = 2 * conn_prec * conn_rec / max(conn_prec + conn_rec, 1e-12)

    # If no joints predicted, return only connection metrics
    if int(joint_mask_pred.sum().item()) == 0:
        return {
            "file_name": file_name,
            "number of parts": int(parts_connections_gt.shape[0]),
            "conn_acc": conn_acc, "conn_prec": conn_prec, "conn_rec": conn_rec, "conn_f1": conn_f1,
            "joint_acc": 0.0, "joint_f1": 0.0,
            "revolute_cos": [], "revolute_ang": [], "revolute_pivot_dist": [],
            "prismatic_cos": [], "prismatic_ang": [],
        }

    # ------------------ EXTRACT predictions explicitly (like your inference) ------------------

    # revolute axis (weighted avg)
    rev_axis_ep3 = revolute_para_pred[joint_mask_pred, :, :3]                  # [E,P,3]
    rev_w_ep1    = torch.sigmoid(revolute_para_pred[joint_mask_pred, :, 3:4])  # [E,P,1]
    rev_axis_e3  = (rev_axis_ep3 * rev_w_ep1).sum(dim=1) / (rev_w_ep1.sum(dim=1) + 1e-6)
    rev_axis_e3  = F.normalize(rev_axis_e3, dim=1)
    print(f"rev_axis_e3:\n{rev_axis_e3}\n")
    # pivot prediction (direction + signed distance)
    edge_dir_pred = revolute_para_pred[:, :, 4:7][joint_mask_pred]             # [E,2P,3] in your usage
    edge_dir_pred = F.normalize(edge_dir_pred, dim=-1)
    edge_dst_pred = revolute_para_pred[:, :, 7:8][joint_mask_pred]             # [E,2P,1]

    # per-edge points from start parts (union of the two parts)
    pcs1 = torch.cat([pc.transpose(1, 2) for pc in parts_start_list], dim=0)    # [N,3,P]
    edge_points = torch.cat([pcs1[src], pcs1[dst]], dim=2).transpose(1, 2)      # [E_total,2P,3]
    edge_points_joint = edge_points[joint_mask_pred]                            # [E,2P,3]

    edge_pivot_per_point = edge_points_joint + edge_dir_pred * edge_dst_pred    # [E,2P,3]
    rev_edge_pivots = edge_pivot_per_point.mean(dim=1)                          # [E,3]

    if snap_pivot_to_contact:
        # snap pivot to nearest contact point (slow; CPU loop)
        for i in range(edge_points_joint.shape[0]):
            part_1 = edge_points_joint[i, :edge_points_joint.shape[1] // 2].detach().cpu().numpy()
            part_2 = edge_points_joint[i, edge_points_joint.shape[1] // 2:].detach().cpu().numpy()
            closest_contact = find_contact_points(
                part_1, part_2, predicted_pivot=rev_edge_pivots[i].detach().cpu().numpy()
            )
            rev_edge_pivots[i] = torch.from_numpy(closest_contact).to(device)

    # prismatic axis (weighted avg)
    pri_axis_ep3 = prismatic_para_pred[joint_mask_pred, :, :3]
    pri_w_ep1    = torch.sigmoid(prismatic_para_pred[joint_mask_pred, :, 3:4])
    pri_axis_e3  = (pri_axis_ep3 * pri_w_ep1).sum(dim=1) / (pri_w_ep1.sum(dim=1) + 1e-6)
    pri_axis_e3  = F.normalize(pri_axis_e3, dim=1)
    print(f"pri_axis_e3:\n{pri_axis_e3}\n")
    # predicted joint types for predicted joints (use jt_logits restricted to predicted joints)
    jt = (torch.sigmoid(jt_logits[joint_mask_pred]) > 0.5).long().view(-1)      # [E] 1=pri,0=rev
    print(f"jt pred:\n{jt}\n")
    # axes_pred aligned to predicted joints
    E = int(joint_mask_pred.sum().item())
    axes_pred = torch.zeros((E, 3), device=device, dtype=rev_axis_e3.dtype)
    rev_mask_pred = (jt == 0)
    pri_mask_pred = (jt == 1)
    axes_pred[rev_mask_pred] = rev_axis_e3[rev_mask_pred]
    axes_pred[pri_mask_pred] = pri_axis_e3[pri_mask_pred]

    # ------------------ ALIGN predicted joint edges to GT edges (same logic you had) ------------------
    gt_src, gt_dst = torch.where(parts_connections_gt > 0)
    edge2idx = {(int(s.item()), int(d.item())): i for i, (s, d) in enumerate(zip(gt_src, gt_dst))}

    src_pred_e = src[joint_mask_pred]
    dst_pred_e = dst[joint_mask_pred]

    gt_idx_list = []
    valid_edge_mask_list = []
    for s, d in zip(src_pred_e, dst_pred_e):
        key = (int(s.item()), int(d.item()))
        if key in edge2idx:
            gt_idx_list.append(edge2idx[key])
            valid_edge_mask_list.append(True)
        else:
            valid_edge_mask_list.append(False)

    if len(gt_idx_list) == 0:
        return {
            "file_name": file_name,
            "number of parts": int(parts_connections_gt.shape[0]),
            "conn_acc": conn_acc, "conn_prec": conn_prec, "conn_rec": conn_rec, "conn_f1": conn_f1,
            "joint_acc": 0.0, "joint_f1": 0.0,
            "revolute_cos": [], "revolute_ang": [], "revolute_pivot_dist": [],
            "prismatic_cos": [], "prismatic_ang": [],
        }

    valid_edge_mask = torch.tensor(valid_edge_mask_list, device=device, dtype=torch.bool)
    gt_idx = torch.tensor(gt_idx_list, device=device, dtype=torch.long)

    # align preds to "valid predicted joint edges"
    jt_pred_valid     = jt[valid_edge_mask]                 # [E_valid]
    axes_pred_valid   = axes_pred[valid_edge_mask]          # [E_valid,3]
    pivots_pred_valid = rev_edge_pivots[valid_edge_mask]    # [E_valid,3]

    jt_gt_edges    = joint_type_list_gt[gt_idx].long()      # [E_valid] (0=rev,1=pri)
    axis_gt_edges  = screw_axis_list_gt[gt_idx]             # [E_valid,3]
    pivot_gt_edges = screw_point_list_gt[gt_idx]            # [E_valid,3]
    print(f"jt_gt_edges:\n{jt_gt_edges}\n")
    # ------------------ joint type metrics ------------------
    tp2 = ((jt_pred_valid == 1) & (jt_gt_edges == 1)).sum().item()
    fp2 = ((jt_pred_valid == 1) & (jt_gt_edges == 0)).sum().item()
    fn2 = ((jt_pred_valid == 0) & (jt_gt_edges == 1)).sum().item()
    tn2 = ((jt_pred_valid == 0) & (jt_gt_edges == 0)).sum().item()

    joint_acc = (tp2 + tn2) / max(tp2 + tn2 + fp2 + fn2, 1)
    prec2 = tp2 / max(tp2 + fp2, 1)
    rec2  = tp2 / max(tp2 + fn2, 1)
    joint_f1 = 2 * prec2 * rec2 / max(prec2 + rec2, 1e-12)

    # ------------------ axis/pivot metrics split by GT type ------------------
    rev_gt_mask = (jt_gt_edges == 0)
    pri_gt_mask = (jt_gt_edges == 1)

    revolute_cos, revolute_ang, revolute_pivot_dist = [], [], []
    prismatic_cos, prismatic_ang = [], []

    if rev_gt_mask.any():
        pred_rev_axis = axes_pred_valid[rev_gt_mask]
        gt_rev_axis   = axis_gt_edges[rev_gt_mask]

        cosv = _cosine_abs(pred_rev_axis, gt_rev_axis)
        ang  = _angular_err_deg_from_cos(cosv)

        revolute_cos = cosv.detach().cpu().tolist()
        revolute_ang = ang.detach().cpu().tolist()

        pred_rev_pivot = pivots_pred_valid[rev_gt_mask]
        gt_rev_pivot   = pivot_gt_edges[rev_gt_mask]
        gt_rev_axis_n  = F.normalize(gt_rev_axis, dim=1)

        dist = _point_to_axis_distance(pred_rev_pivot, gt_rev_pivot, gt_rev_axis_n)
        revolute_pivot_dist = dist.detach().cpu().tolist()

    if pri_gt_mask.any():
        pred_pri_axis = axes_pred_valid[pri_gt_mask]
        gt_pri_axis   = axis_gt_edges[pri_gt_mask]

        cosv = _cosine_abs(pred_pri_axis, gt_pri_axis)
        ang  = _angular_err_deg_from_cos(cosv)

        prismatic_cos = cosv.detach().cpu().tolist()
        prismatic_ang = ang.detach().cpu().tolist()

    return {
        "file_name": file_name,
        "number of parts": int(parts_connections_gt.shape[0]),

        "conn_acc": conn_acc,
        "conn_prec": conn_prec,
        "conn_rec": conn_rec,
        "conn_f1": conn_f1,

        "joint_acc": joint_acc,
        "joint_f1": joint_f1,

        "revolute_cos": revolute_cos,
        "revolute_ang": revolute_ang,
        "revolute_pivot_dist": revolute_pivot_dist,

        "prismatic_cos": prismatic_cos,
        "prismatic_ang": prismatic_ang,
    }


# --------------------------- main ---------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PartsGraphDataset3(
        os.path.expanduser("~/superv_Articulation/data/Shape2Motion_gcn/cabinet_simp/val/scenes/*.npz"),
        device
    )
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    pbar_val = tqdm.tqdm(val_loader, leave=True, dynamic_ncols=True)
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

    checkpoint_path = os.path.expanduser(
        "~/superv_Articulation/pre_trained_models_gcnpp/2026-02-18 17:10:53_C/chkpt_best_model_val.pth"
    )

    model = parts_connection_mlp(**params).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    def _safe_mean(x):
        x = [v for v in x if v is not None]
        return float(np.mean(x)) if len(x) else float("nan")

    def _flatten_listlist(x):
        out = []
        for a in x:
            if a is None:
                continue
            if isinstance(a, (list, tuple)):
                out.extend(a)
            else:
                out.append(a)
        return out

    # Accumulators
    counts = 0
    sum_conn_acc = 0.0
    sum_conn_prec = 0.0
    sum_conn_rec = 0.0
    sum_conn_f1 = 0.0

    sum_joint_acc = 0.0
    sum_joint_f1 = 0.0
    # For per-edge continuous metrics, store all values across dataset
    all_rev_cos = []
    all_rev_ang = []
    all_rev_pivot_dist = []
    all_pri_cos = []
    all_pri_ang = []

    num_skipped = 0
    target_idx = 0
    with torch.no_grad():
        for index, data in enumerate(pbar_val):
            # if index != target_idx:
            #     continue
            metrics = evaluation(model, data)
            if metrics is None:
                num_skipped += 1
                continue

            counts += 1
            sum_conn_acc  += float(metrics["conn_acc"])
            sum_conn_prec += float(metrics["conn_prec"])
            sum_conn_rec  += float(metrics["conn_rec"])
            sum_conn_f1   += float(metrics["conn_f1"])

            sum_joint_acc += float(metrics["joint_acc"])
            sum_joint_f1  += float(metrics["joint_f1"])

            # extend edge-level lists (may be empty)
            all_rev_cos.extend(metrics.get("revolute_cos", []))
            all_rev_ang.extend(metrics.get("revolute_ang", []))
            all_rev_pivot_dist.extend(metrics.get("revolute_pivot_dist", []))
            all_pri_cos.extend(metrics.get("prismatic_cos", []))
            all_pri_ang.extend(metrics.get("prismatic_ang", []))

    # Macro averages (mean over samples)
    macro_conn_acc  = sum_conn_acc  / max(counts, 1)
    macro_conn_prec = sum_conn_prec / max(counts, 1)
    macro_conn_rec  = sum_conn_rec  / max(counts, 1)
    macro_conn_f1   = sum_conn_f1   / max(counts, 1)

    macro_joint_acc = sum_joint_acc / max(counts, 1)
    macro_joint_f1  = sum_joint_f1  / max(counts, 1)

    # Micro-ish summaries for continuous metrics (mean over all evaluated edges)
    rev_cos_mean = _safe_mean(all_rev_cos)
    rev_ang_mean = _safe_mean(all_rev_ang)
    rev_pivot_mean = _safe_mean(all_rev_pivot_dist)

    pri_cos_mean = _safe_mean(all_pri_cos)
    pri_ang_mean = _safe_mean(all_pri_ang)

    print("\n================ FINAL METRICS (DATASET) ================\n")
    print(f"Evaluated samples: {counts} / {len(val_loader)}   (skipped: {num_skipped})\n")

    print("Connection (macro over samples):")
    print(f"  conn_acc : {macro_conn_acc:.4f}")
    print(f"  conn_prec: {macro_conn_prec:.4f}")
    print(f"  conn_rec : {macro_conn_rec:.4f}")
    print(f"  conn_f1  : {macro_conn_f1:.4f}\n")

    print("Joint type (macro over samples):")
    print(f"  joint_acc: {macro_joint_acc:.4f}")
    print(f"  joint_f1 : {macro_joint_f1:.4f}\n")

    print("Axis / Pivot (mean over all matched edges):")
    print(f"  revolute_cos_mean       : {rev_cos_mean:.4f}")
    print(f"  revolute_ang_mean (deg) : {rev_ang_mean:.4f}")
    print(f"  revolute_pivot_dist_mean: {rev_pivot_mean:.4f}")
    print(f"  prismatic_cos_mean      : {pri_cos_mean:.4f}")
    print(f"  prismatic_ang_mean (deg): {pri_ang_mean:.4f}")
