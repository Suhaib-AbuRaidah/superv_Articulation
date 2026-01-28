from xml.parsers.expat import model
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append(os.path.expanduser('~/superv_Articulation'))
import open3d as o3d
from utilis.dataset2 import PartsGraphDataset2
from GNNPP.gnn_pointnet2_network import parts_connection_mlp
from torch.utils.data import DataLoader
from utilis.Inference_graph import visualize_articulated_graph

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

def joint_pred_to_matrix(joint_type_pred, src, dst,num_joints):
    parts_conne = torch.zeros((num_joints, num_joints))
    for i in range(joint_type_pred.shape[0]):
        if joint_type_pred[i] > 0:
            parts_conne[src[i], dst[i]] = 1
            parts_conne[dst[i], src[i]] = 1
    return parts_conne

def canonical_direction(z):
    threshold = 0.15
    mask = (
        (z[:, 2] >= threshold) |
        ((z[:, 2] <= threshold) & (z[:, 1] >= threshold)) |
        ((z[:, 2] <= threshold) & (z[:, 1] <= threshold) & (z[:, 0] >= threshold))
    )

    return torch.where(mask.unsqueeze(1), z, -z)

def draw_example(parts_start_list, revolute_axis_pred=None, prismatic_axis_pred=None, screw_point_list_gt=None, revolute_pivot_pred=None):
    np.random.seed(0)
    part_pcds = []
    for part in parts_start_list:
        part_np = part.squeeze(0).cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(part_np)
        pcd.paint_uniform_color(np.random.rand(3))
        part_pcds.append(pcd)

    if revolute_axis_pred is not None:
        revolute_axis_pred_list = []
        pivot_point_list = []
        pivot_point_pred_list = []

        for i in range(screw_point_list_gt.shape[0]):
            revolute_axis_pred_vec = revolute_axis_pred[i].cpu().numpy()
            pivot_point_coor = screw_point_list_gt[i].cpu().numpy()
            revolute_pivot_pred_coor = revolute_pivot_pred[i].cpu().numpy()
            
            # Create arrow for revolute axis
            rev_axis1 = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.008,
                cone_radius=0.008,
                cylinder_height=0.7,
                cone_height=0.03
            )
            rev_axis1.compute_vertex_normals()
            rev_axis1.paint_uniform_color([0, 0, 1])  # Blue color for axis
            rev_axis2 = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.008,
                cone_radius=0.008,
                cylinder_height=0.7,
                cone_height=0.03
            )
            rev_axis2.compute_vertex_normals()
            rev_axis2.paint_uniform_color([0, 0, 1])  # Blue color for axis


            # Get rotation matrix from axis direction vector
            # The arrow is initially pointing up (along +Z axis), we need to rotate it to match the predicted axis direction
            # Calculate rotation between Z-axis and predicted axis
            z_axis = np.array([0, 0, 1])
            pred_axis = revolute_axis_pred_vec
            
            # Normalize the predicted axis vector
            pred_axis_norm = pred_axis / np.linalg.norm(pred_axis)
            
            # Calculate rotation using Rodrigues' rotation formula
            v = np.cross(z_axis, pred_axis_norm)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, pred_axis_norm)
            
            if s > 1e-6:  # If not parallel
                vx = np.array([[0, -v[2], v[1]],
                              [v[2], 0, -v[0]],
                              [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
            else:
                # If parallel, rotation is identity or 180 degree rotation
                R = np.eye(3) if c > 0 else -np.eye(3)
                R[2, 2] = 1 if c > 0 else -1
            
            # Apply rotation to the arrow
            rev_axis1.rotate(R, center=(0, 0, 0))
            rev_axis2.rotate(-R, center=(0, 0, 0))
            # Translate the arrow to the predicted pivot point location
            # First, we need to move it so the base of the arrow is at the pivot point
            # The arrow is created with its base at the origin
            rev_axis1.translate(revolute_pivot_pred_coor)
            rev_axis2.translate(revolute_pivot_pred_coor)

            revolute_axis_pred_list.append(rev_axis1)
            revolute_axis_pred_list.append(rev_axis2)

            pivot_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.03).translate(pivot_point_coor).paint_uniform_color([1, 0, 0])
            pivot_point_pred = o3d.geometry.TriangleMesh.create_sphere(radius=0.03).translate(revolute_pivot_pred_coor).paint_uniform_color([0, 1, 0])

            pivot_point_list.append(pivot_point)
            pivot_point_pred_list.append(pivot_point_pred)
        
        # Add revolute_axis_pred_list to the visualization
        o3d.visualization.draw_geometries([*part_pcds, *revolute_axis_pred_list, *pivot_point_pred_list])
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
    print(f"screw_axis_list_gt: \n{screw_axis_list_gt.shape}")
    rev_mask = jt_gt_edges.view(-1) == 0
    pri_mask = jt_gt_edges.view(-1) == 1

    revolute_cos, revolute_ang = [], []
    prismatic_cos, prismatic_ang = [], []

    if rev_mask.sum() > 0:
            # Compute per-edge parameter losses (L2)
        revolute_axis_pred = revolute_para_pred[:,:,:3][joint_mask].squeeze()
        rev_weights = torch.sigmoid(revolute_para_pred[:,:,3:4][joint_mask])
        revolute_axis_pred = (revolute_axis_pred * rev_weights).sum(dim=1) / (rev_weights.sum(dim=1) + 1e-6)
        pred_rev_axis = F.normalize(revolute_axis_pred, dim=1)[rev_mask]
        gt_rev   = axis_gt[rev_mask]
        print(f"pred_rev: \n{pred_rev_axis}\ngt_rev: \n{gt_rev}")
        revolute_cos = cosine_similarity_vec(pred_rev_axis, gt_rev).cpu().tolist()
        revolute_ang = angular_err_deg(pred_rev_axis, gt_rev).cpu().tolist()

        pivot_gt = pivot_gt[rev_mask]
        revolute_pivot_pred = revolute_para_pred[:,:,4:7][joint_mask].squeeze()
        rev_piv_weights = torch.sigmoid(revolute_para_pred[:,:,7:8][joint_mask])
        revolute_pivot_pred = (revolute_pivot_pred * rev_piv_weights).sum(dim=1) / (rev_piv_weights.sum(dim=1) + 1e-6)
        print(f"revolute_pivot_pred: \n{revolute_pivot_pred[rev_mask]}\npivot_gt: \n{pivot_gt}")
        print(f"revolute pivot pred shape: {revolute_pivot_pred[rev_mask].shape}")
        # Could also evaluate pivot point error here if desired
        pivot_dist = point_to_axis_distance(
        revolute_pivot_pred[rev_mask],
        pivot_gt,
        gt_rev).cpu().tolist()

        draw_example(parts_start_list, revolute_axis_pred=pred_rev_axis, screw_point_list_gt=screw_point_list_gt, revolute_pivot_pred=revolute_pivot_pred[rev_mask])
        adj_pred = joint_pred_to_matrix(edges_conne_pred, src[joint_mask], dst[joint_mask], parts_connections_gt.shape[0])
        joint_type_pred_valid = (torch.sigmoid(joint_type_pred[joint_mask])>0.5).float()
        revolute_mask = (joint_type_pred_valid == 0).squeeze()  # 0 = revolute
        if joint_type_pred_valid.shape[0]==1:
            if revolute_mask:
                axes_pred = pred_rev_axis.unsqueeze(0)
            else:
                axes_pred = pred_pri_axis.unsqueeze(0)
        else:
            axes_pred = pred_rev_axis[revolute_mask].squeeze(0)
            # axes_pred = torch.cat([revolute_axis_pred[revolute_mask], prismatic_axis_pred[~revolute_mask]], dim=0).squeeze(0)
        visualize_articulated_graph(adj_pred, adj, joint_type_pred_valid, axes_pred)

    if pri_mask.sum() > 0:
        prismatic_axis_pred = prismatic_para_pred[:,:,:3][joint_mask].squeeze()
        pri_weights = torch.sigmoid(prismatic_para_pred[:,:,3:4][joint_mask])
        prismatic_axis_pred = (prismatic_axis_pred * pri_weights).sum(dim=1) / (pri_weights.sum(dim=1) + 1e-6)
        pred_pri_axis = F.normalize(prismatic_axis_pred, dim=1)[pri_mask]
        gt_pri   = axis_gt[pri_mask]
        print(f"pred_pri: \n{pred_pri_axis}\ngt_pri: \n{gt_pri}")
        prismatic_cos = cosine_similarity_vec(pred_pri_axis, gt_pri).cpu().tolist()
        prismatic_ang = angular_err_deg(pred_pri_axis, gt_pri).cpu().tolist()

        # draw_example(parts_start_list, prismatic_axis_pred=pred_pri_axis)



# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def evaluate(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PartsGraphDataset2(
        "../Ditto/Articulated_object_simulation-main/data/Shape2Motion_gcn/LRW/*/val/scenes/*.npz",
        device
    )
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    params = {
        "pointnet_dim": 1024,
        "nlayers": 4,
        "nhidden": 512,
        "out_dim": 256,
        "dropout": 0.3,
        "lamda": 0.5,
        "alpha": 0.1,
        "variant": True,
        "nhidden_mlp": 256,
        "n_class": 1,
        "latent_dim": 1,
        "decoder_out_dim": 128,
        "motion_decoder_out_dim": 256,
    }


    model = parts_connection_mlp(**params).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # np.random.seed()
    # target_idx = np.random.randint(0, len(val_loader)-1)  # choose the example you want
    target_idx = 134  # you can also manually set the index here
    with torch.no_grad():
        for index, data in enumerate(val_loader):
            if index != target_idx:
                continue

            print(f"Evaluating sample {index}")
            eval_step(model, data)
            break



if __name__ == "__main__":
    chk = "./pre_trained_models_gcnpp/2026-01-27 15_52_25_L.R.W/chkpt_best_model_val.pth"
    chk = "./pre_trained_models_gcnpp/2026-01-28 13_44_00_L.R.W/chkpt_best_model_val.pth"
    metrics = evaluate(chk)
    print(f"\n\n{metrics}")
