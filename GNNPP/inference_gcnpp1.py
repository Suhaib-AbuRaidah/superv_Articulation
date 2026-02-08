from xml.parsers.expat import model
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append(os.path.expanduser('~/superv_Articulation'))
import open3d as o3d
from utilis.dataset2 import PartsGraphDataset2
from GNNPP.gnn_pointnet2_network_v2 import parts_connection_mlp
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

def draw_example(parts_start_list, revolute_axis_pred=None, prismatic_axis_pred=None, revolute_pivot_pred=None, jt = None):
    np.random.seed(0)
    part_pcds = []
    for k, part in enumerate(parts_start_list):
        part_np = part.squeeze(0).cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(part_np)
        if k == 0:
            pcd.paint_uniform_color([1, 0, 0])
        elif k == 1:
            pcd.paint_uniform_color([0, 1, 0])
        elif k == 2:
            pcd.paint_uniform_color([0, 0, 1])
        elif k == 3:
            pcd.paint_uniform_color([1, 1, 0])
        elif k == 4:
            pcd.paint_uniform_color([0, 1, 1])
        elif k == 5:
            pcd.paint_uniform_color([1, 0, 1])
        part_pcds.append(pcd)

    revolute_axis_pred_list = []
    pivot_point_pred_list = []
    prismatic_axis_pred_list = []

    # Helper: rotation matrix that aligns +Z to a given direction
    def _R_from_z_to_dir(dir_vec):
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        pred_axis = dir_vec.astype(np.float64)
        n = np.linalg.norm(pred_axis) + 1e-12
        pred_axis_norm = pred_axis / n

        v = np.cross(z_axis, pred_axis_norm)
        s = np.linalg.norm(v)
        c = float(np.dot(z_axis, pred_axis_norm))

        if s > 1e-6:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]], dtype=np.float64)
            R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
        else:
            # parallel or anti-parallel
            if c > 0:
                R = np.eye(3)
            else:
                # 180 deg rotation about X or Y; choose X
                R = np.array([[1, 0, 0],
                              [0, -1, 0],
                              [0, 0, -1]], dtype=np.float64)
        return R

    # -------- Revolute: draw axis arrows + pivot spheres --------
    if revolute_axis_pred is not None and revolute_pivot_pred is not None:
        for i in range(revolute_axis_pred.shape[0]):
            revolute_axis_pred_vec = revolute_axis_pred[i].detach().cpu().numpy()
            revolute_pivot_pred_coor = revolute_pivot_pred[i].detach().cpu().numpy()

            rev_axis1 = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.008,
                cone_radius=0.008,
                cylinder_height=0.7,
                cone_height=0.03
            )
            rev_axis1.compute_vertex_normals()
            rev_axis1.paint_uniform_color([0, 0, 1])

            rev_axis2 = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.008,
                cone_radius=0.008,
                cylinder_height=0.7,
                cone_height=0.03
            )
            rev_axis2.compute_vertex_normals()
            rev_axis2.paint_uniform_color([0, 0, 1])

            R = _R_from_z_to_dir(revolute_axis_pred_vec)
            rev_axis1.rotate(R, center=(0, 0, 0))
            rev_axis2.rotate(-R, center=(0, 0, 0))

            rev_axis1.translate(revolute_pivot_pred_coor)
            rev_axis2.translate(revolute_pivot_pred_coor)

            revolute_axis_pred_list.append(rev_axis1)
            revolute_axis_pred_list.append(rev_axis2)

            pivot_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            pivot_sphere.compute_vertex_normals()
            pivot_sphere.translate(revolute_pivot_pred_coor)
            pivot_sphere.paint_uniform_color([0, 1, 0])
            pivot_point_pred_list.append(pivot_sphere)

    # -------- Prismatic: draw translation axis (no pivot required) --------
    # Strategy: anchor each axis at the centroid of the corresponding part's points.
    if prismatic_axis_pred is not None:
        jt_arr = jt.detach().cpu().numpy()
        indices = np.where(jt_arr==1)[0]
        for i in range(prismatic_axis_pred.shape[0]):
            index = indices[i]
            pri_axis_vec = prismatic_axis_pred[i].detach().cpu().numpy()

            anchor = np.mean(np.asarray(part_pcds[index+1].points), axis=0)


            pri_axis1 = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.008,
                cone_radius=0.008,
                cylinder_height=0.7,
                cone_height=0.03
            )

            pri_axis1.compute_vertex_normals()
            pri_axis1.paint_uniform_color([1, 0, 0])  # Red for prismatic

            pri_axis2 = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.008,
                cone_radius=0.008,
                cylinder_height=0.7,
                cone_height=0.03
            )
            pri_axis2.compute_vertex_normals()
            pri_axis2.paint_uniform_color([1, 0, 0])

            R = _R_from_z_to_dir(pri_axis_vec)
            pri_axis1.rotate(R, center=(0, 0, 0))
            pri_axis2.rotate(-R, center=(0, 0, 0))

            pri_axis1.translate(anchor)
            pri_axis2.translate(anchor)

            prismatic_axis_pred_list.append(pri_axis1)
            prismatic_axis_pred_list.append(pri_axis2)

    # -------- visualize --------
    o3d.visualization.draw_geometries([
        *part_pcds,
        *revolute_axis_pred_list,
        *pivot_point_pred_list,
        *prismatic_axis_pred_list
    ])

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
    
    # Keep only what is needed for inference + visualization
    adj = adj.squeeze().to(device)
    N = adj.shape[0]  # number of parts
    # -------- model forward --------
    edges_conne_pred, joint_type_pred, revolute_para_pred, prismatic_para_pred, (src, dst) = \
        model(parts_start_list, parts_end_list, adj)

    # Aggregate per-point logits -> per-edge logits (if your outputs are [E, P, 1] this becomes [E, 1])
    edges_conne_pred = edges_conne_pred.mean(dim=1)
    joint_type_pred = joint_type_pred.mean(dim=1)

    src = src.long().to(device)
    dst = dst.long().to(device)
    if src.numel() == 0:
        # Nothing to visualize
        return

    # -------- predicted joint mask (from predicted connectivity) --------
    conn_pred_lbl = binarize(edges_conne_pred)          # expected shape [E,1] or [E]
    joint_mask_pred = (conn_pred_lbl > 0).view(-1)      # [E_total]
    E = int(joint_mask_pred.sum().item())

    if E == 0:
        # Visualize empty prediction if you want; otherwise just return
        # You can still build adj_pred = zeros and call visualize if your function supports it
        print("No joints predicted")
        return

    # -------- axes prediction on predicted joints only --------
    # revolute axis
    rev_axis_ep3 = revolute_para_pred[joint_mask_pred, :, :3]                 # [E, P, 3]
    rev_w_ep1    = torch.sigmoid(revolute_para_pred[joint_mask_pred, :, 3:4]) # [E, P, 1]
    rev_axis_e3  = (rev_axis_ep3 * rev_w_ep1).sum(dim=1) / (rev_w_ep1.sum(dim=1) + 1e-6)  # [E, 3]
    rev_axis_e3  = F.normalize(rev_axis_e3, dim=1)

    rev_pivot_ep3 = revolute_para_pred[:, :, 4:7][joint_mask_pred].squeeze()                # [E, P, 3]
    rev_pivot_w_ep1 = torch.sigmoid(revolute_para_pred[:, :, 7:8][joint_mask_pred]) # [E, P, 1]
    rev_pivot_e3  = (rev_pivot_ep3 * rev_pivot_w_ep1).sum(dim=1) / (rev_pivot_w_ep1.sum(dim=1) + 1e-6)  # [E, 3]

    screw_point_list_gt = screw_point_list_gt.squeeze().view(-1, 3).to(device)

    # prismatic axis
    pri_axis_ep3 = prismatic_para_pred[joint_mask_pred, :, :3]                # [E, P, 3]
    pri_w_ep1    = torch.sigmoid(prismatic_para_pred[joint_mask_pred, :, 3:4])# [E, P, 1]
    pri_axis_e3  = (pri_axis_ep3 * pri_w_ep1).sum(dim=1) / (pri_w_ep1.sum(dim=1) + 1e-6)  # [E, 3]
    pri_axis_e3  = F.normalize(pri_axis_e3, dim=1)

    # -------- predicted joint type for those predicted joints --------
    jt = (torch.sigmoid(joint_type_pred[joint_mask_pred]) > 0.5).float().view(-1)  # [E]
    pred_is_prismatic = (jt == 1)
    pred_is_revolute  = ~pred_is_prismatic

    # -------- build axes for visualization aligned with predicted joints --------
    axes_pred = torch.zeros((E, 3), device=device, dtype=rev_axis_e3.dtype)
    axes_pred[pred_is_revolute]  = rev_axis_e3[pred_is_revolute]
    axes_pred[pred_is_prismatic] = pri_axis_e3[pred_is_prismatic]

    pivot_pred_revolute = rev_pivot_e3[pred_is_revolute]

    # -------- build predicted adjacency + visualize --------
    # try:
    adj_pred = joint_pred_to_matrix(
        edges_conne_pred,
        src[joint_mask_pred],
        dst[joint_mask_pred],
        adj.shape[0]  # number of parts
    )

    print(f"pivot_pred_revolute: \n{pivot_pred_revolute}")
    draw_example(parts_start_list, revolute_axis_pred=axes_pred[pred_is_revolute], prismatic_axis_pred=axes_pred[pred_is_prismatic], revolute_pivot_pred= pivot_pred_revolute,jt = jt)
    visualize_articulated_graph(adj_pred, adj, jt.unsqueeze(1), axes_pred)
    # except Exception as e:
    #     print(f"Couldn't generate adj_pred: {e}")
    # ===================== METRICS (added after visualization) =====================
    # Move GT to device (kept here to avoid changing anything before visualize_articulated_graph)
    parts_connections_gt = parts_connections_gt.squeeze().to(device)
    joint_type_list_gt = joint_type_list_gt.view(-1).to(device)          # [E_gt]
    screw_axis_list_gt = screw_axis_list_gt.squeeze().view(-1, 3).to(device)
    screw_axis_list_gt = F.normalize(screw_axis_list_gt, dim=1)
    screw_point_list_gt = screw_point_list_gt.squeeze().view(-1, 3).to(device)
    try:
        # -------- Part connection metrics (edge-level, over model candidate edges src,dst) --------
        conn_gt = parts_connections_gt[src, dst].float().view(-1, 1)  # [E_total,1]
        conn_pred_lbl_m = (conn_pred_lbl.view(-1, 1) > 0).float()     # [E_total,1]

        tp = ((conn_pred_lbl_m == 1) & (conn_gt == 1)).sum().item()
        fp = ((conn_pred_lbl_m == 1) & (conn_gt == 0)).sum().item()
        fn = ((conn_pred_lbl_m == 0) & (conn_gt == 1)).sum().item()
        tn = ((conn_pred_lbl_m == 0) & (conn_gt == 0)).sum().item()

        conn_acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
        conn_prec = tp / max(tp + fp, 1)
        conn_rec  = tp / max(tp + fn, 1)
        conn_f1   = 2 * conn_prec * conn_rec / max(conn_prec + conn_rec, 1e-12)

        # -------- Helper functions for axis metrics --------
        def _cosine_abs(a, b, eps=1e-8):
            a = F.normalize(a, dim=1)
            b = F.normalize(b, dim=1)
            c = (a * b).sum(dim=1)
            return c.abs().clamp(eps, 1.0)

        def _angular_err_deg_from_cos(cosv):
            cosv = cosv.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            return torch.rad2deg(torch.acos(cosv))

        def _point_to_axis_distance(points, axis_points, axis_dirs):
            # points: [K,3], axis_points: [K,3], axis_dirs: [K,3] (assumed normalized)
            v = points - axis_points
            cross = torch.linalg.cross(v, axis_dirs, dim=1)
            return torch.linalg.norm(cross, dim=1)

        # -------- Joint-type + axis/pivot metrics (aligned by predicted joints) --------
        # Build GT edge map so we can align GT params to predicted edges
        gt_src, gt_dst = torch.where(parts_connections_gt > 0)

        edge2idx = {(int(s.item()), int(d.item())): i for i, (s, d) in enumerate(zip(gt_src, gt_dst))}

        # Predicted joint edges
        src_pred_e = src[joint_mask_pred]
        dst_pred_e = dst[joint_mask_pred]

        # Map predicted edges -> GT indices (assumption: number of parts: 5predicted connections match GT, but we still map safely)
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
            joint_acc = 0.0
            joint_f1 = 0.0
            revolute_cos = []
            revolute_ang = []
            prismatic_cos = []
            prismatic_ang = []
            revolute_pivot_dist = []
        else:
            valid_edge_mask = torch.tensor(valid_edge_mask_list, device=device, dtype=torch.bool)
            gt_idx = torch.tensor(gt_idx_list, device=device, dtype=torch.long)

            # Align everything to "valid predicted joint edges"
            jt_pred_valid = jt[valid_edge_mask].long()              # [E_valid]
            axes_pred_valid = axes_pred[valid_edge_mask]            # [E_valid,3]
            src_valid = src_pred_e[valid_edge_mask]
            dst_valid = dst_pred_e[valid_edge_mask]

            jt_gt_edges = joint_type_list_gt[gt_idx].long()         # [E_valid]
            axis_gt_edges = screw_axis_list_gt[gt_idx]              # [E_valid,3]
            pivot_gt_edges = screw_point_list_gt[gt_idx]            # [E_valid,3]

            # Joint type metrics
            tp2 = ((jt_pred_valid == 1) & (jt_gt_edges == 1)).sum().item()
            fp2 = ((jt_pred_valid == 1) & (jt_gt_edges == 0)).sum().item()
            fn2 = ((jt_pred_valid == 0) & (jt_gt_edges == 1)).sum().item()
            tn2 = ((jt_pred_valid == 0) & (jt_gt_edges == 0)).sum().item()

            joint_acc = (tp2 + tn2) / max(tp2 + tn2 + fp2 + fn2, 1)
            prec2 = tp2 / max(tp2 + fp2, 1)
            rec2  = tp2 / max(tp2 + fn2, 1)
            joint_f1 = 2 * prec2 * rec2 / max(prec2 + rec2, 1e-12)

            # Axis metrics split by GT type (0=revolute, 1=prismatic)
            rev_gt_mask = (jt_gt_edges == 0)
            pri_gt_mask = (jt_gt_edges == 1)

            revolute_cos, revolute_ang, revolute_pivot_dist = [], [], []
            prismatic_cos, prismatic_ang = [], []

            # Revolute axis + pivot metrics
            if rev_gt_mask.any():
                pred_rev_axis = axes_pred_valid[rev_gt_mask]
                gt_rev_axis = axis_gt_edges[rev_gt_mask]

                cosv = _cosine_abs(pred_rev_axis, gt_rev_axis)
                ang  = _angular_err_deg_from_cos(cosv)

                revolute_cos = cosv.detach().cpu().tolist()
                revolute_ang = ang.detach().cpu().tolist()

                # Revolute pivot metric: distance from predicted pivot to GT axis line
                # Use your predicted revolute pivots for predicted-revolute edges; to evaluate with GT revolute set,
                # we compute pivots for all predicted joints then subset to rev_gt_mask.
                pivot_pred_valid = rev_pivot_e3[valid_edge_mask]  # [E_valid,3] (still revolute pivot head output)
                pred_rev_pivot = pivot_pred_valid[rev_gt_mask]
                gt_rev_pivot = pivot_gt_edges[rev_gt_mask]
                gt_rev_axis_n = F.normalize(gt_rev_axis, dim=1)

                dist = _point_to_axis_distance(pred_rev_pivot, gt_rev_pivot, gt_rev_axis_n)
                revolute_pivot_dist = dist.detach().cpu().tolist()

            # Prismatic axis metrics
            if pri_gt_mask.any():
                pred_pri_axis = axes_pred_valid[pri_gt_mask]
                gt_pri_axis = axis_gt_edges[pri_gt_mask]

                cosv = _cosine_abs(pred_pri_axis, gt_pri_axis)
                ang  = _angular_err_deg_from_cos(cosv)

                prismatic_cos = cosv.detach().cpu().tolist()
                prismatic_ang = ang.detach().cpu().tolist()

        # Keep return lightweight (optional); remove if your pipeline expects a dict
        return {
            # ----- added metrics -----
            "number of parts": N,
            
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
    except Exception as e:
        print(f"Error occurred during evaluation (predicted connections are not as gt)")
        return None
# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def evaluate(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PartsGraphDataset2(os.path.expanduser(
        "~/superv_Articulation/data/Shape2Motion_gcn/cabinet_simp/train/scenes/*.npz"),
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
    rev_pivot_dst = []
    # np.random.seed()
    target_idx = np.random.randint(0, len(val_loader)-1)  # choose the example you want
    target_idx = 47  # you can also manually set the index here
    with torch.no_grad():
        for index, data in enumerate(val_loader):
            if index != target_idx:
                continue

            print(f"Evaluating sample {index}")
            metrics = eval_step(model, data)
            if metrics is not None:
                for key, value in metrics.items():
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    chk = os.path.expanduser("~/superv_Articulation/pre_trained_models_gcnpp/2026-02-04 20:21:38_C/chkpt_best_model_train.pth")
    evaluate(chk)

