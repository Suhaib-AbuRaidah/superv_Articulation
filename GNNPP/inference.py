from xml.parsers.expat import model
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append(os.path.expanduser('~/superv_Articulation'))
import open3d as o3d
from utilis.dataset2 import PartsGraphDataset3
from GNNPP.gnn_pointnet2_network_v3 import parts_connection_mlp
from torch.utils.data import DataLoader
from utilis.Inference_graph import visualize_articulated_graph
from utilis.contact_points import find_contact_points

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


def draw_example(pc_start, seg_mask_start, rev_mask = None,revolute_axis_pred=None, prismatic_axis_pred=None, revolute_pivot_pred=None, jt = None):
    part_pcds = []
    pc_start_np = pc_start.squeeze(0).cpu().numpy()
    color_pallet = [
        [1, 0, 0],  # Red for part 1
        [0, 1, 0],  # Green for part 2
        [0, 0, 1],  # Blue for part 3
        [1, 0, 1],  # purpule for part 4
        [0, 1, 1],  # Cyan for part 5
    ]
    for i, seg in enumerate(seg_mask_start):
        mask = seg.squeeze(0).cpu().numpy()
        if i == 0:
            color = np.array(color_pallet[0])  # Red for part 1
        elif i == 1:
            color = np.array(color_pallet[1])  # Green for part 2
        elif i == 2:
            color = np.array(color_pallet[2])  # Blue for part 3
        elif i == 3:
            color = np.array(color_pallet[3])  # purpule for part 4
        elif i == 4:
            color = np.array(color_pallet[4])  # Cyan for part 5

        part_points = pc_start_np[mask]
        part_pcd = o3d.geometry.PointCloud()
        part_pcd.points = o3d.utility.Vector3dVector(part_points)
        part_pcd.paint_uniform_color(color)
        part_pcds.append(part_pcd)


    if rev_mask is not None:
        mask_rev = rev_mask.squeeze(0).cpu().numpy()
        rev_indices = np.where(mask_rev==1)[0]

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

            rev_axis1.paint_uniform_color([0, 0, 0])  # Black for revolute

            rev_axis2 = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.008,
                cone_radius=0.008,
                cylinder_height=0.7,
                cone_height=0.03
            )
            rev_axis2.compute_vertex_normals()
            rev_axis2.paint_uniform_color([0, 0, 0])

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
            if i == 0:
                pivot_sphere.paint_uniform_color(color_pallet[rev_indices[0]+1])  # Red for pivot of joint 1
            elif i == 1:
                pivot_sphere.paint_uniform_color(color_pallet[rev_indices[1]+1])
            elif i == 2:
                pivot_sphere.paint_uniform_color(color_pallet[rev_indices[2]+1])
            elif i == 3:
                pivot_sphere.paint_uniform_color(color_pallet[rev_indices[3]+1])
            elif i == 4:
                pivot_sphere.paint_uniform_color(color_pallet[rev_indices[4]+1])

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
            pri_axis1.paint_uniform_color([0.7, 0.7, 0.7])  # Red for prismatic

            pri_axis2 = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.008,
                cone_radius=0.008,
                cylinder_height=0.7,
                cone_height=0.03
            )
            pri_axis2.compute_vertex_normals()
            pri_axis2.paint_uniform_color([0.7, 0.7, 0.7])

            R = _R_from_z_to_dir(pri_axis_vec)
            pri_axis1.rotate(R, center=(0, 0, 0))
            pri_axis2.rotate(-R, center=(0, 0, 0))

            pri_axis1.translate(anchor)
            pri_axis2.translate(anchor)

            prismatic_axis_pred_list.append(pri_axis1)
            prismatic_axis_pred_list.append(pri_axis2)

    o3d.visualization.draw_geometries([part_pcds[1]], window_name="Contact Points Visualization", width=800, height=600)
    # -------- visualize --------
    o3d.visualization.draw_geometries([
        *part_pcds,
        *revolute_axis_pred_list,
        *pivot_point_pred_list,
        *prismatic_axis_pred_list
    ])

# ------------------------------------------------------------
# Inference step for a single sample
# ------------------------------------------------------------
def inference(model, data_dict, verbose=False, skip_invalid=True):
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

    # --- ADDED: pivot projection direction + distance losses (paper-style) ---
    # Predicted projection direction d_p and signed distance h_p (per-point)
    edge_dir_pred = revolute_para_pred[:, :, 4:7][joint_mask_pred]              # (E_joint, P_edge, 3)
    edge_dir_pred = F.normalize(edge_dir_pred, dim=-1)
    edge_dst_pred = revolute_para_pred[:, :, 7:8][joint_mask_pred]              # (E_joint, P_edge, 1)
        # -------- Per-part batching --------
    pcs1 = torch.cat([pc.transpose(1, 2) for pc in parts_start_list], dim=0)  # [N, 3, P]
    edge_points = torch.cat([pcs1[src], pcs1[dst]], dim=2).transpose(1, 2)  # [N, 2P, 3]
    edge_pivot_per_point = edge_points[joint_mask_pred]+edge_dir_pred*edge_dst_pred
    rev_edge_pivots = edge_pivot_per_point.mean(dim=1)  # (E_joint, 3)

    for i in range(edge_points[joint_mask_pred].shape[0]):
        part_1 = edge_points[i, :edge_points.shape[1]//2].squeeze(0).cpu().numpy()
        part_2 = edge_points[i, edge_points.shape[1]//2:].squeeze(0).cpu().numpy()
        closest_contact = find_contact_points(part_1, part_2, predicted_pivot=rev_edge_pivots[i].detach().cpu().numpy())
        rev_edge_pivots[i] = torch.from_numpy(closest_contact).to(device)



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

    pivot_pred_revolute = rev_edge_pivots[pred_is_revolute]

    # -------- build predicted adjacency + visualize --------
    # try:
    adj_pred = joint_pred_to_matrix(
        edges_conne_pred,
        src[joint_mask_pred],
        dst[joint_mask_pred],
        adj.shape[0]  # number of parts
    )

    draw_example(pc_start_unsampled,seg_mask_start_unsampled, rev_mask = pred_is_revolute,revolute_axis_pred=axes_pred[pred_is_revolute], prismatic_axis_pred=axes_pred[pred_is_prismatic], revolute_pivot_pred= pivot_pred_revolute,jt = jt)
    visualize_articulated_graph(adj_pred, adj)

# ------------------------------------------------------------
# Main inference
# ------------------------------------------------------------
def apply_inference(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PartsGraphDataset3(os.path.expanduser(
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
        "nhidden_mlp": 512,
        "n_class": 1,
        "latent_dim": 1,
        "decoder_out_dim": 128,
        "motion_decoder_out_dim": 128,
    }


    model = parts_connection_mlp(**params).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # np.random.seed()
    # target_idx = np.random.randint(0, len(val_loader)-1)  # choose the example you want
    target_idx = 54  # you can also manually set the index here
    with torch.no_grad():
        for index, data in enumerate(val_loader):
            if index != target_idx:
                continue

            print(f"Inference sample {index}")
            inference(model, data)


if __name__ == "__main__":
    chk = os.path.expanduser("~/superv_Articulation/pre_trained_models_gcnpp/2026-02-17 18:21:40_C/chkpt_best_model_train.pth")
    apply_inference(chk)

