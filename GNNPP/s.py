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
    screw_axis_list_gt = F.normalize(screw_axis_list_gt, dim=1)
    screw_point_list_gt = screw_point_list_gt.squeeze().view(-1,3)
    joint_type_list_gt = joint_type_list_gt.squeeze()
    angles = angles.squeeze().view(-1,1)

    # Forward pass
    edges_conne_pred, joint_type_pred, revolute_para_pred, prismatic_para_pred, (src, dst) = model(parts_start_list, parts_end_list, adj)
    conn_gt = parts_connections_gt[src, dst].float().unsqueeze(1)  # [num_edges, 1]

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
    revolute_axis_pred = revolute_para_pred[:,:,:3][joint_mask].squeeze()
    rev_weights = torch.sigmoid(revolute_para_pred[:,:,3:4][joint_mask])
    revolute_axis_pred = (revolute_axis_pred * rev_weights).sum(dim=1) / (rev_weights.sum(dim=1) + 1e-6)
    revolute_axis_pred = F.normalize(revolute_axis_pred, dim=1)
    
    revolute_pivot_pred = revolute_para_pred[:,:,4:7][joint_mask].squeeze()
    rev_pivot_weights = torch.sigmoid(revolute_para_pred[:,:,7:8][joint_mask])
    revolute_pivot_pred = (revolute_pivot_pred * rev_pivot_weights).sum(dim=1) / (rev_pivot_weights.sum(dim=1) + 1e-6)

    prismatic_axis_pred = prismatic_para_pred[:,:,:3][joint_mask].squeeze()
    pri_weights = torch.sigmoid(prismatic_para_pred[:,:,3:4][joint_mask])
    prismatic_axis_pred = (prismatic_axis_pred * pri_weights).sum(dim=1) / (pri_weights.sum(dim=1) + 1e-6)
    prismatic_axis_pred = F.normalize(prismatic_axis_pred, dim=1)
    revolute_axis_loss = 1-torch.abs(torch.sum(revolute_axis_pred * screw_axis_list_gt, dim=1)).mean()
    revolute_pivot_loss = torch.sqrt(F.mse_loss(revolute_pivot_pred, screw_point_list_gt, reduction='none').clamp(min=1e-12)).mean(1)
    revolute_loss = revolute_axis_loss + revolute_pivot_loss

    prismatic_loss = 1-torch.abs(torch.sum(prismatic_axis_pred * screw_axis_list_gt, dim=1)).mean()
    # Apply masks

    revolute_loss = (revolute_loss * revolute_mask.float().view(-1))
    revolute_loss = revolute_loss.mean()
    prismatic_loss = (prismatic_loss * prismatic_mask.float().view(-1))
    prismatic_loss = prismatic_loss.mean()
    
    total_loss = loss_part_conn + loss_joint_type + revolute_loss + prismatic_loss #+ loss_latent
    return total_loss, loss_part_conn, loss_joint_type, revolute_loss, prismatic_loss #, loss_latent