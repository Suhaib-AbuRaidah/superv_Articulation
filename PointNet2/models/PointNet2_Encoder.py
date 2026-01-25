import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.append('/home/suhaib/superv_Articulation/PointNet2')
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134+additional_channel, mlp=[128, 64])

    def forward(self, xyz):
        """
        Args:
            xyz: (B, 3, N)

        Returns:
            global_feat: (B, 1024)
            local_feats: (B, N, 128)
        """

        B, C, N = xyz.shape

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz

        # -------- Set Abstraction --------
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)     # (B, 64+128+128, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)     # (B, 256+256, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)     # (B, 1024, 1)

        # -------- Global feature --------
        global_feat = l3_points.squeeze(-1)                 # (B, 1024)

        # -------- Feature Propagation --------
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        l0_points = self.fp1(
            l0_xyz,
            l1_xyz,
            torch.cat([l0_xyz, l0_points], dim=1),
            l1_points
        )                                                    # (B, 64, N)
        global_feat_rep = global_feat.view(B, -1, 1).repeat(1, 1, N)  # (B, 1024, N)
        l0_points = torch.cat([global_feat_rep, l0_points], dim=1)  # (B, 1024+64, N)
        # -------- Local features --------
        local_feats = l0_points.permute(0, 1, 2).contiguous()  # (B, N, 1088)

        return global_feat, local_feats



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss