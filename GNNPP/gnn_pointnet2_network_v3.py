import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
import os
sys.path.append(os.path.expanduser('~/superv_Articulation'))
from PointNet2.models.PointNet2_Encoder import get_model as PointNetEncoder
# import open3d as o3d
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, out_dim, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, out_dim))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner

class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, out_dim, dropout, lamda, alpha,variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, out_dim))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner
    
class CrossAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = math.sqrt(d)

    def forward(self, f1, f2):
        # support both 2D [N,d] and 3D [B,N,d]
        if f1.dim() == 2:
            attn = torch.softmax((f1 @ f2.T) / self.scale, dim=-1)
            out = attn @ f2
            return torch.cat([f1, out], dim=1)
        elif f1.dim() == 3:
            # f1, f2: [B, N, d]
            attn = torch.softmax(torch.matmul(f1, f2.transpose(-2, -1)) / self.scale, dim=-1)  # [B,N,N]
            out = torch.matmul(attn, f2)  # [B,N,d]
            return torch.cat([f1, out], dim=-1)  # [B,N,2d]
        else:
            raise ValueError("Unsupported tensor dims for CrossAttention")


class FeatureAutoEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=1,decoder_out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, decoder_out_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


class DualPointNetGCNII(nn.Module):
    def __init__(self, pointnet_out_dim, nlayers, nhidden, out_dim, dropout,
                 lamda, alpha, variant, latent_dim, decoder_out_dim, motion_decoder_out_dim):
        super().__init__()

        d = pointnet_out_dim
        local_dim = 64

        # PointNet encoders
        self.pointnet_p1_enc_parts = PointNetEncoder()
        self.pointnet_p2_enc_parts = PointNetEncoder()
        self.pointnet_p1_enc_obj = PointNetEncoder()
        self.pointnet_p2_enc_obj = PointNetEncoder()

        # Motion parameter decoder (per part)
        self.motion_decoder_r = nn.Sequential(
            nn.Linear(4*d + 2*local_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, motion_decoder_out_dim)
        )

        self.motion_decoder_p = nn.Sequential(
            nn.Linear(4*d + 2*local_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, motion_decoder_out_dim)
        )

        self.connection_decoder = nn.Sequential(
            nn.Linear(4*d + 2*local_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, motion_decoder_out_dim)
        )

        self.joint_type_decoder = nn.Sequential(
            nn.Linear(4*d + 2*local_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, motion_decoder_out_dim)
        )
        # Cross-attention modules
        self.attn_kine_global = CrossAttention(2 * d)
        self.attn_kine_local = CrossAttention(2 * d + local_dim)
        self.attn_motion_local = CrossAttention(2 * d + local_dim)

        # # Autoencoder
        # self.ae = FeatureAutoEncoder(
        #     in_dim=4 * d,
        #     latent_dim=latent_dim,
        #     decoder_out_dim=decoder_out_dim
        # )

        # GCN
        gcn_in_dim = 4 * d #+ decoder_out_dim
        self.gcn = GCNII(
            nfeat=gcn_in_dim,
            nlayers=nlayers,
            nhidden=nhidden,
            out_dim=out_dim,
            dropout=dropout,
            lamda=lamda,
            alpha=alpha,
            variant=variant
        )


    def forward(self, part_pcs1, part_pcs2, adj):
        """
        part_pcs1, part_pcs2: list of tensors, each [P, 3]
        adj: [N, N]
        """
        N = len(part_pcs1)
        P = part_pcs1[0].shape[1]
        # -------- Per-part batching --------
        pcs1 = torch.cat([pc.transpose(1, 2) for pc in part_pcs1], dim=0)  # [N, 3, P]
        pcs2 = torch.cat([pc.transpose(1, 2) for pc in part_pcs2], dim=0)  # [N, 3, P]
   
        # -------- Whole-object batching --------
        total_object_1 = torch.cat([pc.transpose(1, 2) for pc in part_pcs1], dim=2)  # [1, 3, N·P]
        total_object_2 = torch.cat([pc.transpose(1, 2) for pc in part_pcs2], dim=2)  # [1, 3, N·P]

        # -------- PointNet encoding --------
        f1_global, f1_local = self.pointnet_p1_enc_parts(pcs1)  # [N, d], [N, d+64, P]
        f2_global, f2_local = self.pointnet_p2_enc_parts(pcs2)  # [N, d], [N, d+64, P]

        obj1_global, _ = self.pointnet_p1_enc_obj(total_object_1)  # [1, d]
        obj2_global, _ = self.pointnet_p2_enc_obj(total_object_2)  # [1, d]

        # -------- Global feature augmentation --------
        obj1_global_rep = obj1_global.repeat(N, 1)
        obj2_global_rep = obj2_global.repeat(N, 1)

        f1_global = torch.cat([f1_global, obj1_global_rep], dim=1)  # [N, 2d]
        f2_global = torch.cat([f2_global, obj2_global_rep], dim=1)  # [N, 2d]

        # -------- Local feature augmentation --------
        obj1_global_exp = obj1_global.unsqueeze(2).repeat(N, 1, P)  # [N, d, P]
        obj2_global_exp = obj2_global.unsqueeze(2).repeat(N, 1, P)  # [N, d, P]

        f1_local = torch.cat([f1_local, obj1_global_exp], dim=1)  # [N, 2d+64, P]
        f2_local = torch.cat([f2_local, obj2_global_exp], dim=1)  # [N, 2d+64, P]

        # f_1_2_global_lst = []
        # f_1_2_local_lst = []
        # for i in range(1):
        #     # -------- PointNet encoding --------
        #     f1_global, f1_local = self.pointnet_p1_encs[i](pcs1)   # [N, d], [N, d+64, P]
        #     f2_global, f2_local = self.pointnet_p2_encs[i](pcs2)

        #     obj1_global, _ = self.pointnet_p1_encs[i](total_object_1) # [1,d]
        #     obj2_global, _ = self.pointnet_p2_encs[i](total_object_2)
        #     if i ==0:
        #         # -------- Global feature augmentation --------
        #         obj1_global_rep = obj1_global.repeat(N, 1)
        #         obj2_global_rep = obj2_global.repeat(N, 1)

        #         f1_global = torch.cat([f1_global, obj1_global_rep], dim=1)  # [N, 2d]
        #         f2_global = torch.cat([f2_global, obj2_global_rep], dim=1)  # [N, 2d]
        #         f_1_2_global_lst.append((f1_global, f2_global))
        #     else:
        #         # -------- Local feature augmentation --------
        #         obj1_global_exp = obj1_global.unsqueeze(2).repeat(N, 1, P)  # [N, d, P]
        #         obj2_global_exp = obj2_global.unsqueeze(2).repeat(N, 1, P)  # [N, d, P]

        #         f1_local = torch.cat([f1_local, obj1_global_exp], dim=1)  # [N, 2d+64, P]
        #         f2_local = torch.cat([f2_local, obj2_global_exp], dim=1)  # [N, 2d+64, P]
        #         f_1_2_local_lst.append((f1_local, f2_local))

        # -------- Cross-attention --------
        f12_kine_global = self.attn_kine_global(f1_global, f2_global)  # [N, 4d+2*64]
        f12_kine_local = self.attn_kine_local(f1_local.transpose(1, 2), f2_local.transpose(1, 2))  # [N, P, 4d+2*64]
        f12_motion_local = self.attn_motion_local(f1_local.transpose(1, 2), f2_local.transpose(1, 2))  # [N, P, 4d+2*64]
        # f12_local_joint_type = self.attn_joint_type(
        #     f_1_2_local_lst[1][0].transpose(1, 2),  # [N, P, 2d+64]
        #     f_1_2_local_lst[1][1].transpose(1, 2)   # [N, P, 2d+64] 
        # )  # [N, P, 4d+2*64]

        # f12_local_motion = self.attn_motion_param(
        #     f_1_2_local_lst[2][0].transpose(1, 2),  # [N, P, 2d+64]
        #     f_1_2_local_lst[2][1].transpose(1, 2)   # [N, P, 2d+64] 
        # )  # [N, P, 4d+2*64]

        # -------- Motion prediction --------
        motion_para_out_r = self.motion_decoder_r(f12_motion_local) # [N, P, motion_decoder_out_dim]
        motion_para_out_p = self.motion_decoder_p(f12_motion_local) # [N, P, motion_decoder_out_dim]

        # -------- Connection prediction --------
        connection_decoder = self.connection_decoder(f12_kine_local) # [N, P, connection_decoder_out_dim]
        joint_type_decoder = self.joint_type_decoder(f12_kine_local) # [N, P, joint_type_decoder_out_dim]

        kine_gcn = self.gcn(f12_kine_global, adj).unsqueeze(1).repeat(1, P, 1)

        parts_conn_out = torch.cat([kine_gcn, connection_decoder], dim=2)  # [N, P, out_dim + connection_decoder_out_dim]
        joint_type_out = torch.cat([kine_gcn, joint_type_decoder], dim=2)  # [N, P, out_dim + joint_type_decoder_out_dim]
        return parts_conn_out, joint_type_out, motion_para_out_r, motion_para_out_p


class parts_connection_mlp(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.pointnetgcn = DualPointNetGCNII(
            kwargs["pointnet_dim"],
            kwargs["nlayers"],
            kwargs["nhidden"],
            kwargs["out_dim"],
            kwargs["dropout"],
            kwargs["lamda"],
            kwargs["alpha"],
            kwargs["variant"],
            kwargs["latent_dim"],
            kwargs["decoder_out_dim"],
            kwargs["motion_decoder_out_dim"],
        )

        # MLP for edge existence (binary-classification)
        self.part_conn_mlp = nn.Sequential(
            nn.Linear(2 * (kwargs["out_dim"]+kwargs["motion_decoder_out_dim"]), kwargs["nhidden_mlp"]),
            nn.Dropout(kwargs["dropout"]),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"], kwargs["nhidden_mlp"]//2),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//2, kwargs["nhidden_mlp"]//4),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//4, kwargs["nhidden_mlp"]//8),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//8, kwargs["n_class"]),
        )

        # MLP for joint type classification (binary-classification)
        self.joint_type_mlp = nn.Sequential(
            nn.Linear(2 * (kwargs["out_dim"]+kwargs["motion_decoder_out_dim"]), kwargs["nhidden_mlp"]),
            nn.Dropout(kwargs["dropout"]),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"], kwargs["nhidden_mlp"]//2),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//2, kwargs["nhidden_mlp"]//4),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//4, kwargs["nhidden_mlp"]//8),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//8, kwargs["n_class"]),
        )


        self.revolute_mlp = nn.Sequential(
            nn.Linear(kwargs["motion_decoder_out_dim"], kwargs["nhidden_mlp"]),
            nn.Dropout(kwargs["dropout"]),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"], kwargs["nhidden_mlp"]//2),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//2, kwargs["nhidden_mlp"]//4),
             nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//4, kwargs["nhidden_mlp"]//8),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//8, 8),
        )

        self.prismatic_mlp = nn.Sequential(
            nn.Linear(kwargs["motion_decoder_out_dim"], kwargs["nhidden_mlp"]),
            nn.Dropout(kwargs["dropout"]),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"], kwargs["nhidden_mlp"]//2),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//2, kwargs["nhidden_mlp"]//4),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//4, kwargs["nhidden_mlp"]//8),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"]//8, 4),
        )
        
    def forward(self, part_pointclouds_start, part_pointclouds_end, adj):
        # Node embeddings
        h_conn, h_joint_type, h_motion_r, h_motion_p = self.pointnetgcn(part_pointclouds_start, part_pointclouds_end, adj)  # [N, out_dim]

        N = adj.size(0)
        device = adj.device
        # Get indices of upper triangle (excluding diagonal)
        src, dst = torch.triu_indices(N, N, offset=1).to(device)

        # Filter only where there is an edge
        mask = adj[src, dst] != 0
        src, dst = src[mask], dst[mask]

        # create per-edge connection and motion features by concatenating node motion embeddings
        edge_conn_feats = torch.cat([h_conn[src], h_conn[dst]], dim=2)  # [num_edges, 2*(out_dim+connection_decoder_out_dim)]
        edge_joint_type_feats = torch.cat([h_joint_type[src], h_joint_type[dst]], dim=2)  # [num_edges, 2*(out_dim+joint_type_decoder_out_dim)]
        # edge_motion_feats = torch.cat([h_motion[src], h_motion[dst]], dim=2)  # [num_edges, P, 2*motion_decoder_out_dim]
        edge_motion_feats_r = torch.cat([h_motion_r[src], h_motion_r[dst]], dim=1)  # [num_edges, 2P, motion_decoder_out_dim]
        edge_motion_feats_p = torch.cat([h_motion_p[src], h_motion_p[dst]], dim=1)  # [num_edges, 2P, motion_decoder_out_dim]
        # Predict edge connection (binary)
        edge_pred = self.part_conn_mlp(edge_conn_feats)  # [num_edges, P, 1]
        # Predict joint type (binary)
        joint_type_pred = self.joint_type_mlp(edge_joint_type_feats)  # [num_edges, P, 1]
        # Predict motion parameters for revolute joints
        revolute_para_pred = self.revolute_mlp(edge_motion_feats_r) # [num_edges, P, 8]
        # Predict motion parameters for prismatic joints
        prismatic_para_pred = self.prismatic_mlp(edge_motion_feats_p) # [num_edges, P, 4]

        return edge_pred, joint_type_pred, revolute_para_pred, prismatic_para_pred, (src, dst)

    
    
if __name__ == '__main__':
    pass