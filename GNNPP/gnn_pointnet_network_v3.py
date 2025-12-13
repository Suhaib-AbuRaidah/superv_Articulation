import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from GNNPP.gnn_PointNetEncoder import PointNetEncoder

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
    def __init__(self, pointnet_out_dim, nlayers, nhidden, out_dim,
                 dropout, lamda, alpha, variant, latent_dim, decoder_out_dim):
        super().__init__()

        # Two separate PointNet encoders
        self.pointnet_p1_enc = PointNetEncoder(global_feat = False, out_dim=pointnet_out_dim)
        self.pointnet_p2_enc = PointNetEncoder(global_feat = False, out_dim=pointnet_out_dim)
        self.pointnet_decoder = PointNetEncoder(global_feat = False, out_dim=pointnet_out_dim)

        # Attention: input d → output 2d
        self.attn_parts_conn = CrossAttention(pointnet_out_dim) # for part connection prediction
        self.attn_motion_param = CrossAttention(pointnet_out_dim+64) # for motion parameter prediction, 64 = local feature dim

        # Autoencoder: input 2d → latent_dim
        self.ae = FeatureAutoEncoder(2 * pointnet_out_dim, latent_dim=latent_dim,decoder_out_dim=decoder_out_dim)

        # Final GCN input = attn_out (2d) + latent (128)
        gcn_in_dim = 2 * pointnet_out_dim + decoder_out_dim

        self.gcn = GCNII(nfeat=gcn_in_dim, nlayers=nlayers,
                         nhidden=nhidden, out_dim=out_dim,
                         dropout=dropout, lamda=lamda,
                         alpha=alpha, variant=variant)

    def forward(self, part_pcs1, part_pcs2, adj):
        # 1. Extract PointNet features for pose1 & pose2
        pcs1 = torch.concat([pc.transpose(1,2) for pc in part_pcs1], 0)
        pcs2 = torch.concat([pc.transpose(1,2) for pc in part_pcs2], 0)

        f1_global,f1_local,_,_ = self.pointnet_p1_enc(pcs1)  # [N, d]
        f2_global,f2_local,_,_ = self.pointnet_p2_enc(pcs2)  # [N, d]

        # 2. Cross Attention (fusion)
        f12_global = self.attn_parts_conn(f1_global, f2_global)        # [N, 2d]
        f12_local = self.attn_motion_param(f1_local.transpose(1,2), f2_local.transpose(1,2))  # [N, 2(d+64)]
        # 3. Autoencoder
        z, recon = self.ae(f12_global)        # recon: [N,128]
        # 4. Concatenate fused features with latent
        h = torch.cat([f12_global, recon], dim=1) # [N, 2d + 128]
        _,motion_para_out,_,_ = self.pointnet_decoder(f12_local)
        # 5. GCN
        parts_conn_out = self.gcn(h, adj)

        return parts_conn_out, motion_para_out, z, recon


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
        )

        # MLP for edge existence (binary-classification)
        self.part_conn_mlp = nn.Sequential(
            nn.Linear(2 * kwargs["out_dim"], kwargs["nhidden_mlp"]),
            nn.Dropout(kwargs["dropout"]),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"], kwargs["n_class"]), #{connection, no-connection}
        )

        # MLP for joint type classification (binary-classification)
        self.joint_type_mlp = nn.Sequential(
            nn.Linear(2 * kwargs["out_dim"], kwargs["nhidden_mlp"]),
            nn.Dropout(kwargs["dropout"]),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"], 1),  # {revolute, prismatic}
        )

        self.revolute_mlp = nn.Sequential(
            nn.Linear(2 * kwargs["out_dim"], kwargs["nhidden_mlp"]),
            nn.Dropout(kwargs["dropout"]),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"], 4),
        )

        self.prismatic_mlp = nn.Sequential(
            nn.Linear(2 * kwargs["out_dim"], kwargs["nhidden_mlp"]),
            nn.Dropout(kwargs["dropout"]),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"], 4),
        )

                      
    def forward(self, part_pointclouds_start, part_pointclouds_end, adj):
        # Node embeddings
        h_conn,h_motion,z,_ = self.pointnetgcn(part_pointclouds_start, part_pointclouds_end, adj)  # [N, out_dim]

        N = adj.size(0)
        # Get indices of upper triangle (excluding diagonal)
        src, dst = torch.triu_indices(N, N, offset=1)
        
        # Filter only where there is an edge
        mask = adj[src, dst] != 0
        src, dst = src[mask], dst[mask]

        # create per-edge connection and motion features by concatenating node motion embeddings
        edge_conn_feats = torch.cat([h_conn[src], h_conn[dst]], dim=1)  # [num_edges, 2*out_dim]        
        edge_motion_feats = torch.cat([h_motion[src], h_motion[dst]], dim=1)  # [num_edges, N, 2*out_dim]

        # Predict edge connection (binary)
        edge_pred = self.part_conn_mlp(edge_conn_feats)  # [num_edges, 1]

        # Predict joint type (multi-class)
        joint_type_pred = self.joint_type_mlp(edge_conn_feats)  # [num_edges, 1]

        revolute_para_pred = self.revolute_mlp(edge_motion_feats)

        prismatic_para_pred = self.prismatic_mlp(edge_motion_feats)

        return edge_pred, joint_type_pred, revolute_para_pred, prismatic_para_pred, z, (src, dst)

    
    
if __name__ == '__main__':
    pass