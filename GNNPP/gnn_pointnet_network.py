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
    

class PointNetGCNII(nn.Module):
    def __init__(self, pointnet_out_dim, nlayers, nhidden, out_dim, dropout, lamda, alpha, variant):
        super().__init__()
        self.pointnet = PointNetEncoder(out_dim=pointnet_out_dim)
        self.gcn = GCNII(nfeat=pointnet_out_dim, nlayers=nlayers,
                         nhidden=nhidden, out_dim=out_dim,
                         dropout=dropout, lamda=lamda,
                         alpha=alpha, variant=variant)

    def forward(self, part_pointclouds, adj):
        """
        part_pointclouds: list of length N, each [P_i, 3]
        adj: [N, N]
        """
        # Stack all parts as a batch
        pcs = torch.concat([pc.transpose(1,2) for pc in part_pointclouds], dim=0) 

        # Masking for variable lengths can be ignored if PointNet does global pooling
        feats,_,_ = self.pointnet(pcs)  # [N, d]  (batch = num_parts)
        # Use the embeddings as node features for GCN
        return self.gcn(feats, adj)


    

class parts_connection_mlp(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.pointnetgcn = PointNetGCNII(
            kwargs["pointnet_dim"],
            kwargs["nlayers"],
            kwargs["nhidden"],
            kwargs["out_dim"],
            kwargs["dropout"],
            kwargs["lamda"],
            kwargs["alpha"],
            kwargs["variant"],
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

        # MLP for parent part classification (binary-classification)
        self.joint_type_mlp = nn.Sequential(
            nn.Linear(2 * kwargs["out_dim"], kwargs["nhidden_mlp"]),
            nn.Dropout(kwargs["dropout"]),
            nn.ReLU(),
            nn.Linear(kwargs["nhidden_mlp"], 1),  # {parent, child}
        )

    def forward(self, part_pointclouds, adj):
        # Node embeddings
        h = self.pointnetgcn(part_pointclouds, adj)  # [N, out_dim]

        N = adj.size(0)
        # Get indices of upper triangle (excluding diagonal)
        src, dst = torch.triu_indices(N, N, offset=1)
        
        # Filter only where there is an edge
        mask = adj[src, dst] != 0
        src, dst = src[mask], dst[mask]

        # Concatenate embeddings
        edge_feats = torch.cat([h[src], h[dst]], dim=1)  # [num_edges, 2*out_dim]

        # Predict edge connection (binary)
        edge_pred = self.part_conn_mlp(edge_feats)  # [num_edges, 1]

        # Predict joint type (multi-class)
        joint_type_pred = self.joint_type_mlp(edge_feats)  # [num_edges, 1]

        return edge_pred, joint_type_pred, (src, dst)

    
    
if __name__ == '__main__':
    pass