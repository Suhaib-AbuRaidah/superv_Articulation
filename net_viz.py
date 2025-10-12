from torchviz import make_dot
import torch
import sys
import torch
from torch import nn
import torch.functional as f
from torch.utils.tensorboard import SummaryWriter
from torchview import draw_graph
sys.path.append('/home/suhaib/superv_Articulation')
from GNNPP.gnn_pointnet_network import GraphConvolution

class experimentnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(16, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.linear(x)

# Suppose `model` is your GraphConvolution instance
# and you have dummy input tensors:
# gc = GraphConvolution(in_features=16, out_features=16)
# x = torch.randn(10, 16)
# adj = torch.eye(10)
# h0 = torch.randn(10, 16)
# lamda, alpha, l = 0.5, 0.1, 1

# out = gc(x, adj, h0, lamda, alpha, l)
# dot = make_dot(out, params=dict(list(gc.named_parameters())), show_attrs=True, show_saved=True)
# dot.render("graph_convolution", format="svg",)




gc = GraphConvolution(in_features=16, out_features=16)
# gc = experimentnn()
x = torch.randn(10, 16)
adj = torch.eye(10)
h0 = torch.randn(10, 16)
lamda, alpha, l = 0.5, 0.1, 1
# graph = draw_graph(gc, input_data=(x), expand_nested=True,hide_module_functions= False, hide_inner_tensors= False, roll= True)

graph = draw_graph(gc, input_data=(x, adj, h0, lamda, alpha, l), expand_nested=True,hide_module_functions= False, hide_inner_tensors= False, roll= True)
graph.visual_graph.render("graph_convolution_architecture", format="svg",)



# gc = GraphConvolution(in_features=16, out_features=16)
# # gc = experimentnn()
# writer = SummaryWriter("runs/gc_demo")

# x = torch.randn(10, 16)
# adj = torch.eye(10)
# h0 = torch.randn(10, 16)

# # Convert scalars to tensors
# lamda = torch.tensor([0.5])
# alpha = torch.tensor([0.1])
# l = torch.tensor([1.0])
# # writer.add_graph(gc, (x))
# writer.add_graph(gc, (x, adj, h0, lamda, alpha, l))
# writer.close()