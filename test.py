import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# adj = torch.tensor([[0, 1, 1,1],
#                     [1, 0, 1,0],
#                     [1, 1, 0,0],
#                     [1, 0, 0,0]])

# N = adj.size(0)
# src, dst = torch.triu_indices(N,N, offset=1)
# mask = adj[src, dst]!=0

# print(f"Adj: \n{adj}\n")
# print(f"Src: \n{src}\n")
# print(f"Dst: \n{dst}\n")
# print(f"Mask: \n{mask}\n")
torch.manual_seed(42)
x = torch.randint(0, 5, (5,3)).float()
y = torch.randint(0, 5, (5,3)).float()
print(f"x: \n{x}\n")
print(f"y: \n{y}\n")
loss = F.mse_loss(x, y, reduction='none')
print(f"Loss before reduction: \n{loss}\n")
print(f"Loss after mean reduction: \n{loss.mean(dim=1).mean()}\n")