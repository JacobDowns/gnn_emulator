from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv
import torch
import firedrake as fd
import os
os.environ['OMP_NUM_THREADS'] = '1'
from data_mapper import DataMapper
import numpy as np

mesh = fd.RectangleMesh(10, 10, 1., 1.)


d = DataMapper(mesh)

coords = d.coords
xs = coords[:,0]
ys = coords[:,1]
edges = np.array(d.edges, dtype=np.int64)
edges = torch.tensor(edges).T
f = fd.Function(d.V_cg)
F = torch.rand(len(f.dat.data), 4, requires_grad=True)



in_channels = 4
out_channels = 2

model = Sequential('x, edge_index', [
    (GCNConv(in_channels, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (GCNConv(64, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    Linear(64, out_channels),
])

y = model(F, edges)
grad_output = torch.ones_like(y)
y.backward(grad_output)

gradients = F.grad
print(gradients)