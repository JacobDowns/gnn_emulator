import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from data_loader import GraphDataLoader
import meshio

"""
This script is just for verifying that the training data makes sense.
"""


d = GraphDataLoader()

n_edges = []
n_nodes = []
edge_stds = []
node_stds = []
y_stds = []

for g in d.data:
    print(g)

    n_edges.append(g.edge_attr.shape[0])
    n_nodes.append(g.x.shape[0])

    edge_stds.append(g.edge_attr.std(axis=0))
    node_stds.append(g.x.std(axis=0))
    y_stds.append(g.y_edge.std(axis=0))

node_stds = np.array(node_stds)
edge_stds = np.array(edge_stds)
y_stds = np.array(y_stds)


n_nodes = np.array(n_nodes, dtype=np.float32)
n_edges = np.array(n_edges, dtype=np.float32)

n_nodes /= n_nodes.sum()
n_edges /= n_edges.sum()

node_stds.shape
node_stds *= n_nodes[:,np.newaxis]
edge_stds *= n_edges[:,np.newaxis]
y_stds *= n_edges[:,np.newaxis]

node_std = np.sum(node_stds, axis=0)
edge_std = np.sum(edge_stds, axis=0)
y_std = np.sum(y_stds, axis=0)

print()
print(node_std)
print(edge_std)
print(y_std)
quit()

data = d.data[30]

cell_coords = data.pos
cell_connections = data.edge_index.T
X_cell = data.x
X_edge = data.edge_attr
Y_edge = data.y_edge

n = int(len(cell_connections) / 2)
X_edge = X_edge[0:n,:]
Y_edge = Y_edge[0:n,:]

print(X_edge.shape)

H = X_cell[:,0]
beta2 = X_cell[:,1]
N = X_cell[:,2]

lines = [
    ('line', cell_connections[0:n])
]

mesh = meshio.Mesh(
    cell_coords,
    lines,
    point_data={
        'H' : H,
        'beta2' : beta2,
        'N' : N
    }, 
    cell_data={
        'dB' : [abs(X_edge[:,-2])],
        'dS' : [abs(X_edge[:,-1])],
        'y' : Y_edge
    }

)

mesh.write(
    "test.vtk",  # str, os.PathLike, or buffer/open file
    # file_format="vtk",  # optional if first argument is a path; inferred from extension
)