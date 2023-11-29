import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from data_loader import GraphDataLoader

"""
Check how well the various features are normalized.
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

    y_stds.append(g.y.std(axis=0))

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
y_stds *= n_edges

node_std = np.sum(node_stds, axis=0)
edge_std = np.sum(edge_stds, axis=0)
y_std = np.sum(y_stds, axis=0)

print()
print(node_std)
print(edge_std)
print(y_std)