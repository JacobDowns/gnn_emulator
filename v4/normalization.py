import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from data_loader import GraphDataLoader

d = GraphDataLoader()
N = 0
X_stds = []
Y_stds = []
for g in d.training_data:
    print(g)

    coords = g.pos
    x = coords[:,0]
    y = coords[:,1]
    n_edge = int(g.edge_index.shape[1] / 2)

    H = g.edge_attr[:,2]
    indexes = H > 1.0001e-3

    X = g.edge_attr[indexes,:]
    Y = g.y[indexes]

    n = len(X)
    X_stds.append(X.std(axis=0).numpy())
    Y_stds.append(float(Y.std()))
    N += n

X_stds = np.array(X_stds)
Y_stds = np.array(Y_stds)
X_std = np.mean(X_stds, axis=0)
Y_std = np.mean(Y_stds)
print(X_std)
print(Y_std) 


