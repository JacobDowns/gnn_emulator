import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from graph_data_loader import GraphDataLoader

d = GraphDataLoader()
N = 0
x_g_stds = []
x_i_stds = []
y_stds = []

for g in d.training_data:
    print(g)

    x_g = g.x_g
    x_i = g.x_i
    y = g.y

    n = len(x_g)

    x_g_stds.append(x_g.std(axis=0).numpy())
    x_i_stds.append(x_i.std(axis=0).numpy())
    y_stds.append(float(y.std()))
    N += n

x_g_stds = np.array(x_g_stds)
x_i_stds = np.array(x_i_stds)
y_stds = np.array(y_stds)
#print(x_g_stds)
#print(x_i_stds)
#print(y_stds)


x_g_std = np.mean(x_g_stds, axis=0)
x_i_std = np.mean(x_i_stds, axis=0)
y_std = np.mean(y_stds)
print(x_g_std)
print(x_i_std) 
print(y_std)


