import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from firedrake_data_loader import FDDataLoader

for i in range(40):
    file_name = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/results/{i}/output.h5'
    fdd_loader = FDDataLoader(file_name)
    graphs = fdd_loader.get_graphs()

    j = 0
    for g in graphs:
        print(i, j)
        print(g)

        x = g.pos[:,0]
        y = g.pos[:,1]

        n_edge = int(g.edge_index.shape[1] / 2)
        edges = g.edge_index[:,0:n_edge].T

        mx = x[edges]
        my = y[edges]
        mx = mx.mean(axis=1)
        my = my.mean(axis=1)

        #print(g.y[0:n_edge].max())

        """
        plt.subplot(2,1,1)
        plt.scatter(mx, my, c=g.edge_attr[0:n_edge,5])
        plt.colorbar()

        plt.subplot(2,1,2)
        plt.scatter(mx, my, c=g.y[0:n_edge])
        plt.colorbar()
        plt.show()
        """

        torch.save(g, f'data/g_{i}_{j}.pt')
        j += 1
