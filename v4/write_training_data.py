import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from firedrake_data_loader import FDDataLoader

for i in range(25):
    file_name = f'/home/jake/ManuscriptCode/examples/training_runs1/results/{i}/output.h5'
    fdd_loader = FDDataLoader(file_name)
    graphs = fdd_loader.get_graphs()

    j = 0
    for g in graphs:
        print(i, j)
        torch.save(g, f'data/g_{i}_{j}.pt')
        j += 1
