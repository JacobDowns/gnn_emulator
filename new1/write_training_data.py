import torch
import numpy as np
from simulation_loader import SimulationLoader

for i in range(40):
    file_name = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/results/{i}/output.h5'
    loader = SimulationLoader(file_name)
    graphs = loader.get_graphs()

    j = 0
    for g in graphs:
        print(i, j)
        print(g)

        torch.save(g, f'data/g_{i}_{j}.pt')
        j += 1