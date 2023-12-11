import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import firedrake as fd
from data_mapper import DataMapper
import torch
from torch_geometric.data import Data
from grad_solver import GradSolver
from torch.utils.data import Dataset, DataLoader

class SimulationLoader:

    def __init__(self, file_name):

        self.file_name = file_name
        
        with fd.CheckpointFile(file_name, 'r') as afile:
            self.mesh = afile.load_mesh()
            self.grad_solver = GradSolver(self.mesh)
            self.data_mapper = DataMapper(self.mesh)

            self.B = afile.load_function(self.mesh, 'B')
            self.grad_B = np.copy(self.grad_solver.solve_grad(self.B).dat.data)
    
    
    def get_vars(self, j):
        d = {}
        with fd.CheckpointFile(self.file_name, 'r') as afile:
            d['B'] = self.B
            d['grad_B'] = self.grad_B
            d['H'] = afile.load_function(self.mesh, 'H0', idx=j)
            d['beta2'] = afile.load_function(self.mesh, 'beta2', idx=j)
            d['Ubar'] = afile.load_function(self.mesh, 'Ubar0', idx=j)

            return d


    # Compile graph features
    def get_graph(self, vars):

        # Edge features
        d = self.get_step(j)

        H_avg = self.data_mapper.get_avg(vars['H'])
        beta2_avg = self.data_mapper.get_avg(vars['beta2'])

        edges = self.data_mapper.edges
        coords = self.data_mapper.coords

        # Geometric variables
        X_g = np.stack([
            self.data_mapper.edge_lens,
            self.data_mapper.a0,
            self.data_mapper.a1, 
            self.data_mapper.v0[:,0],
            self.data_mapper.v0[:,1],
            self.data_mapper.v1[:,0],
            self.data_mapper.v1[:,1]
        ])

        grad_H = np.copy(self.grad_solver.solve_grad(d['H_dg']).dat.data)

        # Input variables
        X_i = np.stack([
            H_avg,
            self.grad_B + grad_H,
            H_avg * beta2_avg,
        ])

        Y = vars['ubar_rt'].dat.data

        data = Data(
            y = torch.tensor(Y, dtype=torch.float32),
            edge_index = torch.tensor(edges, dtype=torch.int64),
            x_g = torch.tensor(X_g, dtype=torch.float32).T,
            x_i = torch.tensor(X_i, dtype=torch.float32).T,
            pos = torch.tensor(coords, dtype=torch.float32)
        )

        return data    

class SimulatorDataset(Dataset):
     
    def __init__(self):

        self.sim_loaders = sim_loaders = []
        for i in range(40):
            print(f'Loading Datastet {i}')
            file_name = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/results/{i}/output.h5'
            sim_loaders.append(SimulationLoader(file_name))

    def __len__(self):
        return 40*140
    
    def __getitem__(self, idx):
        sim_idx = int(idx / 140)
        time_idx = idx % 140
        print(sim_idx, time_idx)

        sim_loader = self.sim_loaders[sim_idx]
        data = sim_loader.get_step(time_idx)
        return data

d = SimulatorDataset()
d.__getitem__(10)
print(d)