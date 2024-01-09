import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import firedrake as fd
from data_mapper import DataMapper
import torch
from torch_geometric.data import Data
from grad_solver import GradSolver
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import numpy as np
from velocity_loss import LossIntegral

class SimulationLoader:

    def __init__(self, file_name):

        self.file_name = file_name
        self.vars = []

        with fd.CheckpointFile(file_name, 'r') as afile:
            self.mesh = afile.load_mesh()
            self.grad_solver = GradSolver(self.mesh)
            self.data_mapper = DataMapper(self.mesh)

            print(self.data_mapper.edges.max())

            self.B = afile.load_function(self.mesh, 'B')
            self.loss_integral = LossIntegral(self.mesh)

        for i in range(140):
            v = self.__get_vars__(i)
            self.vars.append(v)
    
    
    def __get_vars__(self, j):
        d = {}
        d['B'] = self.B

        with fd.CheckpointFile(self.file_name, 'r') as afile:
            d['H'] = afile.load_function(self.mesh, 'H0', idx=j)
            d['beta2'] = afile.load_function(self.mesh, 'beta2', idx=j)
            d['Ubar'] = afile.load_function(self.mesh, 'Ubar0', idx=j)

            return d


    # Compile graph features
    def get_graph(self, vars):

        x_g_std = np.array([
            1e-1,
            1e-3,
            1e-3,
            1e0,
            1e0,
            1e0,
            1e0
        ])

        x_i_std = np.array([
            1e1,
            1e1,
            1e5
        ])

        y_std = 1e0

        H_avg = self.data_mapper.get_avg(vars['H'])
        beta2_avg = self.data_mapper.get_avg(vars['beta2'])

        edges = self.data_mapper.edges
        coords = self.data_mapper.coords

        # Geometric variables
        X_g = np.column_stack([
            self.data_mapper.edge_lens,
            self.data_mapper.a0,
            self.data_mapper.a1, 
            self.data_mapper.v0[:,0],
            self.data_mapper.v0[:,1],
            self.data_mapper.v1[:,0],
            self.data_mapper.v1[:,1]
        ])

        grad_H = np.copy(self.grad_solver.solve_grad(vars['H']).dat.data)
        grad_B = np.copy(self.grad_solver.solve_grad(vars['B']).dat.data)

        # Input variables
        X_i = np.column_stack([
            H_avg,
            grad_B + grad_H,
            H_avg * beta2_avg,
        ])

        Y = vars['Ubar'].dat.data

        # Normalize
        X_g /= x_g_std[np.newaxis,:]
        X_i /= x_i_std[np.newaxis,:]
        Y /= y_std

        data = Data(
            y = torch.tensor(Y, dtype=torch.float32),
            edge_index = torch.tensor(edges, dtype=torch.int64),
            x_g = torch.tensor(X_g, dtype=torch.float32),
            x_i = torch.tensor(X_i, dtype=torch.float32),
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
        return 10*140
    
    def __getitem__(self, idx):
        sim_idx = int(idx / 140)
        time_idx = idx % 140
        #print(sim_idx, time_idx)

        sim_loader = self.sim_loaders[sim_idx]
        data = sim_loader.vars[time_idx]
        return (data, sim_loader)
    
"""
n_train = 140*35
n = len(d)

train_data = Subset(d, np.arange(n_train))
test_data = Subset(d, np.arange(n_train, n))

train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle = True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle = True)



"""