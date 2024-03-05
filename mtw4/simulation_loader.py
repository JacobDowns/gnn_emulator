import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import firedrake as fd
from data_mapper import DataMapper
import torch
from torch_geometric.data import Data
from grad_solver import GradSolver
from torch.utils.data import Dataset
import numpy as np
from velocity_loss import LossIntegral
from multiprocessing import Pool
from torch_geometric.transforms import TwoHop

class SimulationLoader:

    def __init__(self, i):
        
        self.i = i
        base_dir = f'/media/new/gnn_augment_runs/{i}/'
        self.h5_input = f'{base_dir}output.h5'
        self.num_steps = 100

        with fd.CheckpointFile(self.h5_input, 'r') as afile:
            self.mesh = afile.load_mesh()
            self.grad_solver = GradSolver(self.mesh)
            self.data_mapper = DataMapper(self.mesh)
            self.B = afile.load_function(self.mesh, 'B')
            self.loss_integral = LossIntegral(self.mesh)

            self.coords = self.data_mapper.coords
            self.edges = self.data_mapper.edges

            # Input features
            self.Xs = []
            # Output features
            self.Ys = []
            # Graphs
            self.Gs = []
        

    def get_vars(self, j):
        d = {}
        d['B'] = self.B

        with fd.CheckpointFile(self.h5_input, 'r') as afile:
            d['H'] = afile.load_function(self.mesh, 'H0', idx=j)
            d['beta2'] = afile.load_function(self.mesh, 'beta2', idx=j)
            d['Ubar'] = afile.load_function(self.mesh, 'Ubar0', idx=j)
            d['Udef'] = afile.load_function(self.mesh, 'Udef0', idx=j)

            return d
        
    
    def load_features_from_h5(self):

        for j in range(self.num_steps):
            print(self.i, j)
            
            vars = self.get_vars(j)
    
            H_avg = self.data_mapper.get_avg(vars['H'])
            beta2_avg = self.data_mapper.get_avg(vars['beta2'])
            B_grad = self.grad_solver.solve_grad(vars['B'])
            S_grad = self.grad_solver.solve_grad(vars['B'] + vars['H'])
            B_grad = B_grad.dat.data
            S_grad = S_grad.dat.data

            # Geometric variables
            X_g = np.column_stack([
                self.data_mapper.edge_lens,
                self.data_mapper.dx0,
                self.data_mapper.dy0,
                self.data_mapper.dx1,
                self.data_mapper.dy1
            ])

            # Input variables
            X_i = np.column_stack([
                H_avg,
                B_grad,
                S_grad,
                beta2_avg
            ])

            X = np.column_stack([
                X_g,
                X_i
            ])

            # Outputs
            Ubar = vars['Ubar'].dat.data.reshape((-1,3))

            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Ubar, dtype=torch.float32)

            self.Xs.append(X)
            self.Ys.append(Y)


    def load_features_from_arrays(self):

        for j in range(self.num_steps):

            # Load outputs from arrays
            X = np.load(f'/media/new/gnn_augment_runs/{self.i}/X_{j}.npy')
            Y = np.load(f'/media/new/gnn_augment_runs/{self.i}/Y_{j}.npy')
            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Y, dtype=torch.float32)

            self.Xs.append(X)
            self.Ys.append(Y)

    
    def save_feature_arrays(self):
        base_dir = f'/media/new/gnn_augment_runs/{self.i}'
        for j in range(self.num_steps):
            x = self.Xs[j]
            y = self.Ys[j]
            np.save(f'{base_dir}/X_{j}.npy', x)
            np.save(f'{base_dir}/Y_{j}.npy', y)


class SimulatorDataset(Dataset):
     
    def __init__(self):

        self.sim_loaders = sim_loaders = []
        # Number of simulations
        self.num_simulations = 40
        # Number of timesteps per simulation
        self.num_steps = 100

        for i in range(self.num_simulations):
            
            print(f'Loading Datastet {i}')
            sim_loader = SimulationLoader(i)
            sim_loader.load_features_from_arrays()
            sim_loaders.append(sim_loader)

    def __len__(self):
        return self.num_simulations*self.num_steps
    
    def __getitem__(self, idx):
        sim_idx = int(idx / self.num_steps)
        time_idx = idx % self.num_steps
        #print(sim_idx, time_idx)

        sim_loader = self.sim_loaders[sim_idx]
        x = sim_loader.Xs[time_idx]
        y = sim_loader.Ys[time_idx]
        coords = torch.tensor(sim_loader.coords, dtype=torch.float32)
        edge_index = torch.tensor(sim_loader.edges, dtype=torch.int64)

        g = Data(
            pos = coords,
            edge_index = edge_index.T,
            x = x
        )

        return (g, y, sim_loader)

"""
def f(i):
    d = SimulationLoader(i)
    d.load_features_from_h5()
    d.save_feature_arrays()

with Pool(10) as p:
    p.map(f, list(range(40)))
"""