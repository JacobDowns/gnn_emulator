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

class SimulationLoader:

    def __init__(self, i):
        
        self.i = i
        base_dir = f'/media/new/gnn_emulator_runs1/{i}/'
        self.h5_input = f'{base_dir}output.h5'
        self.num_steps = 100

        with fd.CheckpointFile(self.h5_input, 'r') as afile:
            self.mesh = afile.load_mesh()
            self.grad_solver = GradSolver(self.mesh)
            self.data_mapper = DataMapper(self.mesh)
            self.B = afile.load_function(self.mesh, 'B')
            self.loss_integral = LossIntegral(self.mesh)

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

            # Geometric variables
            X_g = np.column_stack([
                self.data_mapper.edge_lens,
                self.data_mapper.dx0,
                self.data_mapper.dy0,
                self.data_mapper.dx1,
                self.data_mapper.dy1
            ])

            B_jump = self.data_mapper.get_jump(vars['B'])
            H_jump = self.data_mapper.get_jump(vars['H'])


            # Input variables
            X_i = np.column_stack([
                H_avg,
                B_jump,
                H_jump,
                beta2_avg
            ])

            X = np.column_stack([
                X_i,
                X_g
            ])

            # Outputs
            Ubar = vars['Ubar'].dat.data.reshape((-1,3))
            Udef = vars['Udef'].dat.data.reshape((-1,3))

            Y = np.column_stack([
                Ubar,
                Udef
            ])

            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Y, dtype=torch.float32)

            self.Xs.append(X)
            self.Ys.append(Y)


    def load_features_from_arrays(self):

        for j in range(self.num_steps):

            # Load outputs from arrays
            X = np.load(f'/media/new/gnn_emulator_runs1/{self.i}/X.npy')
            Y = np.load(f'/media/new/gnn_emulator_runs1/{self.i}/Y.npy')
            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Y, dtype=torch.float32)

            self.Xs.append(X)
            self.Ys.append(Y)


    """
    Creates graphs from features. 
    """
    def create_graphs(self):

        coords = self.data_mapper.coords
        edges = self.data_mapper.edges

        coords = torch.tensor(coords, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.int64)

        for j in range(self.num_steps):
            g = Data(pos = coords, edge_index = edge_index, x = self.Xs[j])
            self.Gs.append(g)
        

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
            sim_loader.create_graphs()
            sim_loaders.append(sim_loader)

    def __len__(self):
        return self.num_simulations*self.num_steps
    
    def __getitem__(self, idx):
        sim_idx = int(idx / self.num_steps)
        time_idx = idx % self.num_steps
        #print(sim_idx, time_idx)

        sim_loader = self.sim_loaders[sim_idx]

        g = sim_loader.Gs[time_idx]
        y = sim_loader.Ys[time_idx]
        return (g, y, sim_loader)