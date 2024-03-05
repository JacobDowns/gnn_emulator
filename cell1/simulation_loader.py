import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import firedrake as fd
from mesh_mapper import MeshMapper
import torch
from torch_geometric.data import Data
from grad_solver import GradSolver
from torch.utils.data import Dataset
import numpy as np
from velocity_loss import LossIntegral


class SimulationLoader:

    def __init__(self, i):
        
        self.i = i
        base_dir = f'/media/new/gnn_augment_runs/{i}/'
        self.h5_input = f'{base_dir}output.h5'
        self.num_steps = 100

        with fd.CheckpointFile(self.h5_input, 'r') as afile:
            self.mesh = afile.load_mesh()
            self.grad_solver = GradSolver(self.mesh)
            self.mapper = MeshMapper(self.mesh)
            self.B = afile.load_function(self.mesh, 'B')
            self.loss_integral = LossIntegral(self.mesh)

            # Input features
            self.X_cell = []
            self.X_edge = []
            # Output features
            self.Y = []
        

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

            ### Cell feature array
            local_cell_xs = self.mapper.local_cell_xs
            local_cell_ys = self.mapper.local_cell_ys
            faces = self.mapper.faces
            beta2_vals = vars['beta2'].dat.data[faces]

            cell_features = np.column_stack([
                local_cell_xs,
                local_cell_ys,
                vars['H'].dat.data,
                beta2_vals
            ])

            ### Edge feature array
            jump_S = self.mapper.get_jump(vars['B'] + vars['H'])
            jump_B = self.mapper.get_jump(vars['B'])
            edge_features = np.column_stack([
                jump_S,
                jump_B
            ])

            ### Edge output array
            Ubar = vars['Ubar'].dat.data

            x_cell = torch.tensor(cell_features, dtype=torch.float32)
            x_edge = torch.tensor(edge_features, dtype=torch.float32)
            y_edge = torch.tensor(Ubar, dtype=torch.float32)

            self.X_cell.append(x_cell)
            self.X_edge.append(x_edge)
            self.Y.append(y_edge)


    def load_features_from_arrays(self):

        for j in range(self.num_steps):

            # Load outputs from arrays
            x_cell = np.load(f'/media/new/gnn_augment_runs/{self.i}/x_cell_{j}.npy')
            x_edge = np.load(f'/media/new/gnn_augment_runs/{self.i}/x_edge_{j}.npy')
            y_edge = np.load(f'/media/new/gnn_augment_runs/{self.i}/y_edge_{j}.npy')
            
            x_edge = torch.tensor(x_edge, dtype=torch.float32)
            x_cell = torch.tensor(x_cell, dtype=torch.float32)
            y_edge = torch.tensor(y_edge, dtype=torch.float32)

            self.X_cell.append(x_cell)
            self.X_edge.append(x_edge)
            self.Y.append(y_edge)

    
    def save_feature_arrays(self):
        base_dir = f'/media/new/gnn_augment_runs/{self.i}'
        for j in range(self.num_steps):
            x_cell = self.X_cell[j]
            x_edge = self.X_edge[j]
            y = self.Y[j]

            np.save(f'{base_dir}/x_cell_{j}.npy', x_cell)
            np.save(f'{base_dir}/x_edge_{j}.npy', x_edge)
            np.save(f'{base_dir}/y_edge_{j}.npy', y)


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
        x_cell = sim_loader.X_cell[time_idx]
        x_edge = sim_loader.X_edge[time_idx]
        y_edge = sim_loader.Y[time_idx]
        
        coords = torch.tensor(sim_loader.mapper.coords, dtype=torch.float32)
        edge_index = torch.tensor(sim_loader.mapper.edges, dtype=torch.int64)

        g = Data(
            pos = coords,
            edge_index = edge_index,
            x_node = x_cell,
            x_edge = x_edge
        )

        return (g, y_edge, sim_loader)