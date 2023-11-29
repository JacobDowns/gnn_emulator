import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import firedrake as fd
from data_mapper import DataMapper
import torch
from torch_geometric.data import Data
from grad_solver import GradSolver


class SimulationLoader:

    def __init__(self, file_name):

        self.file_name = file_name
        

        with fd.CheckpointFile(file_name, 'r') as afile:
            self.mesh = afile.load_mesh()
            self.grad_solver = GradSolver(self.mesh)
            self.data_mapper = DataMapper(self.mesh)

            self.B = afile.load_function(self.mesh, 'B')
            self.grad_B = np.copy(self.grad_solver.solve_grad(self.B).dat.data)
    
    
    def get_step(self, j):
        d = {}
        with fd.CheckpointFile(self.file_name, 'r') as afile:
            d['H_dg'] = afile.load_function(self.mesh, 'H0', idx=j)
            d['beta2_cg'] = afile.load_function(self.mesh, 'beta2', idx=j)
            d['ubar_rt'] = afile.load_function(self.mesh, 'Ubar0', idx=j)

            return d


    # Compile graph features
    def get_graph(self, j):

        # Edge features
        d = self.get_step(j)

        H_avg = self.data_mapper.get_avg(d['H_dg'])
        beta2_avg = self.data_mapper.get_avg(self.B + d['beta2_cg'])

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

        Y = d['ubar_rt'].dat.data

        data = Data(
            y = torch.tensor(Y, dtype=torch.float32),
            edge_index = torch.tensor(edges, dtype=torch.int64),
            x_g = torch.tensor(X_g, dtype=torch.float32).T,
            x_i = torch.tensor(X_i, dtype=torch.float32).T,
            pos = torch.tensor(coords, dtype=torch.float32)
        )

        return data
    

    # Load graphs for each model time step
    def get_graphs(self):
        data = []
        for j in range(140):
            g = self.get_graph(j)
            data.append(g)

        return data