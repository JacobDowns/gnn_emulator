import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from model.simulator import Simulator
import firedrake as fd
from grad_solver import GradSolver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmulatorModel:
    def __init__(self, data_mapper):

        # Mesh data
        self.data_mapper = data_mapper
        self.grad_solver = GradSolver(data_mapper.mesh)
             
        # Edge offsets and lengths
        self.edges = self.data_mapper.edges
        self.coords = self.data_mapper.coords

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

        # Function spaces
        self.V_rt = data_mapper.V_rt
        self.V_dg = data_mapper.V_dg
        self.V_cr = data_mapper.V_cr
        
        # GNN model 
        simulator = Simulator(message_passing_num=24, edge_input_size=10, device=device)
        simulator.load_checkpoint()
        self.simulator = simulator
        simulator.eval()

        # Graph data
        graph = Data(
            edge_index = torch.tensor(self.edges, dtype=torch.int64),
            pos = torch.tensor(self.coords, dtype=torch.float32),
            x_g = torch.tensor(X_g, dtype=torch.float32).T
        )

        x_g_std = torch.tensor([
            0.0443,
            0.0017,
            0.0017,
            1.92,
            1.92,
            1.92,
            1.92
        ])

        graph.x_g /= x_g_std[np.newaxis,:]
        self.graph = graph.cuda()

      
    def solve(self, B_dg, H_dg, beta2_cg):
        
         with torch.no_grad():

            # Create input tensor
            x_i = self.__get_inputs__(B_dg, H_dg, beta2_cg)
            x_i = torch.tensor(x_i, dtype=torch.float32).cuda()
            self.graph.x_i = x_i
            # Estimate velocity
            out = self.simulator(self.graph)
            # Return RT coefficients
            out = out.cpu().numpy()*3.7353

            return out

        
    def __get_inputs__(self, B_dg, H_dg, beta2_cg):
        """
        Prep inputs for GNN model. 
        """
       
        grad_H = np.copy(self.grad_solver.solve_grad(H_dg).dat.data)
        grad_B = np.copy(self.grad_solver.solve_grad(B_dg).dat.data)
        H_avg = self.data_mapper.get_avg(H_dg)
        beta2_avg = self.data_mapper.get_avg(beta2_cg)

        # Input variables
        X_i = np.column_stack([
            H_avg,
            grad_B + grad_H,
            H_avg * beta2_avg,
        ])

        x_i_std = np.array([
            81.189,
            33.786,
            1.1434e5
        ])

    
        X_i /= x_i_std[np.newaxis,:]
        return X_i
