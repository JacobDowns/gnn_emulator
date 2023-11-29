import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
from data_mapper import DataMapper
import torch
from torch_geometric.data import Data
from torch_scatter import scatter_sum, scatter_mean

class FDDataLoader:

    def __init__(self, file_name):

        self.file_name = file_name
        with fd.CheckpointFile(file_name, 'r') as afile:

            # Mesh topology
            self.mesh = afile.load_mesh()
            d = DataMapper(self.mesh)
            self.coords = d.get_coords()
            self.edges = d.get_edges()
            self.normals = d.get_normals()
            self.d = d

            # Edge offsets and lengths
            dx = self.coords[:,0][self.edges]
            dy = self.coords[:,1][self.edges]
            self.dx = dx[:,1] - dx[:,0]
            self.dy = dy[:,1] - dy[:,0]
            self.dd = np.sqrt(self.dx**2 + self.dy**2)
            
            # Function spaces
            self.V_rt = d.V_rt
            self.V_dg = d.V_dg
            self.V_cr = d.V_cr

            # Functions for storing emulator input / output data
            self.u_rt = fd.Function(self.V_rt)
            self.B_dg = fd.Function(self.V_dg)
            self.H_dg = fd.Function(self.V_dg)
            self.S_dg = fd.Function(self.V_dg)
            self.beta2_dg = fd.Function(self.V_dg)
            self.beta2_cr = fd.Function(self.V_cr)
            self.u_rt = fd.Function(self.V_rt)
            self.B_dg.interpolate(afile.load_function(self.mesh, 'B'))


    # Load model data from h5 file
    def load_step(self, j):

        with fd.CheckpointFile(self.file_name, 'r') as afile:

            self.H_dg.interpolate(afile.load_function(self.mesh, 'H0', idx=j))
            self.S_dg.interpolate(self.B_dg + self.H_dg)
            self.beta2_dg.interpolate(afile.load_function(self.mesh, 'beta2', idx=j))
            jump_S_cr = self.d.get_jump(self.S_dg)
            jump_H_cr = self.d.get_jump(self.H_dg)
            H_cr = fd.project(self.H_dg, self.V_cr)
            beta2_cr = fd.project(self.beta2_dg, self.V_cr)
            u = afile.load_function(self.mesh, 'U', idx=j)
            self.u_rt.interpolate(fd.project(u, self.V_rt))

            d = {}
            d['jump_S_cr'] = jump_S_cr
            d['jump_H_cr'] = jump_H_cr
            d['H_cr'] = H_cr.dat.data[:]
            d['beta2_cr'] = beta2_cr.dat.data[:]
            d['u_rt'] = self.u_rt.dat.data[:] / self.dd

            return d

    # Get mesh data for a given timestep
    def get_graph(self, j, normalize=False):

        data = self.load_step(j)

        # Edge input features
        input_features = np.stack([
            data['jump_S_cr'],
            data['jump_H_cr'],
            data['H_cr'],
            data['beta2_cr']*data['H_cr'],
            self.normals[:,0],
            self.normals[:,1],
            self.dd
        ]).T

        # Edge output features
        output_features = data['u_rt']

        """
        Edge features and output features need to be duplicated
        as edges are unidirectional in Pyg. Jumps are flipped, as are normals
        for edges going opposite way. RT velocity coefficients are flipped for edges 
        going opposite way. 
        """

        edges = np.concatenate([self.edges, self.edges[:,::-1]])

        input_features = np.concatenate([
            input_features,
            input_features*[-1,-1,1,1,-1,-1,1]
        ])

        output_features = np.concatenate([output_features, -output_features])

        # Graph data structure for PyG
        data_j = Data(
            y = torch.tensor(output_features, dtype=torch.float32),
            edge_index = torch.tensor(edges, dtype=torch.long).T,
            edge_attr = torch.tensor(input_features, dtype=torch.float32),
            pos = self.coords
        )

     
        #data_j.x = vertex_features

        return data_j
    
    # Load graphs for each model time step
    def get_graphs(self):
        data = []
        for i in range(160):
            print(i)
            g = self.get_graph(i)
            data.append(g)

        return data

        

#file_name = f'/home/jake/ManuscriptCode/examples/hybrid_runs/results/24/output.h5'      
#fd_loader = FDDataLoader(file_name)
#graph = fd_loader.get_graph(10)
