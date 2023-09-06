import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd
from data_mapper import DataMapper

for i in range(25):
    file_name = f'/home/jake/ManuscriptCode/examples/hybrid_runs/results/{i}/output.h5'

    with df.CheckpointFile(file_name, 'r') as afile:

        mesh = afile.load_mesh()
        data_mapper = DataMapper(mesh)
    
        # Edge indexes that connect two cells (not exterior edges)
        indexes = data_mapper.indexes
        # Connections between DG0 cells
        cell_connections = data_mapper.cell_connections
        # Edge midpoints
        edge_midpoints = data_mapper.edge_midpoints
        # Edge lengths
        edge_lens = data_mapper.edge_lens
        # Edge normal vectors
        edge_normals = data_mapper.edge_normals
        # DG0 cell midpoints
        cell_midpoints = data_mapper.cell_midpoints

        Us = []
        beta2s = []
        Hs = [] 
        Bs = []
        Ns = []

        # Bed on DG0 cells
        B = afile.load_function(mesh, 'B')  
        u_rt = fd.Function(data_mapper.V_rt)

        for j in range(1,160, 4):
            print(i, j)
            # Thickness on DG0 cells
            H = afile.load_function(mesh, 'H0', idx=j)
            # Basal traction field on DG0 cells
            beta2 = afile.load_function(mesh, 'beta2', idx=j)
            # Effective pressure on DG0 cells
            N = afile.load_function(mesh, 'N', idx=j)   
            # Velocity in RT space (edges)
            u = afile.load_function(mesh, 'U', idx=j)  
            u_rt.interpolate(fd.project(u, data_mapper.V_rt))

            beta2s.append(beta2.dat.data[:])
            Hs.append(H.dat.data[:])
            Bs.append(B.dat.data[:])
            Us.append(u_rt.dat.data[indexes] / edge_lens)
            Ns.append(N.dat.data[:])
         
        Bs = np.array(Bs, np.float32)
        Hs = np.array(Hs, np.float32)
        Us = np.array(Us, np.float32)
        Ns = np.array(Ns, np.float32)
        beta2s = np.array(beta2s, np.float32)

        # DG0 cell input variables
        X_cells = np.stack([Bs, Hs, beta2s, Ns]).transpose((1,2,0))
        # Edge input variables
        #X_edges = np.stack([edge_lens, edge_midpoints[:,0], edge_midpoints[:,1], edge_normals[:,0], edge_normals[:,1]]).transpose((1,0))
        X_edges = np.stack([edge_normals[:,0], edge_normals[:,1]]).transpose((1,0))
        # Edge output variables
        # Edge output variables
        Y_edges = Us[np.newaxis, :, :].transpose((1,2,0))

        np.save(f'data/cell_coords_{i}.npy', cell_midpoints)
        np.save(f'data/cell_connections_{i}.npy', cell_connections)
        np.save(f'data/X_cells_{i}.npy', X_cells)
        np.save(f'data/X_edges_{i}.npy', X_edges)
        np.save(f'data/Y_edges_{i}.npy', Y_edges)

        print(X_cells.shape)
        print(X_edges.shape)
        print(Y_edges.shape)
