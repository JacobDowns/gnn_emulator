import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
from data_mapper import DataMapper

class SimulationLoader:

    def __init__(self, file_name):

        self.file_name = file_name
        with fd.CheckpointFile(file_name, 'r') as afile:
            # Mesh topology
            self.mesh = afile.load_mesh()
            self.B_dg = afile.load_function(self.mesh, 'B')
          
    # Load model data from h5 file
    def load_step(self, j):

        with fd.CheckpointFile(self.file_name, 'r') as afile:

            H_dg = afile.load_function(self.mesh, 'H0', idx=j)
            beta2_cg = afile.load_function(self.mesh, 'beta2', idx=j)

            d = {}
            d['H_dg'] = H_dg
            d['B_dg'] = self.B_dg
            d['beta2_cg'] = beta2_cg

            return d

    # Load graphs for each model time step
    def get_data(self):
        data = []
        for i in range(10):
            print(i)
            d = self.load_step(i)
            data.append(d)

        return data
    

class DataLoader:

    def __init__(self, file_name):

        self.file_name = file_name
        with fd.CheckpointFile(file_name, 'r') as afile:
            # Mesh topology
            self.mesh = afile.load_mesh()
            self.B_dg = afile.load_function(self.mesh, 'B')
          
          
file_name = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/results/24/output.h5'      
fd_loader = SimulationLoader(file_name)
d = DataMapper(fd_loader.mesh)

data = fd_loader.get_data()
node_features = data[-1]['beta2_cg'].dat.data[:]
faces = d.faces

v_dg = fd.TestFunction(d.V_dg)
A = fd.assemble(v_dg*fd.dx).dat.data
print(A.shape)
print(faces.shape)
quit()

e0 = faces[:,[0,1]]
e1 = faces[:,[1,2]]
e2 = faces[:,[2,0]]




print(e0)
print(e1)
quit()
print(node_features)

print(node_features[d.edges])


#print(data[-1]['H_dg'].dat.data[:])


