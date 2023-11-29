import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
from data_mapper import DataMapper

class FDDataLoader:

    def __init__(self, file_name):

        self.file_name = file_name
        with fd.CheckpointFile(file_name, 'r') as afile:

            # Mesh topology
            self.mesh = afile.load_mesh()

            data_mapper = DataMapper(self.mesh)
            self.data_mapper = data_mapper
            
            # Function spaces
            self.V_rt = data_mapper.V_rt
            self.V_dg = data_mapper.V_dg
            self.V_cr = data_mapper.V_cr
            self.V_cg = data_mapper.V_cg

            # Functions for storing emulator input / output data
            self.u_rt = fd.Function(self.V_rt)
            self.B_dg = fd.Function(self.V_dg)
            self.H_dg = fd.Function(self.V_dg)
            self.beta2_cg = fd.Function(self.V_cg)
            self.adot_dg = fd.Function(self.V_dg)

            self.B_dg.assign(afile.load_function(self.mesh, 'B'))


    # Load model data from h5 file
    def load_step(self, j):
        with fd.CheckpointFile(self.file_name, 'r') as afile:

            self.H_dg.assign(afile.load_function(self.mesh, 'H0', idx=j))
            self.beta2_cg.assign(afile.load_function(self.mesh, 'beta2', idx=j))
            self.u_rt.assign(afile.load_function(self.mesh, 'Ubar0', idx=j))
            self.adot_dg.interpolate(afile.load_function(self.mesh, 'adot', idx=j))

            return self.H_dg, self.beta2_cg, self.u_rt, self.adot_dg
