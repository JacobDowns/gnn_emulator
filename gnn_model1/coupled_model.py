import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
import time
from firedrake.petsc import PETSc
from data_mapper import DataMapper
from velocity_model import VelocityModel
from conservation_model import ConservationModel

class CoupledModel:
    def __init__(
        self, 
        mesh, 
    ):
                
        self.mesh = mesh
        self.data_mapper = DataMapper(self.mesh)

        self.velocity_model = VelocityModel(self.data_mapper)
        self.conservation_model = VelocityModel(self.data_mapper)

    