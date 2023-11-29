import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
from firedrake.petsc import PETSc
from velocity_model import VelocityModel
from conservation_model import ConservationModel
from firedrake_data_loader import FDDataLoader
import matplotlib.pyplot as plt

i = 33
base_dir = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/results/{i}'
file_name = f'{base_dir}/output.h5'

print(file_name)

loader = FDDataLoader(file_name)
B_dg = loader.B_dg
data_mapper = loader.data_mapper
conservation_model = ConservationModel(data_mapper)
velocity_model = VelocityModel(data_mapper)

V_cg = fd.FunctionSpace(loader.mesh, 'CG', 1)
Q_cg = fd.VectorFunctionSpace(loader.mesh, 'CG', 1)
U = fd.Function(Q_cg)
H_dg, beta2_dg, u_rt, adot_dg = loader.load_step(50)

Ubar_out = fd.Function(data_mapper.V_rt, name='Ubar')
H_file = fd.File('test1/H.pvd')
Ubar_file = fd.File('test1/Ubar.pvd')
S_file = fd.File('test1/S.pvd')
S_out = fd.Function(data_mapper.V_dg)

for j in range(100):
    print(j)
    
    S_dg = B_dg + H_dg   

    u = velocity_model.solve(B_dg, H_dg, beta2_dg)
    Ubar_out.dat.data[:] = -u
    U.assign(fd.project(Ubar_out, Q_cg))

    H = conservation_model.step(0.01, Ubar_out, H_dg, adot_dg)
    H_dg.assign(H)

    if j % 1 == 0:
        H_file.write(H_dg, time=j)
        Ubar_file.write(Ubar_out, time=j)
        #U_file.write(U, time=j)
        S_out.interpolate(S_dg)
        S_file.write(S_out, time=j)
