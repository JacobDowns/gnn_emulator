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

loader = FDDataLoader(file_name)
B_dg = loader.B_dg
data_mapper = loader.data_mapper
conservation_model = ConservationModel(data_mapper)
velocity_model = VelocityModel(loader.mesh)


H_dg, beta2_dg, u_rt, adot_dg = loader.load_step(5)

V_cg = fd.FunctionSpace(data_mapper.mesh, 'CG', 1)
beta2 = fd.Function(V_cg)
beta2.interpolate(beta2_dg)

velocity_model.step(B_dg, H_dg, beta2)

Ubar_out = fd.Function(data_mapper.V_rt, name='Ubar')
Ubar_file = fd.File('test2/Ubar.pvd')
Ubar_out.assign(velocity_model.Ubar0)
Ubar_file.write(Ubar_out, idx=0)
quit()


H_file = fd.File('test1/H.pvd')

S_file = fd.File('test1/S.pvd')
S_out = fd.Function(data_mapper.V_dg)

for j in range(100):
    print(j)
    
    S_dg = B_dg + H_dg   

    velocity_model.step()
    

    H = conservation_model.step(0.01, Ubar_out, H_dg, adot_dg)
    H_dg.assign(H)

    if j % 1 == 0:
        H_file.write(H_dg, time=j)
        Ubar_file.write(Ubar_out, time=j)
        #U_file.write(U, time=j)
        S_out.interpolate(S_dg)
        S_file.write(S_out, time=j)
