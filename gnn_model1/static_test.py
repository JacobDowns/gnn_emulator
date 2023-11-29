import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
from firedrake.petsc import PETSc
from emulator_model import EmulatorModel
from firedrake_data_loader import FDDataLoader
import matplotlib.pyplot as plt

i = 37
base_dir = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/results/{i}'
file_name = f'{base_dir}/output.h5'

print(file_name)

loader = FDDataLoader(file_name)
B_dg = loader.B_dg

data_mapper = loader.data_mapper
velocity_model = EmulatorModel(data_mapper)



Ubar_obs_out = fd.Function(loader.V_rt, name='Ubar_obs')
Ubar_mod_out = fd.Function(loader.V_rt, name='Ubar_mod')

Ubar_obs_file = fd.File(f'stuff/Ubar_obs.pvd')
Ubar_mod_file = fd.File(f'stuff/Ubar_mod.pvd')

for j in range(140):
    print(j)
    H_dg, beta2_cg, u_obs, adot_dg = loader.load_step(j)
    
    u_mod = velocity_model.solve(B_dg, H_dg, beta2_cg)

    Ubar_mod_out.dat.data[:] = u_mod    
    Ubar_obs_out.assign(u_obs)

    Ubar_obs_file.write(Ubar_obs_out, time=j)
    Ubar_mod_file.write(Ubar_mod_out, time=j)

    
