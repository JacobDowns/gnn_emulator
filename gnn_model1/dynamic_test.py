import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
from firedrake.petsc import PETSc
from emulator_model import EmulatorModel
from conservation_model import ConservationModel
from firedrake_data_loader import FDDataLoader
import matplotlib.pyplot as plt

i = 37
base_dir = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/results/{i}'
file_name = f'{base_dir}/output.h5'

loader = FDDataLoader(file_name)
B_dg = loader.B_dg
H_dg, beta2_cg, u_obs, adot_dg = loader.load_step(0)

data_mapper = loader.data_mapper
H0 = fd.Function(data_mapper.V_dg)
H0.assign(H_dg)
velocity_model = EmulatorModel(data_mapper)
conservation_model = ConservationModel(data_mapper)


H_out = fd.Function(loader.V_dg, name='H')
Ubar_out = fd.Function(loader.V_rt, name='Ubar')
Ubar_file = fd.File('test/Ubar.pvd')
H_file = fd.File('test/H.pvd')

j = 0
t = 0
t_end = 50.
dt = 0.1
while t < t_end:
    print(t)

    u_mod = velocity_model.solve(B_dg, H0, beta2_cg)
    Ubar_out.dat.data[:] = u_mod

    H = conservation_model.step(dt, Ubar_out, H0, adot_dg)
    H0.assign(H)

    if j % 5 == 0:
        H_out.assign(H0)
        Ubar_file.write(Ubar_out, time=t)
        H_file.write(H_out, time=t)


    t += dt
    j += 1


    
