import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
from firedrake.petsc import PETSc
from velocity_model import VelocityModel
from conservation_model import ConservationModel
from data_mapper import DataMapper
from emulator_model import EmulatorModel

i = 37
base_dir = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/inputs'
file_name = f'{base_dir}/input_{i}.h5'


data_mapper, inputs = load_inputs(file_name)

conservation_model = ConservationModel(data_mapper)
emulator_model = EmulatorModel(data_mapper)

H0 = inputs['H_dg']
B = inputs['B_dg']
beta2 = fd.Function(data_mapper.V_cg)
adot_dg = fd.Function(data_mapper.V_dg)
Ubar = fd.Function(data_mapper.V_rt)
beta2 = fd.Function(data_mapper.V_cg)

t = 0.
t_end = 20
dt = 0.25

Ubar_out = fd.Function(data_mapper.V_rt, name='Ubar_mod')
H_out = fd.Function(data_mapper.V_dg, name='H')

Ubar_file = fd.File(f'test/Ubar.pvd')
H_file = fd.File(f'test/H.pvd')

j = 0
while t < t_end:
    print(t)

    beta0 = 1. + (2./3.)*np.cos(t*2.*np.pi / 100.)
    adot0 = 0.5*np.cos(t*2.*np.pi / 400. ) 

    beta2_cg.interpolate(inputs['beta2']*fd.Constant(beta0))
    adot_dg.interpolate(inputs['adot'] + fd.Constant(adot0))

    #velocity_model.step(B, H0, beta2)

    u_mod = velocity_model.solve(B_dg, H_dg, beta2_cg)
    Ubar_mod_out.dat.data[:] = u_mod

    #H = conservation_model.step(dt, velocity_model.Ubar0, H0, adot_dg)
    H = conservation_model.step(dt, Ubar_mod_out, H0, adot_dg)
    H0.assign(H)

    if j % 5 == 0:
        H_out.assign(H0)
        Ubar_file.write(Ubar_out, time=t)
        H_file.write(H_out, time=t)
        Ubar_mod_file.write(Ubar_mod_out, time=t)


    t += dt
    j += 1





  