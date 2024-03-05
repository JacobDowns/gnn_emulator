#import torch as torch
import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
from coupled_model import CoupledModel
import torch
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

#i = 10
#file_name = f'/media/new/gnn_emulator_runs/{i}/output.h5'

i = 2
base_dir = f'/media/new/gnn_emulator_runs1/{i}'
file_name = f'{base_dir}/output.h5'

with fd.CheckpointFile(file_name, 'r') as afile:

    mesh = afile.load_mesh()

    j = 20
    
    B = afile.load_function(mesh, 'B')
    H = afile.load_function(mesh, 'H0', idx=j)
    beta2 = afile.load_function(mesh, 'beta2', idx=j)
    Ubar = afile.load_function(mesh, 'Ubar0', idx=j)
    #Udef = afile.load_function(mesh, 'Udef0', idx=j)

    vel_scale = 1.
    thk_scale = 1.
    len_scale = 1e3
    beta_scale = 1e4
   
    config = {
        'solver_type': 'direct',
        'sliding_law': 'linear',
        'vel_scale': vel_scale,
        'thk_scale': thk_scale,
        'len_scale': len_scale,
        'beta_scale': beta_scale,
        'theta': 1.0,
        'thklim': 1.,
        'alpha': 1000.0,
        'z_sea': -1000.,
        'calve': False,
        'velocity_function_space' : 'MTW'
    }
    
    model = CoupledModel(mesh, **config)
    R_full = fd.replace(model.R_stress, {model.W_i:model.W})
    J_full = fd.derivative(R_full, model.W)
    model.B.assign(B)
    model.H.assign(H)
    model.beta2.assign(beta2)
    model.B_grad_solver.solve()
    model.S_grad_solver.solve()
    
    #n = len(u.dat.data)
    u0 = fd.Function(model.W)
    x = fd.Function(model.W)

    ubar, udef = fd.split(u0) 

    v0 = fd.Function(model.Q_vel)
    v1 = fd.Function(model.Q_vel)


    for i in range(10):

        print(i)


        R = fd.assemble(R_full)
        J = fd.assemble(J_full).M.handle


        with u0.dat.vec as u0_p:
            with R.dat.vec_ro as r_p:
                with x.dat.vec as x_p:
                    J.multTranspose(r_p, x_p)
                    u0_p.axpy(-1.0, x_p)


        model.W.assign(u0)
        r = np.concatenate(R.dat.data)**2
        print(r.sum())

    quit()

    v0.dat.data[:] = u0.dat.data[0]
    v1.dat.data[:] = u0.dat.data[1]

    out1 = fd.File('a.pvd')
    out2 = fd.File('b.pvd')
    out1.write(v0)
    out2.write(v1)


    








    