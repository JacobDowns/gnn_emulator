#import torch as torch
import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
from velocity_model import VelocityModel
from mixed_loss_model import LossIntegral, VelocityLoss
import torch
from petsc4py import PETSc
import numpy as np

i = 0
file_name = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/results/{i}/output.h5'

with fd.CheckpointFile(file_name, 'r') as afile:

    mesh = afile.load_mesh()

    j = 20
    
    B = afile.load_function(mesh, 'B')
    H = afile.load_function(mesh, 'H0', idx=j)
    beta2 = afile.load_function(mesh, 'beta2', idx=j)
    u = afile.load_function(mesh, 'Ubar0', idx=j)

    velocity_model = VelocityModel(mesh)
    loss_integral = LossIntegral(mesh)
    loss = VelocityLoss(loss_integral)

    velocity_model.solve(
        B,
        H,
        beta2,
        picard_tol=1e-3,
        momentum=0.0,
        max_iter=50,
        convergence_norm='linf'
    )

    L = loss.apply

    Ubar = torch.tensor(velocity_model.Ubar0.dat.data[:], requires_grad=True)
    Udef = torch.tensor(velocity_model.Ubar0.dat.data[:], requires_grad=True)
    Ubar_obs = torch.tensor(velocity_model.Ubar0.dat.data[:])*0.
    Udef_obs = torch.tensor(velocity_model.Udef0.dat.data[:])*0.

    x = L(Ubar, Udef, Ubar_obs, Udef_obs, loss_integral)
    x.backward()

    print(x)
    print(Ubar.grad)
    print(Udef.grad)

    quit()
    result = custom_function(velocity_model.Ubar0.dat.data[:], y, additional_info)