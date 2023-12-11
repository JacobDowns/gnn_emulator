#import torch as torch
import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import firedrake.ml as fd_ml
import firedrake_adjoint as fda
from velocity_model import VelocityModel
from loss_model import FiredrakeLossModel, LossModel

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
    firedrake_loss = FiredrakeLossModel(mesh)
    loss_model = LossModel()

    velocity_model.solve(
        B,
        H,
        beta2,
        picard_tol=1e-3,
        momentum=0.0,
        max_iter=50,
        convergence_norm='linf'
    )


    firedrake_loss.B.assign(B)
    firedrake_loss.H.assign(H)
    firedrake_loss.beta2.assign(beta2)
    
    """
    R = loss_model.forward(
        firedrake_loss,
        velocity_model.Ubar0.dat.data,
        velocity_model.Udef0.dat.data
    )
    """
    
    R = loss_model.backward(
        firedrake_loss,
        velocity_model.Ubar0.dat.data,
        velocity_model.Udef0.dat.data
    )

    print(R)