#import torch as torch
import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
from velocity_model import VelocityModel
from loss_model import LossModel, VelocityCost
import torch
from petsc4py import PETSc
import numpy as np
import torch.optim as optim

i = 0
file_name = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/results/{i}/output.h5'

with fd.CheckpointFile(file_name, 'r') as afile:

    mesh = afile.load_mesh()

    j = 20
    
    B = afile.load_function(mesh, 'B')
    H = afile.load_function(mesh, 'H0', idx=j)
    beta2 = afile.load_function(mesh, 'beta2', idx=j)
    u = afile.load_function(mesh, 'Ubar0', idx=j)

    loss_integral = LossModel(mesh)
    loss_integral.B.assign(B)
    loss_integral.H.assign(H)
    loss_integral.beta2.assign(beta2)
    loss_integral.B_grad_solver.solve()
    loss_integral.S_grad_solver.solve()


    cost = VelocityCost()

    ubar = fd.Function(loss_integral.Q_vel)
    udef = fd.Function(loss_integral.Q_vel)

    ubar0 = torch.zeros(ubar.dat.data.shape, requires_grad=True)
    udef0 = torch.zeros(udef.dat.data.shape, requires_grad=True)

    optimizer = optim.SGD([ubar0, udef0], lr=1e-10)
    num_steps = 100

    L = cost.apply

    for step in range(num_steps):
        # Step 3a: Forward pass - compute the loss
        loss = L(ubar0, udef0, loss_integral)

        # Step 3b: Backward pass - compute gradients
        optimizer.zero_grad()  # Zero gradients to avoid accumulation
        loss.backward()

        quit()

        # Step 3c: Update parameters using the optimizer
        optimizer.step()

        # Optionally, print the loss at each step
        if step % 1 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

        # Access the optimized values of x and y
        #print("Optimized x:", x.item())
        #print("Optimized y:", y.item())

    








    