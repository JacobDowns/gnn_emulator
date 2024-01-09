import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
import torch
from velocity_model import VelocityModel
from loss_model import FiredrakeLossModel, LossModel
import torch.optim as optim

i = 10
file_name = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/results/{i}/output.h5'

with fd.CheckpointFile(file_name, 'r') as afile:

    mesh = afile.load_mesh()

    j = 10
    B = afile.load_function(mesh, 'B')
    H = afile.load_function(mesh, 'H0', idx=j)
    beta2 = afile.load_function(mesh, 'beta2', idx=j)


    
    velocity_model = VelocityModel(mesh)
    firedrake_loss = FiredrakeLossModel(mesh)
    loss_model = LossModel()

    ubar = fd.Function(firedrake_loss.Q_vel)
    udef = fd.Function(firedrake_loss.Q_vel)

    ubar0 = torch.zeros(ubar.dat.data.shape, requires_grad=True)
    udef0 = torch.zeros(udef.dat.data.shape, requires_grad=True)

    optimizer = optim.SGD([ubar0, udef0], lr=1e-10)
    num_steps = 100

    firedrake_loss.B.assign(B)
    firedrake_loss.H.assign(H)
    firedrake_loss.beta2.assign(beta2)

    for step in range(num_steps):
        # Step 3a: Forward pass - compute the loss
        L = loss_model.apply
        loss = L(ubar0, udef0, firedrake_loss)

        # Step 3b: Backward pass - compute gradients
        optimizer.zero_grad()  # Zero gradients to avoid accumulation
        loss.backward()

        # Step 3c: Update parameters using the optimizer
        optimizer.step()

        # Optionally, print the loss at each step
        if step % 1 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

        # Access the optimized values of x and y
        #print("Optimized x:", x.item())
        #print("Optimized y:", y.item())


