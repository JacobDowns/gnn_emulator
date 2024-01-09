#import torch as torch
import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
from velocity_model import VelocityModel
from loss_model import LossModel
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

    loss_integral = LossModel(mesh)


    loss_integral.B.assign(B)
    loss_integral.H.assign(H)
    loss_integral.beta2.assign(beta2)

    loss_integral.B_grad_solver.solve()
    loss_integral.S_grad_solver.solve()


    #fd.solve(loss_integral.R_full == 0, loss_integral.W)

    
    #n = len(u.dat.data)
    u0 = fd.Function(loss_integral.V)
    x = fd.Function(loss_integral.V)


    for i in range(3):

        print(i)


        R = fd.assemble(loss_integral.R_full)
        #r = np.concatenate(R.dat.data)
        #r_p = PETSc.Vec().createWithArray(r)
        J = fd.assemble(loss_integral.J_full)

        ksp = PETSc.KSP().create()
        ksp.setOperators(J.M.handle)

        with u0.dat.vec as u0_p:
            with R.dat.vec_ro as r_p:
                with x.dat.vec as x_p:
                    ksp.solve(r_p, x_p)
                    u0_p.axpy(-1.0, x_p)


        loss_integral.W.assign(u0)
        print(u0.dat.data)

    out = fd.File('test1.pvd')
    out.write(u0.sub(0))


    








    