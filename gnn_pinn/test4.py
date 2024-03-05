#import torch as torch
import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
from velocity_model import VelocityModel
from loss_model import LossModel
import torch
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

i = 20
file_name = f'/home/jake/ManuscriptCode/examples/gnn_emulator_runs/results/{i}/output.h5'

with fd.CheckpointFile(file_name, 'r') as afile:

    mesh = afile.load_mesh()

    j = 50
    
    B = afile.load_function(mesh, 'B')
    H = afile.load_function(mesh, 'H0', idx=j)
    beta2 = afile.load_function(mesh, 'beta2', idx=j)
    #u = afile.load_function(mesh, 'Ubar0', idx=j)

    loss_integral = LossModel(mesh)

    print(beta2.dat.data)

    loss_integral.B.assign(B)
    loss_integral.H.assign(H)
    loss_integral.beta2.assign(beta2)

    loss_integral.B_grad_solver.solve()
    loss_integral.S_grad_solver.solve()

    z = fd.Function(loss_integral.V)
    fd.solve(loss_integral.R == 0, loss_integral.W)

    a = fd.File('c.pvd')
    b = fd.File('d.pvd')
    a.write(loss_integral.W.sub(0))
    b.write(loss_integral.W.sub(1))

    quit()
    
    #n = len(u.dat.data)
    u0 = fd.Function(loss_integral.V)
    x = fd.Function(loss_integral.V)

    ubar, udef = fd.split(u0) 

    v0 = fd.Function(loss_integral.Q_bar)
    v1 = fd.Function(loss_integral.Q_def)


    for i in range(10):

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

        #print(np.sqrt(np.concatenate(R.dat.data)**2).sum())
        print((u0.dat.data[0] - u0.dat.data[1]).max())


    v0.dat.data[:] = u0.dat.data[0]
    v1.dat.data[:] = u0.dat.data[1]

    out1 = fd.File('a.pvd')
    out2 = fd.File('b.pvd')
    out1.write(v0)
    out2.write(v1)


    








    