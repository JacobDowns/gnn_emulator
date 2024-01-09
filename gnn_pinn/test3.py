#import torch as torch
import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
from velocity_model import VelocityModel
from loss_model import PinnLoss
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
    loss_integral = PinnLoss(mesh)

    out = fd.File('out.pvd')
    out.write(u)

    """
    velocity_model.solve(
        B,
        H,
        beta2,
        picard_tol=1e-3,
        momentum=0.0,
        max_iter=50,
        convergence_norm='linf'
    )
    """

    #u0 = np.concatenate(loss_integral.W.dat.data)
    #u0_p = PETSc.Vec().createWithArray(u0)
    loss_integral.B.assign(B)
    loss_integral.H.assign(H)
    loss_integral.beta2.assign(beta2)

    loss_integral.B_grad_solver.solve()
    loss_integral.S_grad_solver.solve()

    fd.solve(loss_integral.R_stress == 0, loss_integral.W)


    print(loss_integral.Ubar0)

   

    quit()
    
    n = len(u.dat.data)

    u0 = fd.Function(loss_integral.V)
    u1 = fd.Function(loss_integral.V) 
    x = fd.Function(loss_integral.V)


    for i in range(10):

        #loss_integral.W.sub(0).dat.data[:] = u0[0:n]
        #loss_integral.W.sub(1).dat.data[:] = u0[n:]

      


        R = fd.assemble(loss_integral.R_stress)
        #r = np.concatenate(R.dat.data)
        #r_p = PETSc.Vec().createWithArray(r)
        J = fd.assemble(loss_integral.J)

        print(J)

        ksp = PETSc.KSP().create()
        ksp.setOperators(J.M.handle)

        with R.dat.vec_ro as r_p:
            with x.dat.vec as x_p:
                print(r_p)
                print(x_p)

                ksp.solve(r_p, x_p)

                print(x_p)
                print(r_p)
    
        print(R)

        #x_p = Petsc.Vec().createWithArray(r)


        with self.delta.dat.vec_ro as vec:
            with x.dat.vec as x:
                self.ksp.solve(r_p, x_p)
        loss_integral.W
        print(r)








    