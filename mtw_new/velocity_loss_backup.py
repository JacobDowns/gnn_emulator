import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
import numpy as np
import torch
from petsc4py import PETSc

class LossIntegral:
    def __init__(self, mesh):
        self.V_rt = df.FunctionSpace(mesh, "RT", 1)
        self.V_dg = df.FunctionSpace(mesh, 'DG', 0)
        
        self.Ubar_obs = Ubar_obs =  df.Function(self.V_rt)
        self.Ubar = Ubar = df.Function(self.V_rt)
        self.w_bar = df.TestFunction(self.V_rt)
        self.H = H = df.Function(self.V_dg)

        c0 = df.Constant(0.9)
        c1 = df.Constant(0.1)

        rbar = Ubar - Ubar_obs

        self.I = c0*df.dot(rbar, rbar)*df.dx
        #self.I += c1*df.div(rbar*H)**2*df.dx
        #self.I += c1*df.div(Ubar)**2*df.dx
        self.I += c1*df.dot(df.grad(Ubar[0]), df.grad(Ubar[0]))*df.dx
        self.I += c1*df.dot(df.grad(Ubar[1]), df.grad(Ubar[1]))*df.dx
        self.J_bar = df.derivative(self.I, self.Ubar, self.w_bar)


class VelocityLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Ubar, Ubar_obs, H, loss_integral):
        ctx.loss_integral = loss_integral 

        ctx.Ubar = Ubar    
        ctx.Ubar_obs = Ubar_obs
        ctx.H = H

        loss_integral.Ubar_obs.dat.data[:] = Ubar_obs
        loss_integral.Ubar.dat.data[:] = Ubar
        loss_integral.H.dat.data[:] = H

        return torch.tensor(df.assemble(loss_integral.I)) 

    @staticmethod
    def backward(ctx, grad_output):
        loss_integral = ctx.loss_integral  

        Ubar_obs = ctx.Ubar_obs
        Ubar = ctx.Ubar
        H = ctx.H

        loss_integral.Ubar_obs.dat.data[:] = Ubar_obs
        loss_integral.Ubar.dat.data[:] = Ubar
        loss_integral.H.dat.data[:] = H

        j_bar = df.assemble(loss_integral.J_bar)  

        return torch.tensor(j_bar.dat.data[:])*grad_output, None, None, None