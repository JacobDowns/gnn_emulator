import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
import numpy as np
import torch
from petsc4py import PETSc

class LossIntegral:
    def __init__(self, mesh):
        self.V = df.FunctionSpace(mesh, "RT", 1)
       
        self.Ubar_obs = Ubar_obs =  df.Function(self.V)
        self.Ubar = Ubar = df.Function(self.V)
        self.w_bar = df.TestFunction(self.V)

        c0 = df.Constant(1.)
        c1 = df.Constant(0.3)

        rbar = Ubar - Ubar_obs

        self.I = c0*df.dot(rbar, rbar)*df.dx
        #self.I += c1*df.div(Ubar)**2*df.dx
        self.I += c1*df.dot(df.grad(Ubar[0]), df.grad(Ubar[0]))*df.dx
        self.I += c1*df.dot(df.grad(Ubar[1]), df.grad(Ubar[1]))*df.dx
        self.J_bar = df.derivative(self.I, self.Ubar, self.w_bar)


class VelocityLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Ubar, Ubar_obs, loss_integral):
        ctx.loss_integral = loss_integral 

        ctx.Ubar = Ubar    
        ctx.Ubar_obs = Ubar_obs

        loss_integral.Ubar_obs.dat.data[:] = Ubar_obs
        loss_integral.Ubar.dat.data[:] = Ubar

        return torch.tensor(df.assemble(loss_integral.I)) 

    @staticmethod
    def backward(ctx, grad_output):
        loss_integral = ctx.loss_integral  

        Ubar_obs = ctx.Ubar_obs
        Ubar = ctx.Ubar

        loss_integral.Ubar_obs.dat.data[:] = Ubar_obs
        loss_integral.Ubar.dat.data[:] = Ubar

        j_bar = df.assemble(loss_integral.J_bar)  

        return torch.tensor(j_bar.dat.data[:])*grad_output, None, None, None