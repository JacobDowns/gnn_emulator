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
        self.Udef_obs = Udef_obs = df.Function(self.V)
        self.Udef = Udef = df.Function(self.V)
        self.Ubar = Ubar = df.Function(self.V)
        self.w_bar = df.TestFunction(self.V)
        self.w_def = df.TestFunction(self.V)

        c0 = df.Constant(0.9)
        c1 = df.Constant(0.1)

        rbar = Ubar - Ubar_obs
        rdef = Udef - Udef_obs

        self.I = c0*(df.dot(rbar, rbar) + df.dot(rdef, rdef))*df.dx
        self.I += c1*df.div(Ubar)**2*df.dx
        self.J_bar = df.derivative(self.I, self.Ubar, self.w_bar)
        self.J_def = df.derivative(self.I, self.Udef, self.w_def)


class VelocityLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Ubar, Udef, Ubar_obs, Udef_obs, loss_integral):
        ctx.loss_integral = loss_integral 

        ctx.Ubar = Ubar    
        ctx.Udef = Udef
        ctx.Ubar_obs = Ubar_obs
        ctx.Udef_obs = Udef_obs

        loss_integral.Ubar_obs.dat.data[:] = Ubar_obs
        loss_integral.Udef_obs.dat.data[:] = Udef_obs
        loss_integral.Ubar.dat.data[:] = Ubar
        loss_integral.Udef.dat.data[:] = Udef

        return torch.tensor(df.assemble(loss_integral.I)) 

    @staticmethod
    def backward(ctx, grad_output):
        loss_integral = ctx.loss_integral  

        Ubar_obs = ctx.Ubar_obs
        Udef_obs = ctx.Udef_obs
        Ubar = ctx.Ubar
        Udef = ctx.Udef

        loss_integral.Ubar_obs.dat.data[:] = Ubar_obs
        loss_integral.Udef_obs.dat.data[:] = Udef_obs
        loss_integral.Ubar.dat.data[:] = Ubar
        loss_integral.Udef.dat.data[:] = Udef

        j_bar = df.assemble(loss_integral.J_bar)  
        j_def = df.assemble(loss_integral.J_def)

        return torch.tensor(j_bar.dat.data[:])*grad_output, torch.tensor(j_def.dat.data[:])*grad_output, None, None, None