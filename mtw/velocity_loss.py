import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
import numpy as np
import torch
from petsc4py import PETSc

class LossIntegral:
    def __init__(self, mesh):
        self.V_mt = df.FunctionSpace(mesh, "MTW", 3)
        self.V_dg = df.FunctionSpace(mesh, 'DG', 0)
        
        self.Ubar_obs = Ubar_obs = df.Function(self.V_mt)
        self.Udef_obs = Udef_obs = df.Function(self.V_mt)
        self.Ubar = Ubar = df.Function(self.V_mt)
        self.Udef = Udef = df.Function(self.V_mt)

        self.w_bar = df.TestFunction(self.V_mt)
        self.w_def = df.TestFunction(self.V_mt)

        r0 = Ubar - Ubar_obs
        r1 = Udef - Udef_obs

        self.I = df.dot(r0, r0)*df.dx + df.dot(r1, r1)*df.dx

        self.J_bar = df.derivative(self.I, self.Ubar, self.w_bar)
        self.J_def = df.derivative(self.I, self.Udef, self.w_def)


class VelocityLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Ubar, Udef, Ubar_obs, Udef_obs, loss_integral):

        ctx.loss_integral = loss_integral 

        ctx.Ubar = Ubar 
        ctx.Udef = Udef
        ctx.Udef_obs = Udef_obs   
        ctx.Ubar_obs = Ubar_obs

        loss_integral.Ubar.dat.data[:] = Ubar
        loss_integral.Udef.dat.data[:] = Udef
        loss_integral.Ubar_obs.dat.data[:] = Ubar_obs
        loss_integral.Udef_obs.dat.data[:] = Udef_obs

        return torch.tensor(df.assemble(loss_integral.I))

    @staticmethod
    def backward(ctx, grad_output):
        loss_integral = ctx.loss_integral  

        Ubar = ctx.Ubar
        Udef = ctx.Udef
        Ubar_obs = ctx.Ubar_obs
        Udef_obs = ctx.Udef_obs

        loss_integral.Ubar.dat.data[:] = Ubar
        loss_integral.Udef.dat.data[:] = Udef
        loss_integral.Ubar_obs.dat.data[:] = Ubar_obs
        loss_integral.Udef_obs.dat.data[:] = Udef_obs
       
        j_bar = torch.tensor(df.assemble(loss_integral.J_bar).dat.data[:]).clone().detach()
        j_def = torch.tensor(df.assemble(loss_integral.J_def).dat.data[:]).clone().detach()

        return j_bar*grad_output, j_def*grad_output, None, None, None