#import torch as torch
import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
from velocity_model import VelocityModel
from loss_model import FiredrakeLossModel, VelocityLoss
import torch
from petsc4py import PETSc
import numpy as np


mesh = fd.RectangleMesh(20, 20, 10, 10)

V_cg = fd.FunctionSpace(mesh, 'CG', 1)
V_dg = fd.FunctionSpace(mesh, 'DG', 0)
V_rt = fd.FunctionSpace(mesh, 'RT', 1)

B = fd.Function(V_dg)

x, y = fd.SpatialCoordinate(mesh)
H = fd.Function(V_dg, name='H').interpolate(500. - 110.*fd.sqrt((x-5)**2 + (y-5)**2))

H.dat.data[H.dat.data < 1.] = 1.
beta2 = fd.Function(V_cg)
beta2.dat.data[:] = 0.5

firedrake_loss = FiredrakeLossModel(mesh)
loss_model = VelocityLoss(firedrake_loss)

Ubar = fd.Function(V_rt)
Udef = fd.Function(V_rt)
firedrake_loss.W.sub(0).assign(Ubar)
