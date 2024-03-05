from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv
import torch
import firedrake as fd
import os
os.environ['OMP_NUM_THREADS'] = '1'
from data_mapper import DataMapper
import numpy as np

#dx = 16.
#dy = 16.
#mesh = fd.RectangleMesh(4, 4, dx, dy)

mesh = fd.UnitDiskMesh(1)







nhat = fd.FacetNormal(mesh)
#nhat /= fd.sqrt(nhat[0]**2 + nhat[1]**2)

d = DataMapper(mesh)

f = fd.Function(d.V_mtw)


v = fd.TestFunction(d.V_cr)
edge_lens = d.edge_lens

"""
F0 = (fd.dot(f, nhat)*v)('-')
F1 = (fd.dot(f, nhat)*v)('-')

f.dat.data[0::3] = 1.
x0 = fd.assemble(F0*fd.dS).dat.data
print(x0)

f.dat.data[:] = 0.
f.dat.data[1::3] = 2.
x1 = fd.assemble(F0*fd.dS).dat.data
print(x1)

f.dat.data[:] = 0.
f.dat.data[2::3] = 3.
x2 = fd.assemble(F0*fd.dS).dat.data
print(x2)

print('sum')

s = x0 + x1 +x2
print(x0 + x1 + x2)

f.dat.data[0::3] = 1.
f.dat.data[1::3] = 2.
f.dat.data[2::3] = 3.
x = fd.assemble(F0*fd.dS).dat.data
print(s - x)
"""

f0 = np.random.randn(len(f.dat.data[0::3]))
f1 = np.random.randn(len(f.dat.data[1::3]))
f2 = np.random.randn(len(f.dat.data[2::3]))

f.dat.data[0::3] = f0
f.dat.data[1::3] = f1
f.dat.data[2::3] = f2

x = fd.assemble(fd.dot(fd.avg(f), nhat('+'))*v('+')*fd.dS).dat.data

x0 = fd.assemble((fd.dot(f, nhat)*v)('+')*fd.dS).dat.data
x1 = fd.assemble((fd.dot(f, nhat)*v)('+')*fd.dS).dat.data

x0 = fd.assemble((fd.dot(f, nhat)*v)('+')*fd.dS).dat.data
x1 = fd.assemble((fd.dot(f, nhat)*v)('+')*fd.dS).dat.data


#y = fd.assemble(fd.dot(abs(fd.avg(f)), abs(nhat('+')))*v('-')*fd.dS).dat.data
#print(y)

y0 = fd.assemble(0.5*abs(fd.dot(f('+'),nhat('+')))*v('+')*fd.dS).dat.data
y1 = fd.assemble(0.5*abs(fd.dot(f('+'),nhat('-')))*v('-')*fd.dS).dat.data
print(y0)
print(y1)
quit()

print(x)
print((x0+x1)/2.)

