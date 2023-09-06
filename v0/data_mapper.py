import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd

class DataMapper:

    def __init__(self, mesh):
        self.mesh = mesh 

        V_dg = fd.FunctionSpace(mesh, 'DG', 0)
        V_cr = fd.FunctionSpace(mesh, 'CR', 1)
        V_rt = fd.FunctionSpace(mesh, 'RT', 1)
        self.V_dg = V_dg
        self.V_cr = V_cr
        self.V_rt = V_rt

        f_dg = fd.Function(V_dg)
        f_dg.dat.data[:] = np.arange(0, len(f_dg.dat.data[:])) + 1

        # Edge Lengths
        v_cr = fd.TestFunction(V_cr)
        edge_lens = fd.assemble(v_cr('+')*fd.dS + v_cr*fd.ds).dat.data
        self.edge_lens = edge_lens

        # Edge midpoints
        x = fd.SpatialCoordinate(mesh)
        x_cr = fd.interpolate(x[0], V_cr).dat.data
        y_cr = fd.interpolate(x[1], V_cr).dat.data
        self.edge_midpoints = np.c_[x_cr, y_cr]

        # Cell midpoints
        x_dg = fd.interpolate(x[0], V_dg).dat.data
        y_dg = fd.interpolate(x[1], V_dg).dat.data
        self.cell_midpoints = np.c_[x_dg, y_dg]

        # Normal vectors for each edge
        n = df.FacetNormal(mesh)
        n0 = fd.assemble(n[0]('+')*v_cr('+')*fd.dS).dat.data / edge_lens
        n1 = fd.assemble(n[1]('+')*v_cr('+')*fd.dS).dat.data / edge_lens
        self.edge_normals = np.c_[n0, n1]

        # Determine connectivity between cells
        c0 = fd.assemble(f_dg('+')*v_cr('+')*fd.dS).dat.data / edge_lens
        c1 = fd.assemble(f_dg('-')*v_cr('-')*fd.dS).dat.data / edge_lens
        c0 = np.rint(c0).astype(int) 
        c1 = np.rint(c1).astype(int)
        indexes = np.logical_and(c0 > 0, c1 > 0)
        cell_connections = np.c_[c0[indexes] - 1, c1[indexes] - 1]
        self.cell_connections = cell_connections

        self.indexes = indexes
        self.edge_lens = self.edge_lens[indexes]
        self.edge_midpoints = self.edge_midpoints[indexes]
        self.edge_normals = self.edge_normals[indexes]

     