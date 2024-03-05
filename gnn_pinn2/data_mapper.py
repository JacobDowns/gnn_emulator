import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd
from matplotlib.tri import Triangulation
import os
os.environ['OMP_NUM_THREADS'] = '1'

class DataMapper:

    def __init__(self, mesh):
        self.mesh = mesh
        V_cg = self.V_cg = fd.FunctionSpace(mesh, 'CG', 1)
        V_dg = self.V_dg = fd.FunctionSpace(mesh, 'DG', 0)
        V_cr = self.V_cr = fd.FunctionSpace(mesh, 'CR', 1)
        V_rt = self.V_rt = fd.FunctionSpace(mesh, 'RT', 1)
        V_mtw = self.V_mtw = fd.FunctionSpace(mesh, 'MTW', 3)

        # Node coordinates
        self.xs = mesh.coordinates.dat.data[:,0]
        self.ys = mesh.coordinates.dat.data[:,1]
        self.coords = np.c_[self.xs, self.ys]

        # Faces
        self.faces = faces = V_cg.cell_node_list

        # Get edges
        e0 = faces[:,[0,1]]
        e1 = faces[:,[1,2]]
        e2 = faces[:,[2,0]]
        e0.sort(axis=1)
        e1.sort(axis=1)
        e2.sort(axis=1)
        edges = np.concatenate((e0, e1, e2)) 
        edges = np.unique(edges, axis=0)
        self.edges = edges

        # Edge midpoint coordinates
        x = fd.SpatialCoordinate(mesh)
        x_cr = fd.interpolate(x[0], V_cr)
        y_cr = fd.interpolate(x[1], V_cr)
        x_cr = x_cr.dat.data
        y_cr = y_cr.dat.data
        self.x_cr = x_cr
        self.y_cr = y_cr

        def get_edge_map():
            ex0 = 0.5*self.xs[edges].sum(axis=1)
            ey0 = 0.5*self.ys[edges].sum(axis=1)

            self.ei0 = np.lexsort((ex0, ey0)) 
            self.ei1 = np.lexsort((x_cr, y_cr))

        get_edge_map()
        edges1 = np.zeros_like(edges)
        edges1[self.ei1] = self.edges[self.ei0]
        self.edges = edges1


        # Normal vectors
        nhat = fd.FacetNormal(mesh)
        v_cr = fd.TestFunction(V_cr)
        self.n0 = fd.assemble((nhat[0]*v_cr)('+')*fd.dS + nhat[0]*v_cr*fd.ds).dat.data
        self.n1 = fd.assemble((nhat[1]*v_cr)('+')*fd.dS + nhat[1]*v_cr*fd.ds).dat.data
        self.n = np.c_[self.n0, self.n1]

        # Edge lengths
        self.edge_lens = fd.assemble(v_cr('+')*fd.dS + v_cr*fd.ds).dat.data
        self.v_cr = v_cr
        
        # Cell areas 
        v_dg = fd.TestFunction(V_dg)
        A = fd.assemble(v_dg * fd.dx).dat.data
        areas = fd.Function(V_dg)
        areas.dat.data[:] = A

        self.a0 = fd.assemble((areas*v_cr)('+')*fd.dS + areas*v_cr*fd.ds).dat.data
        self.a1 = fd.assemble((areas*v_cr)('-')*fd.dS + areas*v_cr*fd.ds).dat.data