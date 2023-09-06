import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
from firedrake import FunctionSpace, Function

class DataMapper:

    def __init__(self, mesh):
        self.mesh = mesh
        
        V_dg = fd.FunctionSpace(mesh, 'DG', 0)
        V_cr = fd.FunctionSpace(mesh, 'CR', 1)
        V_cg = fd.FunctionSpace(mesh, 'CG', 1)
        V_rt = fd.FunctionSpace(mesh, 'RT', 1)

        self.V_dg = V_dg
        self.V_cr = V_cr
        self.V_cg = V_cg
        self.V_rt = V_rt

        faces = V_cg.cell_node_list

        e0 = faces[:,[0,1]]
        e1 = faces[:,[1,2]]
        e2 = faces[:,[2,0]]

        e0.sort(axis=1)
        e1.sort(axis=1)
        e2.sort(axis=1)
       
        edges = np.concatenate((e0, e1, e2)) 
        edges = np.unique(edges, axis=0)
        self.edges = edges

        # Node coordinates
        x = fd.SpatialCoordinate(mesh)
        x_cg = fd.interpolate(x[0], V_cg)
        y_cg = fd.interpolate(x[1], V_cg)
        x_cg = x_cg.dat.data
        y_cg = y_cg.dat.data
        self.x = x_cg
        self.y = y_cg    

        # Edge midpoint coordinates
        x_cr = fd.interpolate(x[0], V_cr)
        y_cr = fd.interpolate(x[1], V_cr)
        x_cr = x_cr.dat.data
        y_cr = y_cr.dat.data
        self.x_cr = x_cr
        self.y_cr = y_cr

        def get_edge_map():
            ex0 = 0.5*x_cg[edges].sum(axis=1)
            ey0 = 0.5*y_cg[edges].sum(axis=1)

            self.ei0 = np.lexsort((ex0, ey0)) 
            self.ei1 = np.lexsort((x_cr, y_cr))

        get_edge_map()
        edges1 = np.zeros_like(edges)
        edges1[self.ei1] = self.edges[self.ei0]
        self.edges = edges1

        def get_normals():
            # Normal vectors for each edge, oriented consistently with RT elements
            v0 = fd.Function(V_cg)
            v1 = fd.Function(V_cg)
            v0.dat.data[:] = 1.
            v1.dat.data[:] = 0.
            f0 = fd.as_vector([v0, v1])
            f1 = fd.as_vector([v1, v0])
            n0 = fd.project(f0, V_rt)
            n1 = fd.project(f1, V_rt)
            n_len = np.sqrt(n0.dat.data[:]**2 + n1.dat.data[:]**2)
            n0.dat.data[:] /= n_len
            n1.dat.data[:] /= n_len
            self.edge_normals = np.c_[n0.dat.data[:], n1.dat.data[:]]

        get_normals()
        
        self.v_cr = fd.TestFunction(self.V_cr)
        self.edge_lens = fd.assemble(self.v_cr('+')*fd.dS + self.v_cr*fd.ds).dat.data

        
    def get_jump(self, f):
        v_cr = self.v_cr
        return fd.assemble(fd.jump(f*v_cr)*fd.dS).dat.data / self.edge_lens
    
    def get_normals(self):
        return self.edge_normals
    
    def get_coords(self):
        return np.c_[self.x, self.y]

    def get_edges(self):
        return self.edges

    def get_edge_midpoints(self):
        return np.c_[self.x_cr, self.y_cr]
        
