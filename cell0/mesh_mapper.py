import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt


class MeshMapper:
     
    def __init__(self, mesh):
        self.mesh = mesh

        V_cg = fd.FunctionSpace(mesh, 'CG', 1)
        V_dg = fd.FunctionSpace(mesh, 'DG', 0) 
        V_cr = fd.FunctionSpace(mesh, 'CR', 1)
        
        ### Vertex coordinates
        #######################################################
        vertex_xs = mesh.coordinates.dat.data[:,0]
        vertex_ys = mesh.coordinates.dat.data[:,1]

        # Cell coordinates
        ########################################################
        faces = V_cg.cell_node_list
        self.faces = faces
        cell_xs = vertex_xs[faces]
        cell_ys = vertex_ys[faces]
        # Cell midpoint coordinates
        cell_mid_xs = cell_xs.mean(axis=1)
        cell_mid_ys = cell_ys.mean(axis=1)
        self.coords = np.c_[cell_mid_xs, cell_mid_ys]
        # Local cell coordiantes 
        local_cell_xs = cell_xs - cell_xs[:,0][:,np.newaxis]
        local_cell_ys = cell_ys - cell_ys[:,0][:,np.newaxis]
        self.local_cell_xs = local_cell_xs
        self.local_cell_ys = local_cell_ys

        ### Connections between cells
        #######################################################
        cell_index = fd.Function(V_dg)
        cell_index.dat.data[:] = np.arange(len(cell_index.dat.data))
        v_cr = self.v_cr = fd.TestFunction(V_cr)

        edge_lens = fd.assemble(v_cr('+')*fd.dS + v_cr*fd.ds).dat.data
        self.edge_lens = edge_lens
        e0 = fd.assemble((v_cr*cell_index)('+')*fd.dS + (v_cr*cell_index)*fd.ds)
        e1 = fd.assemble((v_cr*cell_index)('-')*fd.dS + (v_cr*cell_index)*fd.ds)
        e0 = np.rint(e0.dat.data / edge_lens).astype(int)
        e1 = np.rint(e1.dat.data / edge_lens).astype(int)
        self.edges = np.c_[e0, e1]


    # Get average of f over an edge
    def get_avg(self, f):
        vals = fd.assemble(fd.avg(f*self.v_cr)*fd.dS + f*self.v_cr*fd.ds).dat.data
        vals /= self.edge_lens
        return vals
    

    # Get the jump of f over an edge
    def get_jump(self, f):
        x = f*self.v_cr
        vals = fd.assemble((x('+') - x('-'))*fd.dS).dat.data
        vals /= self.edge_lens
        return vals