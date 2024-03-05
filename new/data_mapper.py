import numpy as np
import firedrake as fd
import os
os.environ['OMP_NUM_THREADS'] = '1'

class DataMapper:

    def __init__(self, mesh):
        self.mesh = mesh
        V_cg = fd.FunctionSpace(mesh, 'CG', 1)
        V_dg = fd.FunctionSpace(mesh, 'DG', 0)
        V_rt = fd.FunctionSpace(mesh, 'RT', 1)
        V_cr = fd.FunctionSpace(mesh, 'CR', 1)

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

        # Cell midpoints
        x_dg = fd.interpolate(x[0], V_dg)
        y_dg = fd.interpolate(x[1], V_dg)
        print(x_dg.dat.data)
        quit()

        def get_edge_map():
            ex0 = 0.5*self.xs[edges].sum(axis=1)
            ey0 = 0.5*self.ys[edges].sum(axis=1)

            self.ei0 = np.lexsort((ex0, ey0)) 
            self.ei1 = np.lexsort((x_cr, y_cr))

        get_edge_map()
        edges1 = np.zeros_like(edges)
        edges1[self.ei1] = self.edges[self.ei0]
        self.edges = edges1

        # Velocity direction for RT test function
        v_rt = fd.TestFunction(V_rt)
        self.vx0 = fd.assemble(v_rt[0]('+')*fd.dS + v_rt[0]*fd.ds).dat.data
        self.vy0 = fd.assemble(v_rt[1]('+')*fd.dS + v_rt[1]*fd.ds).dat.data
        self.v0 = np.c_[self.vx0, self.vy0]
        self.vx1 = fd.assemble(v_rt[0]('-')*fd.dS + v_rt[0]*fd.ds).dat.data
        self.vy1 = fd.assemble(v_rt[1]('-')*fd.dS + v_rt[0]*fd.ds).dat.data
        self.v1 = np.c_[self.vx1, self.vy1]

        # Normal vectors
        nhat = fd.FacetNormal(mesh)
        v_cr = fd.TestFunction(V_cr)
        self.n0 = fd.assemble((nhat[0]*v_cr)('+')*fd.dS + nhat[0]*v_cr*fd.ds).dat.data
        self.n1 = fd.assemble((nhat[1]*v_cr)('+')*fd.dS + nhat[1]*v_cr*fd.ds).dat.data
        self.n = np.c_[self.n0, self.n1]

        # Edge lengths
        self.edge_lens = fd.assemble(v_cr('+')*fd.dS + v_cr*fd.ds).dat.data
        self.v_cr = v_cr
        
        # Edge to cell map
        v_dg = fd.Function(V_dg)
        v_dg.dat.data[:] = np.arange(len(v_dg.dat.data))
        self.c0 = fd.assemble((v_dg*v_cr)('+')*fd.dS + v_dg*v_cr*fd.ds).dat.data
        self.c1 = fd.assemble((v_dg*v_cr)('-')*fd.dS + v_dg*v_cr*fd.ds).dat.data
        self.c0 /= self.edge_lens
        self.c1 /= self.edge_lens

       
        

      

    def get_avg(self, f):
        vals = fd.assemble(fd.avg(f*self.v_cr)*fd.dS + f*self.v_cr*fd.ds).dat.data
        vals /= self.edge_lens
        return vals
    
    def get_jump(self, f):
        x = f*self.v_cr
        vals = fd.assemble((x('+') - x('-'))*fd.dS).dat.data
        vals /= self.edge_lens
        return vals