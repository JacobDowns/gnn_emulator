import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
from data_mapper import DataMapper

for i in range(25):
    file_name = f'/home/jake/ManuscriptCode/examples/hybrid_runs/results/{i}/output.h5'

    with fd.CheckpointFile(file_name, 'r') as afile:

        mesh = afile.load_mesh()
        d = DataMapper(mesh)

        coords = d.get_coords()
        edges = d.get_edges()
        normals = d.get_normals()

        """
        plt.scatter(coords[:,0], coords[:,1])

        for i in range(len(edges)):
            e = edges[i]
            m = edge_coords[i]
            n = 0.05*normals[i]

        

            cx = coords[:,0][e]
            cy = coords[:,1][e]

            plt.plot([m[0], m[0] + n[0]], [m[1], m[1] + n[1]], 'r-')
            plt.plot(cx, cy, 'k-')
 
        plt.gca().set_aspect('equal')
        plt.show()
        """
        V_rt = d.V_rt
        V_dg = d.V_dg
        V_cr = d.V_cr
        u_rt = fd.Function(V_rt)

        B_dg = fd.Function(V_dg)
        H_dg = fd.Function(V_dg)
        S_dg = fd.Function(V_dg)
        beta2_dg = fd.Function(V_dg)

        jS_cr = []
        jB_cr = []
        jbeta2_cr = []
        jH_cr = []
        beta2s_cr = []
        Hs_cr = []
        us_rt = []

        B_dg.interpolate(afile.load_function(mesh, 'B'))

        for j in range(1,160,1):
            print(i, j)
            H_dg.interpolate(afile.load_function(mesh, 'H0', idx=j))
            S_dg.interpolate(B_dg+H_dg)
            beta2_dg.interpolate(afile.load_function(mesh, 'beta2', idx=j))
            
            jump_S_cr = d.get_jump(S_dg)
            jump_B_cr = d.get_jump(B_dg)
            jump_beta2_cr = d.get_jump(B_dg)
            jump_H_cr = d.get_jump(B_dg)

            H_cr = fd.project(H_dg, d.V_cr)
            beta2_cr = fd.project(beta2_dg, d.V_cr)
            jS_cr.append(jump_S_cr)
            jB_cr.append(jump_B_cr)
            jbeta2_cr.append(jump_beta2_cr)
            jH_cr.append(jump_H_cr)

            beta2s_cr.append(beta2_cr.dat.data)
            Hs_cr.append(H_cr.dat.data)
            
            u = afile.load_function(mesh, 'U', idx=j)
            u_rt.interpolate(fd.project(u, V_rt))
            us_rt.append(u_rt.dat.data)

        jS_cr = np.array(jS_cr, np.float32)
        jB_cr = np.array(jB_cr, np.float32)
        Hs_cr = np.array(Hs_cr, np.float32)
        beta2s_cr = np.array(beta2s_cr, np.float32)
        us_rt = np.array(us_rt, dtype=np.float32)

        edge_features = np.stack([jS_cr, jH_cr, Hs_cr, beta2s_cr*Hs_cr, us_rt]).transpose((1,2,0))
        np.save(f'data/edge_features_{i}.npy', edge_features)
        np.save(f'data/coords_{i}', coords)
        np.save(f'data/edges_{i}.npy', edges)
        np.save(f'data/normals_{i}.npy', normals)


            
