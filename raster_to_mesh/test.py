import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandamesh as pm
from matplotlib.tri import Triangulation
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import LinearNDInterpolator
import xarray as xr
import shapely.geometry as sg
from vel import *


def load_data(cluster):
    store = xr.open_zarr(f'glaciers.zarr/{cluster}')
    x = store['x'].data
    y = store['y'].data

    # This is a mask that has ones on glacier pixels and 0 elsewhere
    mask = store['mask'].data
    # Glacier surface elevation
    z = store['dem'].data
    # Glacier Thickness
    h = np.nan_to_num(store['H'].data)
    # h = np.nan_to_num(store['H_millan'].data)
    h*= mask 
    # Get bed elevation
    bed = z - h
    # Here's some made up, elevation dependent SMB field
    adot = (z - np.mean(z))*5e-3 
    # Dimensions of the raster domain in meters
    dx = x.max() - x.min()
    dy = y.max() - y.min()

    return x, y, mask, z, h, bed, adot, dx, dy

def call_vel(num_glaciers = 5):
    
    # Pick out 5 random glaciers to interpolate onto meshes
    np.random.seed(150)
    stats = pd.read_csv('selected_clusters.csv')
    indexes = np.logical_and(stats['area'] > 25., stats['area'] < 1000.)
    stats = stats[indexes]
    js = np.random.choice(np.arange(len(stats)), num_glaciers, replace=False)
    stats = stats.iloc[js]

    for i, row in stats.iterrows():
        
        cluster = int(row['cluster'])

        # Load data 
        x, y, mask, z, h, bed, adot, dx, dy = load_data(cluster)
        

        #### COMPUTE VELOCITY & traction_squared

        v, tn = compute_vel_from_mesh(z, adot, h)
        # plot_vel(tn, v, cluster)


        ### Create a mesh
        ########################################################

        polygon = sg.Polygon(
        [
            [0.0, 0.0],
            [dx, 0.0],
            [dx, dy],
            [0.0, dy],
        ]
        )
        
        gdf = gpd.GeoDataFrame(geometry=[polygon])
        # 200 m resolution
        gdf["cellsize"] = 200.

        mesher = pm.TriangleMesher(gdf)
        vertices, triangles = mesher.generate()
        triang = Triangulation(vertices[:,0], vertices[:,1], triangles)

        # Shift the domain so the bottom left corner is at the origin
        # This is just for convenience. 
        x -= x.min()
        y -= y.min()
        xx, yy = np.meshgrid(x, y)

        
        #### Interpolate raster data onto the mesh
        ########################################################

        points = np.c_[xx.flatten(), yy.flatten()]

        h_interp = LinearNDInterpolator(points, h.flatten())
        b_interp = LinearNDInterpolator(points, bed.flatten())
        adot_interp = LinearNDInterpolator(points, adot.flatten())
        s_interp = LinearNDInterpolator(points,z.flatten())

        mask_interp = LinearNDInterpolator(points, mask.flatten())
        traction_interp = LinearNDInterpolator(points, tn.flatten())

        # Raster data interpolated onto mesh vertices
        H_mesh = h_interp(vertices[:,0], vertices[:,1])
        B_mesh = b_interp(vertices[:,0], vertices[:,1])
        adot_mesh = adot_interp(vertices[:,0], vertices[:,1])
        S_mesh = s_interp(vertices[:,0], vertices[:,1])

        mask_mesh = mask_interp(vertices[:,0], vertices[:,1])
        traction_mesh = traction_interp(vertices[:,0], vertices[:,1])

        
        #plotter(triang, H_mesh, B_mesh, adot_mesh)
       
        # Prep mesh inputs needed for NN: X, X_edge, coords, faces, edges

       
        # Make edges
        edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        # Remove duplicate edges
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        # Make elements for X_edge
        dx, dy, dmag, dS = make_xedge(edges,points,S_mesh)

        # Create  Inputs for Neural Network 
        big_X = np.column_stack((H_mesh, traction_mesh, mask_mesh))
        X_edge = np.column_stack((dx,dy,dmag,dS))
        faces = triangles

        #dS = S_mesh[edges[:,0]] - S_mesh[edges[:,1]]

        cx = 0.5*vertices[:,0][edges].sum(axis=1)
        cy = 0.5*vertices[:,1][edges].sum(axis=1)
        plt.scatter(cx, cy, c = dS)
        plt.colorbar()
        plt.show()
        quit()

    return S_mesh, edges, vertices, points, big_X, X_edge, faces, cluster, v, tn
        

def save_results(cluster,X_edge,coords,faces,edges,X, folder_name):
    
    file_name = f"{folder_name}/X_{cluster}.npy"
    np.save(file_name, X)
    file_name = f"{folder_name}/X_edge_{cluster}.npy"
    np.save(file_name,X_edge)

    file_name = f"{folder_name}/coords_{cluster}.npy"
    np.save(file_name,coords)

    file_name = f"{folder_name}/faces_{cluster}.npy"
    print(file_name)
    np.save(file_name, faces)

    file_name = f"{folder_name}/edges_{cluster}.npy"
    np.save(file_name,edges)

    print("Files all saved for cluster:", cluster)

def make_xedge(edges, coords,smesh):
    dmag = np.zeros(edges.shape[0])
    dS = np.zeros(edges.shape[0])
    dx = np.zeros(edges.shape[0])
    dy = np.zeros(edges.shape[0])

    for i, edge_indices in enumerate(edges):        
        
        # Calculate dx and dy
        dx[i] = coords[edge_indices[0],0] - coords[edge_indices[1],0]
        dy[i] = coords[edge_indices[0],1] - coords[edge_indices[1],1]

        #Calculate dmag 
        dmag[i] = np.sqrt(dx[i]**2+dy[i]**2)

        dS[i]    = smesh[edge_indices[0]] - smesh[edge_indices[1]]
        #  max = 0
        

    return dx, dy, dmag, dS

def load_sample_data():

    data_set  = 1
    time_step = 1
    

    E = np.load(f"data/edges_{data_set}.npy")
    C = np.load(f"data/coords_{data_set}.npy")
    X = np.load(f"data/X_edge_{data_set}.npy")
    X = np.load(f"data/X_{data_set}.npy")
    X_edge = np.load(f"data/X_edge_{data_set}.npy")
    # read edge values from file
    dx_f   = X_edge[time_step,:,0]
    dy_f   = X_edge[time_step,:,1]
    dmag_f = X_edge[time_step,:,2]
    dS_f   = X_edge[time_step,:,3]
    n,m,o = X_edge.shape
    return dx_f, dy_f, dmag_f, dS_f

def check_mesh(dx,dy,dmag,ds):
    tolerance = 1e-6
    dx_f, dy_f, dmag_f, dS_f = load_sample_data()
    diff_list = []
    diff_dx = dx-dx_f
    diff_dy = dy-dy_f
    diff_dmag = dmag-dmag_f
    diff_ds = ds-dS_f
    diff_list.extend([diff_dx,diff_dy,diff_dmag,diff_ds])
    l = np.abs(diff_list) > tolerance
    sumlist = []
    sumlist.extend([np.sum(l[:,0]),np.sum(l[:,1]),np.sum(l[:,2]),np.sum(l[:,3])])
    return diff_list, sumlist

def plot_vel(tn, v, cluster):
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(np.log(tn), origin='lower')
    axes[0].set_title('log t')
    axes[1].imshow(v, origin='lower')
    axes[1].set_title('v')
    plt.suptitle("cluster "+ str(cluster))
    plt.show()


def plotter(triang, H_mesh, B_mesh, adot_mesh):
    # Plot the mesh
    plt.triplot(triang)
    plt.show()

    # Plot thickness
    plt.tricontourf(triang, H_mesh, levels=256)
    plt.colorbar()
    plt.show()

    # Plot Bed
    plt.tricontourf(triang, B_mesh, levels=256)
    plt.colorbar()
    plt.show()

    # Plot SMB
    plt.tricontourf(triang, adot_mesh, levels=256)
    plt.colorbar()
    plt.show()


def make_mesh_inputs(H_mesh, traction_mesh,mask_mesh, vx_mesh, vy_mesh, dx_mesh, dy_mesh,dmag_mesh,dS_mesh,x_mesh,y_mesh, faces):
    big_X = np.column_stack(H_mesh, traction_mesh, mask_mesh)
    
    
    coords = np.column_stack(x_mesh,y_mesh)

    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    # Remove duplicate edges
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    # return big_X, big_Y, X_edge, coords, edges

S_mesh, edges, vertices, points, big_X,X_edge, faces, cluster, v, tn = call_vel()
print(S_mesh)