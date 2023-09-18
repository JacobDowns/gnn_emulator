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

# Pick out 5 random glaciers to interpolate onto meshes
np.random.seed(150)
stats = pd.read_csv('selected_clusters.csv')
indexes = np.logical_and(stats['area'] > 25., stats['area'] < 1000.)
stats = stats[indexes]
js = np.random.choice(np.arange(len(stats)), 5, replace=False)
stats = stats.iloc[js]

for i, row in stats.iterrows():
    cluster = int(row['cluster'])
    store = xr.open_zarr(f'glaciers.zarr/{cluster}')
    x = store['x'].data
    y = store['y'].data

    # This is a mask that has ones on glacier pixels and 0 elsewhere
    mask = store['mask'].data
    # Glacier surface elevation
    z = store['dem'].data
    # Glacier Thickness
    h = np.nan_to_num(store['H'].data)
    h*= mask 
    # Get bed elevation
    bed = z - h
    # Here's some made up, elevation dependent SMB field
    adot = (z - np.mean(z))*5e-3 
    # Dimensions of the raster domain in meters
    dx = x.max() - x.min()
    dy = y.max() - y.min()

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
    
    # Raster data interpolated onto mesh vertices
    H_mesh = h_interp(vertices[:,0], vertices[:,1])
    B_mesh = b_interp(vertices[:,0], vertices[:,1])
    adot_mesh = adot_interp(vertices[:,0], vertices[:,1])
    
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
