import os
os.environ['OMP_NUM_THREADS'] = '1'
from simulation_loader import SimulationLoader
import numpy as np
import matplotlib.pyplot as plt

i = 4
d = SimulationLoader(i)
d.load_features_from_arrays()

j = 90
x_cell = d.X_cell[j]
x_edge = d.X_edge[j]
y_edge = d.Y[j]


print(x_cell.shape)
print(x_edge.shape)
print(y_edge.shape)
quit()

xm = d.mapper.cell_mid_xs
ym = d.mapper.cell_mid_ys

xm = xm[d.edges].mean(axis=1)
ym = ym[d.edges].mean(axis=1)

plt.subplot(2,1,1)
plt.scatter(xm, ym, c=x_edge[:,0])

plt.subplot(2,1,2)
plt.scatter(xm, ym, c=x_edge[:,1]-x_edge[:,0])
plt.show()
