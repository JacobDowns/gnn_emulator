import numpy as np
import matplotlib.pyplot as plt

for i in range(30):

        coords = np.load(f'data/coords_{i}.npy')
        edges = np.load(f'data/edges_{i}.npy')
        # H, beta2, jS, jB, dx, dy, dmag
        X_edge = np.load(f'data/X_edge_{i}.npy')
        # vx, vy
        Y_vertex = np.load(f'data/Y_vertex_{i}.npy')

        print(coords.shape)
        print(edges.shape)
        print(X_edge.shape)
        print(Y_vertex.shape)
        print()

        x = 0.5*(coords[:,0][edges].sum(axis=1))
        y = 0.5*(coords[:,1][edges].sum(axis=1))
                            
        plt.scatter(x, y, c=X_edge[40,:,6])
        plt.colorbar()
        plt.show()
        plt.scatter(coords[:,0], coords[:,1], c=Y_vertex[40,:,0], s=2)
        plt.show()

            
