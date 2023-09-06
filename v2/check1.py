import numpy as np
import matplotlib.pyplot as plt
from data_loader import GraphDataLoader

loader = GraphDataLoader()
data = loader.data

for i in range(len(data)):
        d = data[i]
        coords = d.pos
        edges = d.edge_index.T

        x_edge = d.edge_attr
        y_vertex = d.y_vertex
        x_vertex = d.x

        print(x_edge.shape)
        print(x_vertex.shape)
        print(y_vertex.shape)
        print(edges.shape)
        print(coords.shape)
        
        x = 0.5*(coords[:,0][edges].sum(axis=1))
        y = 0.5*(coords[:,1][edges].sum(axis=1))

        
        plt.scatter(x, y, c=x_edge[:,6], s=2)
        plt.colorbar()
        plt.show()
        
        #plt.show()
        plt.scatter(coords[:,0], coords[:,1], c=x_vertex[:,1], s=2)
        plt.colorbar()
        plt.show()

            
