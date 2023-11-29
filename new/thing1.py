import torch
from model.simulator import Simulator
from graph_data_loader import GraphDataLoader
from torch_geometric.loader import DataLoader
import torch 
import matplotlib.pyplot as plt
import numpy as np

loader = GraphDataLoader()
train_data = loader.training_data
test_data = loader.test_data

train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
simulator = Simulator(message_passing_num=10, edge_input_size=11, device=device)


for batch_index, graph in enumerate(train_loader):

    if graph.edge_index.max() > 2882303:
        print(graph)

        print(graph.edge_index.min(), graph.edge_index.max())

        edges = graph.edge_index
        coords = graph.pos.numpy()

        plt.scatter(graph.pos[:,0], graph.pos[:,1])
        plt.show()

        dx = coords[:,0][edges[:,0]] - coords[:,0][edges[:,1]] 
        dy = coords[:,1][edges[:,0]] - coords[:,1][edges[:,1]] 
        dmag = np.sqrt(dx**2 + dy**2)
        dx /= dmag
        dy /= dmag

        plt.quiver(coords[:,0][edges[:,0]], coords[:,1][edges[:,0]], -dx, -dy, scale=1./dmag, scale_units='xy')
        #plt.set_aspect('equal')
        plt.gca().set_aspect('equal')
        plt.show()
    #graph = graph.cuda()
    #y = simulator(graph)
    #print(y)
    #quit()