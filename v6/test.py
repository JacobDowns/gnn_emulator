import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from model.simulator import Simulator
from data_loader import GraphDataLoader
dataset_dir = "data/"
batch_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=10, node_input_size=4, edge_input_size=7, device=device)
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
simulator.load_checkpoint()

def test(model:Simulator, test_loader):

    model.eval()
    test_error = 0.
    n = 0
    with torch.no_grad():
        for batch_index, graph in enumerate(test_loader):
            pos = graph.pos[0]
            x = pos[:,0]
            y = pos[:,1]
           
            n_edge = int(graph.edge_index.shape[1] / 2)
            edges = graph.edge_index[:,0:n_edge].T
            z = graph.y[0:n_edge]

            mx = x[edges]
            my = y[edges]
            mx = mx.mean(axis=1)
            my = my.mean(axis=1)

            edge_features = graph.edge_attr

            plt.scatter(mx, my, c=edge_features[0:n_edge,0])
            plt.colorbar()
            plt.show()

            
            graph = graph.cuda()
            out = model(graph)
            out = out.cpu().numpy()[0:n_edge]


            plt.subplot(2,1,1)
            plt.scatter(mx, my, c=z, s=2, vmin=z.min(), vmax=z.max())
            plt.colorbar()

            plt.subplot(2,1,2)
            plt.scatter(mx, my, c=out-z.cpu().numpy(), s=2, vmin=z.min(), vmax=z.max())
            plt.colorbar()

            plt.show()

            #errors = (out - graph.y)**2
            #loss = torch.mean(errors).item()
            #test_error += loss
            n += 1
        print('Test Error: ', test_error / n)


if __name__ == '__main__':
   
    loader = GraphDataLoader()
    train_data = loader.training_data
    test_data = loader.test_data

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle = True)
    

    test(simulator, test_loader)
