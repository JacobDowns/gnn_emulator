from model.simulator import Simulator
import torch
import time
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from data_loader import GraphDataLoader
import numpy as np

dataset_dir = "data/"
batch_size = 1

print_batch = 10
save_epoch = 10
epochs = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=15, node_input_size=6, edge_input_size=6, device=device)
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
simulator.load_checkpoint()
print('Optimizer initialized')

def train(model:Simulator, train_loader, test_loader, optimizer):

    for ep in range(epochs):
        print('Epoch', ep)
        model.train() 
        train_error = 0.

        n = 0
        for batch_index, graph in enumerate(train_loader):
            del graph['pos']

            eps0 = torch.tensor(np.random.randn(graph.edge_attr.shape[0])) * (graph.edge_attr[:,0].std() / 20.)
            eps1 = torch.tensor(np.random.randn(graph.edge_attr.shape[0])) * (graph.edge_attr[:,1] / 20.)
            eps2 = torch.tensor(np.random.randn(graph.edge_attr.shape[0])) * (graph.edge_attr[:,2] / 20.)
            
            graph.edge_attr[:,0] += eps0
            graph.edge_attr[:,1] += eps1
            graph.edge_attr[:,2] += eps2
            graph.edge_attr[:,1] = torch.clip(graph.edge_attr[:,1], min=0.)
            graph.edge_attr[:,2] = torch.clip(graph.edge_attr[:,2], min=0.)

            graph = graph.cuda()
            
            y = graph.y
            out = model(graph)

            errors = (out - y)**2
            loss = torch.mean(errors)
            train_error += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n += 1


        print('Train error: ', train_error / n)

        if ep % save_epoch == 0:
            model.save_checkpoint()

        if ep % 25 == 0:
            model.eval()
            test_error = 0.
            n = 0
            with torch.no_grad():
                for batch_index, graph in enumerate(test_loader):
                    del graph['pos']
                    graph = graph.cuda()
                    y = graph.y
                    out = model(graph)
                    errors = (out - y)**2
                    loss = torch.mean(errors)
                    test_error += loss.item()
                    n += 1
                print('Test Error: ', test_error / n)


if __name__ == '__main__':

    loader = GraphDataLoader()
    train_data = loader.training_data
    test_data = loader.test_data

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle = True)
    train(simulator, train_loader, test_loader, optimizer)
