from model.simulator import Simulator
import torch
from torch_geometric.loader import DataLoader
from graph_data_loader import GraphDataLoader

dataset_dir = "data/"
batch_size = 1

print_batch = 10
save_epoch = 10
epochs = 125

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=24, edge_input_size=10, device=device)
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')

def train(model:Simulator, train_loader, test_loader, optimizer):

    for ep in range(epochs):
        print('Epoch', ep)
        model.train() 
        train_error = 0.
        n = 0
        for batch_index, graph in enumerate(train_loader):
            del graph['pos']

            #graph.x_i = torch.normal(mean = graph.x_i, std=0.075*torch.abs(graph.x_i))
            graph = graph.cuda()

            # Area
            a = graph.x_g[:,[1,2]].sum(axis=1)
            y = graph.y
            out = model(graph)

            errors = a*(out - y)**2
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
                    a = graph.x_g[:,[1,2]].sum(axis=1)
                    y = graph.y
                    out = model(graph)
                    errors = a*(out - y)**2
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
