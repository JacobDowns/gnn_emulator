from model.simulator import Simulator
import torch
from simulation_loader import SimulatorDataset
from torch.utils.data.dataset import Subset
from velocity_loss import VelocityLoss
import numpy as np
import random

dataset_dir = "data/"
batch_size = 1

print_batch = 10
save_epoch = 10
epochs = 301

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=6, edge_input_size=10, device=device)
#simulator.load_checkpoint()
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
vel_loss = VelocityLoss().apply

pytorch_total_params = sum(p.numel() for p in simulator.parameters() if p.requires_grad)


print('Optimizer initialized')

def train(model:Simulator, train_data, test_data, optimizer):

    for ep in range(epochs):
        print('Epoch', ep)
        model.train() 
        train_error = 0.
        n = 0

        js = list(range(len(train_data)))
        random.shuffle(js)
        for j in js:
            data = train_data[j]

            sim_loader = data[1]
            Ubar_obs = data[0]['Ubar']
            H = data[0]['H']

            graph = sim_loader.get_graph(data[0])
            graph = graph.cuda()

            Ubar_obs = torch.tensor(Ubar_obs.dat.data[:], dtype=torch.float32)
            H = torch.tensor(H.dat.data[:], dtype=torch.float32)

            # Get Velocity from GNN
            Ubar = model(graph).cpu()
            loss = vel_loss(Ubar, Ubar_obs, H, sim_loader.loss_integral)

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
                 for j in range(len(test_data)):
                    data = test_data[j]

                    sim_loader = data[1]
                    Ubar_obs = data[0]['Ubar']
                    H = data[0]['H']

                    graph = sim_loader.get_graph(data[0])
                    graph = graph.cuda()

                    Ubar_obs = torch.tensor(Ubar_obs.dat.data[:], dtype=torch.float32)
                    H = torch.tensor(H.dat.data[:], dtype=torch.float32)

                    # Get Velocity from GNN
                    Ubar = model(graph).cpu()
                    loss = vel_loss(Ubar, Ubar_obs, H, sim_loader.loss_integral)

                    test_error += loss.item()
                    n += 1
            print('Test error: ', test_error / n)
                


if __name__ == '__main__':


    data = SimulatorDataset()

    n = len(data)
    n_test = int(0.1*n)

    test_data = Subset(data, np.arange(0, n_test))
    train_data = Subset(data, np.arange(n_test, n))

    for j in range(len(train_data)):
        data = train_data[j]
        print(len(data[0]['Ubar'].dat.data))

    quit()
    
    print(len(train_data))
    print(len(test_data))

    train(simulator, train_data, test_data, optimizer)
