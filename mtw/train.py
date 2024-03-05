from model.simulator import Simulator
import torch
from simulation_loader import SimulatorDataset
from torch.utils.data.dataset import Subset
import numpy as np
import random

dataset_dir = "data/"
batch_size = 1
save_epoch = 5
epochs = 301

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
simulator = Simulator(message_passing_num=5, edge_input_size=9, device=device)
simulator.load_checkpoint()
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
#vel_loss = VelocityLoss().apply

def train(model:Simulator, train_data, test_data, optimizer):

    for ep in range(epochs):
        print('Epoch', ep)
        model.train() 
        train_error = 0.
        n = 0

        js = list(range(len(train_data)))
        random.shuffle(js)
        for j in js:
            g, y_obs, sim_loader = train_data[j]


            g = g.cuda()
            y = model(g)
            y = y.cpu()


            errors = (y_obs - y)**2
            loss = torch.mean(errors)


            #loss = vel_loss(Ubar, Udef, Ubar_obs, Udef_obs, sim_loader.loss_integral)
            train_error += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n += 1

        torch.cuda.empty_cache() 
        print('Train error: ', train_error / n)

        if ep % save_epoch == 0:
            model.save_checkpoint()

        if ep % 25 == 0:
            model.eval()
            test_error = 0.
            n = 0
            with torch.no_grad():
                 for j in range(len(test_data)):
                    g, y_obs, sim_loader = train_data[j]

                    g = g.cuda()
                    y = model(g)
                    y = y.cpu()

                    errors = (y_obs - y)**2
                    loss = torch.mean(errors)

                    test_error += loss.item()
                    n += 1

            print('Test error: ', test_error / n)
            torch.cuda.empty_cache() 

if __name__ == '__main__':


    data = SimulatorDataset()

    n = len(data)
    n_test = int(0.1*n)

    test_data = Subset(data, np.arange(0, n_test))
    train_data = Subset(data, np.arange(n_test, n))

    train(simulator, train_data, test_data, optimizer)