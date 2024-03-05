from model.simulator import Simulator
import torch
from simulation_loader import SimulatorDataset
from torch.utils.data.dataset import Subset
from velocity_loss import VelocityLoss
import numpy as np
import random
import firedrake as fd

dataset_dir = "data/"
batch_size = 1
save_epoch = 5
epochs = 301

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
simulator = Simulator(message_passing_num=10, edge_input_size=9, device=device)
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
vel_loss = VelocityLoss().apply


out_mod = fd.File('test/out_mod.pvd')
out_obs = fd.File('test/out_obs.pvd')


def train(model:Simulator, train_data, test_data, optimizer):

    k = 0

    for ep in range(epochs):
        print('Epoch', ep)
        model.train() 
        train_error = 0.
        n = 0
        

        js = list(range(len(train_data)))
        random.shuffle(js)
        for j in js:
            g, y_obs, sim_loader = train_data[j]

            Ubar_obs = y_obs.flatten()
            g = g.cuda()
            y = model(g).cpu()
            Ubar_mod = y.flatten()
            loss = vel_loss(Ubar_mod, Ubar_obs, sim_loader.loss_integral)
            train_error += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n += 1

        print('Train error: ', train_error / n)

        if ep % save_epoch == 0:
            model.save_checkpoint()

        if ep % 15 == 0:
            model.eval()
            test_error = 0.
            n = 0
            with torch.no_grad():
                 for j in range(len(test_data)):
                    g, y_obs, sim_loader = train_data[j]

                    Ubar_obs = y_obs.flatten()
                    g = g.cuda()
                    y = model(g).cpu()
                    Ubar_mod = y.flatten()

                    loss = vel_loss(Ubar_mod, Ubar_obs, sim_loader.loss_integral)
                    
                    if j == 50:
                        out_mod.write(sim_loader.loss_integral.Ubar, idx=k)
                        out_obs.write(sim_loader.loss_integral.Ubar_obs, idx=k)
                        k += 1

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