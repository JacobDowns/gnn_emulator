import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from model.simulator import Simulator
from firedrake_data_loader import FDDataLoader
import firedrake as fd
dataset_dir = "data/"
batch_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=10, node_input_size=4, edge_input_size=7, device=device)
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
simulator.load_checkpoint()

def test(model:Simulator, test_loader, fd_loader):

    model.eval()
    test_error = 0.
    n = 0

    v_mod = fd.Function(fd_loader.V_rt)
    v_obs = fd.Function(fd_loader.V_rt)

    v_mod_file = fd.File(f'test/v_mod.pvd')
    v_obs_file = fd.File(f'test/v_obs.pvd')


    with torch.no_grad():
        for batch_index, graph in enumerate(test_loader):

            print(batch_index)
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

            #plt.scatter(mx, my, c=edge_features[0:n_edge,0])
            #plt.colorbar()
            #plt.show()

            
            graph = graph.cuda()
            out = model(graph)
            out = out.cpu().numpy()[0:n_edge]

            v_mod.dat.data[:] = out
            v_obs.dat.data[:] = z


            if batch_index % 2 == 0:
                v_mod_file.write(v_mod, idx=batch_index)
                v_obs_file.write(v_obs, idx=batch_index)

            """
            plt.subplot(2,1,1)
            plt.scatter(mx, my, c=z, s=2, vmin=z.min(), vmax=z.max())
            plt.colorbar()

            plt.subplot(2,1,2)
            plt.scatter(mx, my, c=out, s=2, vmin=z.min(), vmax=z.max())
            plt.colorbar()

            plt.show()
            """

            #errors = (out - graph.y)**2
            #loss = torch.mean(errors).item()
            #test_error += loss
            n += 1
        print('Test Error: ', test_error / n)


if __name__ == '__main__':
   
    
    file_name = f'/home/jake/ManuscriptCode/examples/hybrid_runs/results/23/output.h5'      
    fd_loader = FDDataLoader(file_name)
    graphs = fd_loader.get_graphs()
    test_loader = DataLoader(dataset=graphs, batch_size=batch_size, shuffle = False)
     
    test(simulator, test_loader, fd_loader)
