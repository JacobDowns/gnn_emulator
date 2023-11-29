import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from model.simulator import Simulator
from firedrake_data_loader import FDDataLoader
dataset_dir = "data/"
batch_size = 1


file_name = '/home/jake/ManuscriptCode/examples/hybrid_runs/results/23/output.h5'      
fd_loader = FDDataLoader(file_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Simulator(message_passing_num=8, node_input_size=4, edge_input_size=7, device=device)
model.load_checkpoint()



graph = fd_loader.get_graph(10)
model.eval()

with torch.no_grad():
    pos = graph.pos
    x = pos[:,0]
    y = pos[:,1]

    n_edge = int(graph.edge_index.shape[1] / 2)
    edges = graph.edge_index[:,0:n_edge].T
    mx = x[edges]
    my = y[edges]
    mx = mx.mean(axis=1)
    my = my.mean(axis=1)
    
   
    edges = graph.edge_index[:,0:n_edge].T
    y_target = graph.y[0:n_edge]

    graph = graph.cuda()
    y_model = model(graph)
    y_model = y_model.cpu().numpy()[0:n_edge]

    #_rt = fd.Function(fd_loader.V_rt)


   

    plt.subplot(2,1,1)
    plt.scatter(mx, my, c=y_model, s=2, vmin=y_target.min(), vmax=y_target.max())
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.scatter(mx, my, c=y_target, s=2, vmin=y_target.min(), vmax=y_target.max())
    plt.colorbar()

    plt.show()

    quit()
    plt.subplot(2,1,1)
    plt.scatter(mx, my, c=z, s=2, vmin=z.min(), vmax=z.max())
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.scatter(mx, my, c=out, s=2, vmin=z.min(), vmax=z.max())
    plt.colorbar()

    plt.show()

    #errors = (out - graph.y)**2
    #loss = torch.mean(errors).item()
    #test_error += loss
    n += 1
print('Test Error: ', test_error / n)


