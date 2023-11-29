import torch
import numpy as np
from torch_geometric.data import Data

class GraphDataLoader(object):
    
    def __init__(self, normalize = False):
        super(GraphDataLoader, self).__init__()
        self.training_data = []
        self.test_data = []

        
        x_std = torch.tensor([
            0.02533848,
            0.02103466,
            0.11764277,
            0.06610081,
            0.7071,
            0.7071,
            0.04447203
        ])

        y_std = 0.163099

        for i in range(25):
            print(f'Loading Datastet {i}')
            for j in range(160):
                g = torch.load(f'data/g_{i}_{j}.pt')
                g.num_nodes = len(g.pos)

                if normalize:
                    g.edge_attr /= x_std[np.newaxis,:]
                    g.y /= y_std

                if i <= 20:
                    self.training_data.append(g)
                else:
                    self.test_data.append(g)


