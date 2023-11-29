import torch
import numpy as np
from torch_geometric.data import Data

class GraphDataLoader(object):
    
    def __init__(self, normalize = True):
        super(GraphDataLoader, self).__init__()
        self.training_data = []
        self.test_data = []

        x_std = torch.tensor([
            25.971,
            8.122,
            83.282,
            44.536,
            .7072,
            .7072,
            0.0443
        ])

        y_std = 16.672

        for i in range(40):
            print(f'Loading Datastet {i}')
            for j in range(120):
                g = torch.load(f'data/g_{i}_{j}.pt')
                g.num_nodes = len(g.pos)

                if normalize:
                    g.edge_attr /= x_std[np.newaxis,:]
                    g.y /= y_std

                if i <= 35:
                    self.training_data.append(g)
                else:
                    self.test_data.append(g)


