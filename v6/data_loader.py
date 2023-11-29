import torch
import numpy as np
from torch_geometric.data import Data

class GraphDataLoader(object):
    
    def __init__(self, normalize = True):
        super(GraphDataLoader, self).__init__()
        self.training_data = []
        self.test_data = []

        x_std = torch.tensor([
            26.053,
            78.630,
            43.570,
            0.707,
            0.707,
            0.0443
        ])

        y_std = 3.677

        for i in range(40):
            print(f'Loading Datastet {i}')
            for j in range(140):
                g = torch.load(f'data/g_{i}_{j}.pt')
                g.num_nodes = len(g.pos)

                if normalize:
                    g.edge_attr /= x_std[np.newaxis,:]
                    g.y /= y_std

                if i <= 30:
                    self.training_data.append(g)
                else:
                    self.test_data.append(g)


