import torch
import numpy as np
from torch_geometric.data import Data

class GraphDataLoader(object):
    
    def __init__(self, normalize = True):
        super(GraphDataLoader, self).__init__()
        self.training_data = []
        self.test_data = []

        x_g_std = torch.tensor([
            0.0443,
            0.0017,
            0.0017,
            0.1608,
            0.1608,
            0.6615,
            0.6573
        ])

        x_i_std = torch.tensor([
            81.8970,
            26.9916,
            25.9940,
            41.8627
        ])

        y_std = 3.7353

        for i in range(40):
            print(f'Loading Datastet {i}')
            for j in range(140):
                g = torch.load(f'data/g_{i}_{j}.pt')
                g.num_nodes = len(g.pos)

                if g.edge_index.max() < 2882303:
                    if normalize:
                        g.x_g /= x_g_std[np.newaxis,:]
                        g.x_i /= x_i_std[np.newaxis,:]
                        g.y /= y_std

                    if i <= 35:
                        self.training_data.append(g)
                    else:
                        self.test_data.append(g)


