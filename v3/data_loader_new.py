import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from torch_scatter import scatter_sum, scatter_mean
#import torch_geometric.transforms as T

class GraphDataLoader(object):
    
    def __init__(self, cull=True):
        super(GraphDataLoader, self).__init__()
        self.cull = True
        self.data = []

        for i in range(25):
            self.load_dataset(i)

    def load_dataset(self, i):
        print(i) 

        # Load graph data
        coords = np.load(f'data/coords_{i}.npy')
        edges = np.load(f'data/edges_{i}.npy')
        normals = np.load(f'data/normals_{i}.npy')

        # edge features (jump S, jump B, avg H, avg beta2, u_rt)
        edge_features = np.load(f'data/edge_features_{i}.npy')
    
        for j in range(0, edge_features.shape[0]):
            edge_features_j = edge_features[j]
          

            # This will create a mask where every node within h "hops" of the ice mask ice 1 and everything else i s 0
            # Basically just gets a region around the glacier
            def create_mask(hops = 2):
                H_e = torch.tensor(edge_features_j[:,2])
                mask0 = scatter_mean(H_e, torch.tensor(edges[:,1], dtype=torch.int64), dim=0, dim_size=len(coords))
                mask1 = scatter_mean(H_e, torch.tensor(edges[:,0], dtype=torch.int64), dim=0, dim_size=len(coords))
                mask = torch.zeros(len(coords))
                mask[mask0 > 1.0001e-3] = 1.
                mask[mask1 > 1.0001e-3] = 1.

                for h in range(hops):

                    prop = 0.5*(mask[edges[:,0]] + mask[edges[:,1]])
                    mask0 = scatter_mean(prop, torch.tensor(edges[:,1], dtype=torch.int64), dim=0, dim_size=len(coords))
                    mask1 = scatter_mean(prop, torch.tensor(edges[:,0], dtype=torch.int64), dim=0, dim_size=len(coords))
                    mask[mask0 > 1.0001e-3] = 1.
                    mask[mask1 > 1.0001e-3] = 1.

                return mask
            

            mask = create_mask(2)
            mask = mask.numpy()*0.+1


            # edge offsets and lengths
                        
            def cull_graph(coords, edges, mask, edge_features, normals):
                indexes = np.where(mask == 1.)[0].astype(int)
                mapping = np.zeros_like(mask) + 1e16
                mapping[indexes] = np.arange(len(indexes))
                coords = coords[indexes]
                
                edges = mapping[edges]
                indexes = np.max(edges, axis=1) < 1e16
                edges = edges[indexes].astype(int)
                edge_features = edge_features[indexes]
                normals = normals[indexes]

                return coords, edges, edge_features, normals

            coords_c, edges_c, edge_features_c, normals_c = cull_graph(
                coords,
                edges,
                mask,
                edge_features_j,
                normals
            )

            """
            plt.scatter(coords_c[:,0], coords_c[:,1])
            for k in range(len(edges_c)):
                e = edges_c[k]
                cx = coords_c[:,0][e]
                cy = coords_c[:,1][e]
                plt.plot(cx, cy, 'k-')
            """

            y_c = edge_features_c[:,-1]
            edge_features_c = np.concatenate([edge_features_c[:,:-1], normals_c], axis=1)

            # Add edges going the opposite way
            edges_c = np.concatenate([edges_c, edges_c[:,::-1]])

            dx = coords_c[:,0][edges_c]
            dy = coords_c[:,1][edges_c]
            dx = dx[:,1] - dx[:,0]
            dy = dy[:,1] - dy[:,0]
            dd = np.sqrt(dx**2 + dy**2)
            # Edge features are the same for the edges going the opposite way except, jumps and normals are flipped
            edge_features_c = np.concatenate([edge_features_c, edge_features_c*[-1,-1,1,1,-1,-1]])
            # Similarly, rt coefficients are flipped
            y_c = np.concatenate([y_c, -y_c])
            # Add vertex offsets and edge lengths as features
            edge_features_c = np.concatenate([edge_features_c, dd[:,np.newaxis]], axis=1)
        
            data_j = Data(
                y = torch.tensor(y_c, dtype=torch.float32),
                edge_index = torch.tensor(edges_c, dtype=torch.long).T,
                edge_attr = torch.tensor(edge_features_c, dtype=torch.float32),
                pos = coords_c
            )

            # Normalization
            data_j.edge_attr /= torch.tensor([0.0247, 0.0284, 0.1312, 0.0738, 0.707, 0.707, 0.0444])
            data_j.y = data_j.y / torch.tensor(0.0458)

            vertex_features_c = scatter_mean(data_j.edge_attr[:,0:4], data_j.edge_index[1], dim=0, dim_size=len(coords_c))
            data_j.x = vertex_features_c

            
            self.data.append(data_j)
            #print(data_j)