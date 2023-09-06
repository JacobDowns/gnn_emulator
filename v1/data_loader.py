import torch
import numpy as np
from torch_geometric.data import Data

class GraphDataLoader(object):
    
    def __init__(self, cull=True):
        super(GraphDataLoader, self).__init__()
        self.cull = True
        self.data = []

        for i in range(25):
            self.load_dataset(i)


    def load_dataset(self, i):
        print(i)

        ### Load emulator inputs / outputs

        # DG0 cell midpoints
        cell_coords = np.load(f'data/cell_coords_{i}.npy')
        # Connections between DG0 cells (common edges)
        cell_connections = np.load(f'data/cell_connections_{i}.npy')
        # Cell input (B, H, beta2, N)
        X_cells = np.load(f'data/X_cells_{i}.npy') 
        # Edge variables (edge_len, edge_mid_x, edge_mid_y, edge_normal_x, edge_normal_y)   
        X_edges = np.load(f'data/X_edges_{i}.npy')
        # Velocity output in RT space
        Y_edges = np.load(f'data/Y_edges_{i}.npy') 
        
        for j in range(0, X_cells.shape[0]):
            X_cells_j = X_cells[j]
            Y_edges_j = Y_edges[j]

            # Create a mask where any cell with positive ice thickness or
            # adjacent to a node with positive ice thickness has value 1. 
            mask = np.zeros_like(X_cells_j[:,1])
            mask[X_cells_j[:,1] > 1e-3] = 1.
            mask[cell_connections[np.min(mask[cell_connections], axis=1) > 0]]
                        
            def cull_graph(coords, edges, mask, X_vertex_j, X_edge_j, Y_edge_j):
                indexes = np.where(mask == 1.)[0].astype(int)
                mapping = np.zeros_like(mask) + 1e16
                mapping[indexes] = np.arange(len(indexes))
                X_vertex_j = X_vertex_j[indexes]
                coords = coords[indexes]
                
                edges = mapping[edges]
                indexes = np.max(edges, axis=1) < 1e16
                edges = edges[indexes].astype(int)
                X_edge_j = X_edge_j[indexes]
                Y_edge_j = Y_edge_j[indexes]
                
                return coords, edges, X_vertex_j, X_edge_j, Y_edge_j

            
            cell_coords_c, cell_connections_c, X_cells_c, X_edges_c, Y_edges_c = cull_graph(
                cell_coords,
                cell_connections,
                mask,
                X_cells_j,
                X_edges,
                Y_edges_j
            )

            ### Construct edge features

            # We want to have multidirectional edges
            cell_connections_c = np.concatenate([
                cell_connections_c,
                cell_connections_c[:,::-1]
            ])

            # Same edge features except flipped normals
            X_edges_c = np.concatenate([
                X_edges_c,
                -X_edges_c
            ])

            # Cell midpoint offsets
            dx = cell_coords[:,0][cell_connections_c]
            dx = dx[:,1] - dx[:,0]
            dy = cell_coords[:,0][cell_connections_c]
            dy = dy[:,1] - dy[:,0]
            dd = np.sqrt(dx**2 + dy**2)

            # Surface and bed differences over edges
            B = X_cells_c[:,0]
            H = X_cells_c[:,1]
            S = B + H
            dS = S[cell_connections_c[:,0]] - S[cell_connections_c[:,1]]
            dB = B[cell_connections_c[:,0]] - B[cell_connections_c[:,1]]



            # All edge features (dx, dy, nx, ny, dB, dS)
            X_edge = np.concatenate([
                dx[:,np.newaxis], 
                dy[:,np.newaxis],
                dd[:,np.newaxis],
                X_edges_c, 
                dB[:,np.newaxis], 
                dS[:,np.newaxis]], 
                axis=1
            )
            
            # Cell features will be (H, beta2, N) on reduced graph
            X = X_cells_c[:,[1,2]]

            # Edge output features
            Y = np.concatenate([Y_edges_c, -Y_edges_c])

            # Normalization
            X_scale = [0.1302583, 0.16530661]
            X_edge_scale = [8.427722, 8.427722, 8.117857, 1, 1, 0.02819121, 0.02354354]
            Y_scale = 0.17659752
            X /= X_scale
            X_edge /= X_edge_scale
            Y /= Y_scale

            data_j = Data(
                x = torch.tensor(X, dtype=torch.float32),
                y_edge = torch.tensor(Y, dtype=torch.float32),
                edge_index = torch.tensor(cell_connections_c, dtype=torch.long).T,
                edge_attr = torch.tensor(X_edge, dtype=torch.float32),
                pos = cell_coords_c
            )

            self.data.append(data_j)
