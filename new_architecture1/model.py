import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch_scatter import scatter_add
from torch_geometric.data import Data


def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_size)
    )
    if lay_norm: return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    return module


# Encodes edge + node features in a graph into a single set of node features
class EncoderBlock(nn.Module):

    def __init__(self, edge_input_size=32, node_input_size=32, hidden_size=64):
        super(EncoderBlock, self).__init__()
        self.mlp0 = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.mlp1 = build_mlp(hidden_size+node_input_size, hidden_size, hidden_size)
    
    def forward(self, graph):

        edges = graph.edge_index
        x_edge = graph.x_edge
        x_node = graph.x_node

        x_edge1 = self.mlp0(x_edge)
        x_node1 = torch.zeros((graph.num_nodes, x_edge1.shape[1]), device=x_edge.device)

        scatter_add(x_edge1, edges[:,0], dim=0, out=x_node1)
        scatter_add(x_edge1, edges[:,1], dim=0, out=x_node1)

        x_node2 = [
            x_node,
            x_node1
        ]

        x_node2 = torch.cat(x_node2, dim=1)
        x_node3 = self.mlp1(x_node2)

        return Data(x_node = x_node3, x_edge = x_edge1, edge_index=edges)
    

# Chebyshev convolution block that maps node features to node features
class GnBlock(nn.Module):
    def __init__(self, node_input_size=64, K=5):
        super(GnBlock, self).__init__()
        self.conv = ChebConv(in_channels=node_input_size, out_channels=node_input_size, K=K, bias=True)

    def forward(self, graph):
        x = graph.x_node
        edges = graph.edge_index

        x1 = self.conv(x, edges.T)
        return Data(x_node = x1, edge_index=edges)

# In a GN block all information is encoded in nodes. 
# Here we break it into node and edge information.
class GnDecoder(nn.Module):
    def __init__(self, node_input_size=64, hidden_size=64):
        super(GnDecoder, self).__init__()
        
        self.mlp0 = build_mlp(node_input_size, hidden_size, hidden_size)
        self.mlp1 = build_mlp(2*node_input_size, hidden_size, hidden_size)

    def forward(self, graph):
        # Run node features through MLP then create edge features.

        x_node = graph.x_node
        edges = graph.edge_index
        senders_idx, receivers_idx = edges.T

        x_node1 = self.mlp0(x_node)

        senders_attr = x_node1[senders_idx]
        receivers_attr = x_node1[receivers_idx]

        x_edge1 = []
        x_edge1.append(senders_attr)
        x_edge1.append(receivers_attr)

        x_edge1 = torch.cat(x_edge1, dim=1)
        x_edge2 = self.mlp1(x_edge1)

        return Data(x_node=x_node1, x_edge=x_edge2, edge_index=graph.edge_index)

# Decodes edge features of a graph
class EdgeDecoder(nn.Module):
    def __init__(self, node_input_size=64, edge_input_size=64, edge_output_size=64):
        super(EdgeDecoder, self).__init__()
        self.mlp = build_mlp(2*node_input_size + edge_input_size, edge_output_size, edge_output_size)

    def forward(self, graph):
        x_node = graph.x_node
        x_edge = graph.x_edge
        edges = graph.edge_index
        senders_idx, receivers_idx = edges.T

        senders_attr = x_node[senders_idx]
        receivers_attr = x_node[receivers_idx]

        x_edge1 = []
        x_edge1.append(senders_attr)
        x_edge1.append(receivers_attr)
        x_edge1.append(x_edge)

        x_edge1 = torch.cat(x_edge1, dim=1)
        x_edge2 = self.mlp(x_edge1)

        return Data(x_node=x_node, x_edge=x_edge2, edge_index=graph.edge_index)


class GNN(nn.Module):
    def __init__(self, edge_input_size=32, node_input_size=32, edge_output_size=32, hidden_size=64, K=5):
        super(GNN, self).__init__()

        self.encoder = EncoderBlock(edge_input_size, node_input_size, hidden_size)

        self.GnBlock0 = GnBlock(hidden_size, K)
        self.GnBlock1 = GnBlock(hidden_size, K)
        self.GnBlock2 = GnBlock(hidden_size, K)

        self.d0 = GnDecoder(64, 64)
        self.d1 = GnDecoder(64, 64)
        self.d2 = GnDecoder(64, 64)


    def forward(self, graph):
        g0 = self.encoder(graph)

        g1 = self.GnBlock0(g0)
        g1.x_edge = g0.x_edge
        g1 = self.EdgeBlock0(g1)
     
        g2 = self.GnBlock1(g1)
        g2.x_edge = g1.x_edge
        g2 = self.EdgeBlock1(g2)
        
        g3 = self.GnBlock2(g2)
        g3.x_edge = g2.x_edge
        g3 = self.EdgeBlock2(g3)

        return g3.x_edge