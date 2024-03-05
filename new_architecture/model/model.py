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


# Encodes a graph with only edge features into one with edge and node features
class EncoderBlock(nn.Module):

    def __init__(self, edge_input_size=32, hidden_size=64):
        super(EncoderBlock, self).__init__()
        self.mlp = build_mlp(edge_input_size, hidden_size, hidden_size)
    
    def forward(self, graph):

        edges = graph.edge_index
        x_edge = graph.x_edge
        x_edge1 = self.mlp(x_edge)
        x_node = torch.zeros((graph.num_nodes, x_edge1.shape[1]), device=x_edge.device)

        scatter_add(x_edge1, edges[:,0], dim=0, out=x_node)
        scatter_add(x_edge1, edges[:,1], dim=0, out=x_node)

        return Data(x_node = x_node, x_edge = x_edge1, edge_index=edges)
    

# Chebyshev convolution block that maps node features to node features
class GnBlock(nn.Module):
    def __init__(self, node_input_size=64, K=5):
        super(GnBlock, self).__init__()
        self.conv = ChebConv(in_channels=node_input_size, out_channels=node_input_size, K=K, bias=True)
        self.mlp = build_mlp(node_input_size, node_input_size, node_input_size)

    def forward(self, graph):
        x = graph.x_node
        edges = graph.edge_index

        x1 = self.conv(x, edges.T)
        x2 = self.mlp(x1)
        return Data(x_node = x2, edge_index=edges)


# Takes in a graph with only nodes and decodes node and edge features
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
    def __init__(self, edge_input_size=13, edge_output_size=3, hidden_size=64, K=5):
        super(GNN, self).__init__()

        self.encoder = EncoderBlock(edge_input_size, hidden_size)

        self.GnBlock0 = GnBlock(hidden_size, K)
        self.GnBlock1 = GnBlock(hidden_size, K)
        self.GnBlock2 = GnBlock(hidden_size, K)

        self.EdgeBlock0 = EdgeDecoder(hidden_size, hidden_size, hidden_size)
        self.EdgeBlock1 = EdgeDecoder(hidden_size, hidden_size, hidden_size)
        self.EdgeBlock2 = EdgeDecoder(hidden_size, hidden_size, edge_output_size)


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