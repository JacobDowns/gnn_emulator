import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_sum
from graph_data_loader import GraphDataLoader
from torch_geometric.loader import DataLoader
import torch 

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


def copy_geometric_data(graph):

    x_edge = graph.x_edge
    x_node = graph.x_node
    edge_index = graph.edge_index
    
    ret = Data(x_edge = x_edge, x_node = x_node, edge_index=edge_index)
    
    return ret


class Encoder(nn.Module):

    def __init__(self,
                edge_input_size=32,
                hidden_size=32):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
    
    def forward(self, graph):

        x_g = graph.x_g
        x_i = graph.x_i

        x_edge = torch.column_stack([
            x_g,
            x_i
        ])
        
        edges = graph.edge_index
        x_node = torch.zeros((graph.num_nodes, x_edge.shape[1]))

        print(x_node)
        quit()
        scatter_sum(x_edge, edges[:,0], dim=0, out=x_node)
        scatter_sum(x_edge, edges[:,1], dim=0, out=x_node)

        x_node_ = self.nb_encoder(x_node)
        x_edge_ = self.eb_encoder(x_edge)
        return Data(x_node = x_node_, x_edge = x_edge_, edge_index=edges)


class EdgeBlock(nn.Module):

    def __init__(self, custom_func=None):
        
        super(EdgeBlock, self).__init__()
        self.net = custom_func


    def forward(self, graph):

        x_edge = graph.x_edge
        x_node = graph.x_node
        edges = graph.edge_index

        e0 = edges[:,0]
        e1 = edges[:,1]

        v0 = x_node[e0]
        v1 = x_node[e1]

        x_new = torch.column_stack([
            v0,
            v1,
            x_edge
        ])

        x_edge = self.net(x_new)  

        return Data(x_edge=x_edge, x_node=x_node, edge_index=edges)


class NodeBlock(nn.Module):

    def __init__(self, custom_func=None):

        super(NodeBlock, self).__init__()

        self.net = custom_func

    def forward(self, graph):

        x_node = graph.x_node
        x_edge = graph.x_edge
        edges = graph.edge_index

        # Aggregate edge features
        x_node1 = torch.zeros((graph.num_nodes, x_edge.shape[1]))
        scatter_sum(x_edge, edges[:,0], dim=0, out=x_node1)
        scatter_sum(x_edge, edges[:,1], dim=0, out=x_node1)

        x_node = torch.column_stack([
            x_node,
            x_node1
        ])

        x_node = self.net(x_node)
        return Data(x_node=x_node, x_edge=x_edge, edge_index=edges)
       

class GnBlock(nn.Module):

    def __init__(self, hidden_size=32):

        super(GnBlock, self).__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
    
        graph_last = copy_geometric_data(graph)
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)

        x_edge = graph_last.x_edge + graph.x_edge
        x_node = graph_last.x_node + graph.x_node
        return Data(x_node=x_node, x_edge=x_edge, edge_index=graph.edge_index)


class Decoder(nn.Module):

    def __init__(self, hidden_size=32, output_size=1):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x_edge)


loader = GraphDataLoader()
train_data = loader.training_data
test_data = loader.test_data

train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle = True)


f0 = Encoder(edge_input_size = 11, hidden_size=32)
g0 = GnBlock(hidden_size=32)
g1 = GnBlock(hidden_size=32)
d0 = Decoder(hidden_size=32, output_size=1)


for batch_index, graph in enumerate(train_loader):
    print(graph)

    graph = f0(graph)
    graph = g0(graph)
    graph = g1(graph)
    y = d0(graph)

  
    quit()
