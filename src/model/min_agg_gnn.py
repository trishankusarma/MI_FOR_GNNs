import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MinAggGNNLayer(MessagePassing):
    def __init__(self, in_channels, hidden_layers, out_channels=1):
        # Crucial: Set the aggregator to 'min' for Bellman-Ford path tracking
        super(MinAggGNNLayer, self).__init__(aggr='min')

        layers = []
        current_in = in_channels

        # 1. Build hidden layers with explicit non-linear activation functions
        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_in, h_dim))
            layers.append(nn.ReLU())  # Prevents multi-layer mathematical collapse
            current_in = h_dim

        # 2. Final regression projection layer (Collapses hidden dimensions to 1 channel)
        layers.append(nn.Linear(current_in, out_channels))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_weight):
        # Pass edge_weight into the propagation loop smoothly
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # Bellman-Ford core heuristic: Distance(neighbor) + Weight(edge)
        # edge_weight is automatically broadcasted across the hidden features
        return x_j + edge_weight.view(-1, 1)

    def update(self, aggr_out):
        # Process the optimized min-aggregated paths through our unified MLP pipeline
        return self.mlp(aggr_out)