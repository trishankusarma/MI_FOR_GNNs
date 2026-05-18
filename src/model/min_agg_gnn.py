import torch
from torch_geometric.nn import MessagePassing

class MinAggGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Crucial: Set the aggregator to 'min'
        super(MinAggGNNLayer, self).__init__(aggr='min')
        self.mlp = torch.nn.Linear(in_channels, out_channels) #<5,2>

    def forward(self, x, edge_index, edge_weight):
        # Pass edge_weight into the propagation loop
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # x_j represents neighbor features. 
        # Bellman-Ford logic: Distance(neighbor) + Weight(edge)
        # We assume edge_weight is shaped correctly to broadcast
        return x_j + edge_weight.view(-1, 1)

    def update(self, aggr_out):
        # Process the chosen minimum paths through a simple linear layer/MLP
        return self.mlp(aggr_out) #<4,2> @ <5,2>.T => <4,5>