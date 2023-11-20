from typing import Optional, Callable

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import RGCNConv, MLP


class WeightedSum(torch.nn.Module):

    def __init__(self, input_units):
        super(WeightedSum, self).__init__()

        self.lin1 = Linear(input_units, 1)

    def forward(self, x, batch):

        weights = F.sigmoid(self.lin1(x))
        
        # Sum all weighted nodes per graph 
        return scatter(weights * x, batch, dim=0, reduce="sum")


class RGCN(torch.nn.Module):
    """
    Implementation of a relational graph neural network 
    used by Wu et al. for explaining molecular property
    prediction on a chemically intuitive approach. 
    source: https://doi.org/10.1038/s41467-023-38192-3
    """

    def __init__(
        self, 
        num_node_features: int, 
        num_rgcn_layers: int = 2,
        num_mlp_layers: int = 3,
        num_rgcn_hidden_units: int = 256,
        num_mlp_hidden_units: int = 64,
        activation_function: Optional[Callable] = F.relu,
        dropout_rate: float = 0.5
    ):
        super(RGCN, self).__init__()

        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        self.rgcn1 = RGCNConv(
            num_node_features,
            num_rgcn_hidden_units,
            num_relations=65
        )

        self.rgcn_layers = []
        for _ in range(num_rgcn_layers):
            self.rgcn_layers.append(
                RGCNConv(
                    num_rgcn_hidden_units,
                    num_rgcn_hidden_units,
                    num_relations=65
                )
            )

        self.weighted_sum = WeightedSum(num_rgcn_hidden_units)
        self.mlp = MLP(
            in_channels=num_rgcn_hidden_units,
            hidden_channels=num_mlp_hidden_units,
            out_channels=1,
            num_layers=3
        )

    def forward(self, x, edge_index, edge_type, batch):

        h = self.rgcn1(x, edge_index, edge_type.view(-1))
        h = self.activation_function(h)
        h = F.dropout(h, self.dropout_rate)

        for rgcn in self.rgcn_layers:
            h = rgcn(h, edge_index, edge_type.view(-1))
            h = self.activation_function(h)
            h = F.dropout(h, self.dropout_rate)

        molecular_embedding = self.weighted_sum(h, batch)
        
        return self.mlp(molecular_embedding)
