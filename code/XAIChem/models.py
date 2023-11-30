from typing import Optional, Callable

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import RGCNConv, MLP


class WeightedSum(torch.nn.Module):

    def __init__(self, input_units: int):
        super(WeightedSum, self).__init__()

        self.lin1 = Linear(input_units, 1)

    def forward(
        self, 
        x: torch.Tensor, 
        batch: torch.Tensor, 
        mask: torch.Tensor = None
    ) -> torch.Tensor:

        weights = F.sigmoid(self.lin1(x))

        if mask is None:
            # Sum all weighted nodes per graph 
            return scatter(weights * x, batch, dim=0, reduce="sum")
        else:
            # Sum all weighted nodes present in the maske per graph
            return scatter(weights * x * mask, batch, dim=0, reduce="sum")


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
        num_mlp_output_units: int = 1,
        activation_function: Optional[Callable] = F.relu,
        rgcn_dropout_rate: float = 0.5,
        mlp_dropout_rate: float | list = 0.1,
    ):
        super(RGCN, self).__init__()

        self.activation_function = activation_function
        self.dropout_rate = rgcn_dropout_rate

        self.rgcn1 = RGCNConv(
            num_node_features,
            num_rgcn_hidden_units,
            num_relations=65
        )

        self.rgcn_layers = torch.nn.ModuleList([
            RGCNConv(num_rgcn_hidden_units, num_rgcn_hidden_units, num_relations=65)
            for _ in range(1, num_rgcn_layers)
        ])

        self.weighted_sum = WeightedSum(num_rgcn_hidden_units)
        self.mlp = MLP(
            in_channels=num_rgcn_hidden_units,
            hidden_channels=num_mlp_hidden_units,
            out_channels=num_mlp_output_units,
            num_layers=num_mlp_layers,
            dropout=mlp_dropout_rate,
        )

    def forward(self, x, edge_index, edge_type, batch, mask=None):

        h = self.rgcn1(x, edge_index, edge_type.view(-1))
        h = self.activation_function(h)
        h = F.dropout(h, self.dropout_rate, training=self.training)

        # h = self.rgcn2(h, edge_index, edge_type.view(-1))
        # h = self.activation_function(h)
        # h = F.dropout(h, self.dropout_rate, training=self.training)

        for rgcn in self.rgcn_layers:
            h = rgcn(h, edge_index, edge_type.view(-1))
            h = self.activation_function(h)
            h = F.dropout(h, self.dropout_rate, training=self.training)

        molecular_embedding = self.weighted_sum(h, batch, mask)
        
        return self.mlp(molecular_embedding)
