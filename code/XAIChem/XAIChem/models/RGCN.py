from typing import Callable

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import RGCNConv, BatchNorm, FastRGCNConv 


class RGCN(torch.nn.Module):

    def __init__(
        self,
        num_node_features: int,
        num_layers: int,
        num_hidden_units: list,
        rgcn_conv = RGCNConv,
        activation_function: Callable = F.relu,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = False,
        num_bases: int | None = None,
        loop: bool = True,
    ):
        super(RGCN, self).__init__()

        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.self_loop = loop

        # Initialize parameters for each layer 
        self.rgcn_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        self.self_loop_weights = torch.nn.ParameterList()

        for layer_id in range(num_layers):
            self.rgcn_layers.append(
                rgcn_conv(
                    num_node_features,
                    num_hidden_units[layer_id],
                    num_relations=65,
                    num_bases=num_bases,
                )
            )

            # Initialize batch norm for current layer if used 
            if self.use_batch_norm:
                self.batch_norm_layers.append(BatchNorm(num_hidden_units[layer_id]))

            # Initialize weights tensor for self loops for current layer if used 
            if self.self_loop:
                weight = torch.nn.Parameter(torch.Tensor(num_node_features, num_hidden_units[layer_id]))
                torch.nn.init.xavier_uniform_(weight, gain=torch.nn.init.calculate_gain("relu"))
                self.self_loop_weights.append(weight)
            
            # Number of input units is equal to the number of output units of the previous layer
            num_node_features = num_hidden_units[layer_id]

    def forward(self, data: torch_geometric.data.Data, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Iterate through the RGCN 

        :param data: a graph or a batch of graphs
        """

        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        for layer_id, rgcn in enumerate(self.rgcn_layers):
          
            # Add self loops using the weight of the current layer
            if self.self_loop:
                edge_index, edge_type = add_self_loops(data.edge_index, data.edge_type, fill_value=self.self_loop_weights[layer_id].data)
           
            x = rgcn(x, edge_index, edge_type.view(-1))
            x = self.activation_function(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)

            if self.use_batch_norm:
                x = self.batch_norm_layers[layer_id](x)

        return x 

