from typing import Callable

import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import RGCNConv


class RGCN(torch.nn.Module):
    """
    Create a relational graph convolutional neural network using pytorch geometric.
    """

    def __init__(
        self,
        num_node_features: int,
        num_hidden_units: list,
        rgcn_conv=RGCNConv,
        activation_function: Callable = F.relu,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = False,
        use_residual: bool = True,
        num_bases: int | None = None,
    ):
        super().__init__()

        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # Initialize parameters for each layer
        self.rgcn_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        self.residual_layers = torch.nn.ModuleList()

        for num_units in num_hidden_units:
            self.rgcn_layers.append(
                rgcn_conv(
                    num_node_features,
                    num_units,
                    num_relations=65,
                    num_bases=num_bases,
                )
            )

            # Initialize residual for current layer if used
            if self.use_residual:
                self.residual_layers.append(Linear(num_node_features, num_units))

            # Initialize batch norm for current layer if used
            if self.use_batch_norm:
                self.batch_norm_layers.append(BatchNorm1d(num_units))

            # Number of input units is equal to the number of output units of the previous layer
            num_node_features = num_units

    def _forwardLayer(
        self,
        layer_id: int,
        rgcn: Callable,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        mask: torch.Tensor | None,
        mask_method: str | None,
    ):
        """
        Go through one relational graph convolutional network (rgcn) layer.
        Apply residual and batch norm if requested.

        :param layer_id: identifier of the current RGCN layer to apply
        :param rgcn: the current RGCN layer
        :param x: input tensor
        :param edge_index: tensor specifying the edges of the graph
        :param edge_type: tensor specifying the type of edges in the graph
        :param data: a graph or a batch of graphs
        :param mask: tensor that masks part of the nodes in the graph (optional,
            default is None)
        :param mask_method: when to apply the mask, during the RGCN part of the
            model (i.e. 'rgcn') or during the aggregation step (i.e. 'aggregation')
            (optional, default is None)
        """

        h = rgcn(x, edge_index, edge_type.view(-1))
        h = self.activation_function(h)
        h = F.dropout(h, self.dropout_rate, training=self.training)

        if self.use_residual:
            res_x = self.activation_function(self.residual_layers[layer_id](x))

            h = h + res_x

        if self.use_batch_norm:
            h = self.batch_norm_layers[layer_id](h)

        if mask_method == "rgcn":
            h = h * mask

        return h

    def forward(
        self,
        data: torch_geometric.data.Data,
        mask: torch.Tensor | None = None,
        mask_method: str | None = None,
    ) -> torch.Tensor:
        """
        Iterate through the RGCN

        :param data: a graph or a batch of graphs
        :param mask: tensor that masks part of the nodes in the graph (optional,
            default is None)
        :param mask_method: when to apply the mask, during the RGCN part of the
            model (i.e. 'rgcn') or during the aggregation step (i.e. 'aggregation')
            (optional, default is None)
        """

        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        if mask_method == "before":
            x = x * mask

        for layer_id, rgcn in enumerate(self.rgcn_layers):
            x = self._forwardLayer(
                layer_id, rgcn, x, edge_index, edge_type, mask, mask_method
            )

        return x
