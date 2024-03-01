from typing import Callable

import torch


class MolecularPropertyPredictor(torch.nn.Module):
    """
    Implementation of a machine learning model to predict
    molecular properties. This model consist of four parts:
        1) A graph neural network
        2) An aggregation to generate a molecular embedding
        3) A multi-layer-peceptron
        4) Post processing if necessairy (e.g. sigmoid for classification)

    :param gnn: the graph neural network used in the first step
    :param aggregation: specify how the molecular embedding is
        computed from the gnn output
    :param mlp: the multi-layer-peceptron used in step three
    :param out: a function applied to the output of the mlp
    """

    def __init__(
        self,
        gnn: torch.nn.Module,
        aggregation: torch.nn.Module,
        mlp: torch.nn.Module,
        out: Callable | None = None,
    ):
        super().__init__()

        self.gnn = gnn
        self.aggregation = aggregation
        self.mlp = mlp
        self.out = out

    def forward(self, data, mask: torch.Tensor | None = None):
        x = self.gnn.forward(data, mask)
        molecular_embedding = self.aggregation(x, data, mask)
        x = self.mlp(molecular_embedding)

        if self.out is not None:
            x = self.out(x)

        return x
