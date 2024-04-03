import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import scatter


class WeightedSum(torch.nn.Module):
    def __init__(self, input_units: int):
        super(WeightedSum, self).__init__()

        self.lin1 = Linear(input_units, 1)

    def forward(
        self,
        x: torch.Tensor,
        data,
        mask: torch.Tensor | None = None,
        mask_method: str | None = None,
    ) -> torch.Tensor:
        """
        Compute a weighted sum where the weigths for each node are equal to the sigmoid of a
        linear combination of the features of that node.

        :param x: node feature matrix
        :param data: pytorch data.Data object containing batch information
        :param mask: tensor that masks part of the nodes in the graph
        :param mask_method: when to apply the mask, during the RGCN part of the
            model (i.e. 'rgcn') or during the aggregation step (i.e. 'aggregation')
            (optional, default is None)
        """

        weights = F.sigmoid(self.lin1(x))

        if mask_method == "aggregation":
            # Sum all weighted nodes present in the maske per graph
            return scatter(weights * x * mask, data.batch, dim=0, reduce="sum")

        # Sum all weighted nodes per graph
        return scatter(weights * x, data.batch, dim=0, reduce="sum")
