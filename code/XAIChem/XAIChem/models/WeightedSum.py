import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import scatter


class WeightedSum(torch.nn.Module):
    def __init__(self, input_units: int):
        super(WeightedSum, self).__init__()

        self.lin1 = Linear(input_units, 1)

    def forward(
        self, x: torch.Tensor, data, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        weights = F.sigmoid(self.lin1(x))

        if mask is None:
            # Sum all weighted nodes per graph
            return scatter(weights * x, data.batch, dim=0, reduce="sum")
        else:
            # Sum all weighted nodes present in the maske per graph
            return scatter(weights * x * mask, data.batch, dim=0, reduce="sum")
