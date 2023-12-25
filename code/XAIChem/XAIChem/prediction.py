from typing import List

import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


def predict(
    data: Data,
    models: List[torch.nn.Module],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:

    # Disable gradient computation to save memory
    with torch.no_grad():

        predictions = torch.zeros(len(models))
        data.batch = torch.zeros(1, dtype=torch.long)

        for i, model in enumerate(models):
            predictions[i] = model(data, mask)

    return torch.mean(predictions)


def predictBatch(
    data: DataLoader,
    models: List[torch.nn.Module],
    masks: torch.Tensor | None = None,
):
    # Disable gradient computation to save memory
    with torch.no_grad():
        predictions = torch.zeros(data.batch_size, len(models))

        for i, model in enumerate(models):
            predictions[:, i] = model(data, masks).view(-1)

    return torch.mean(predictions, dim=1)
