from typing import List

import torch
import torch_geometric


def predict(
    graph: torch_geometric.data.Data,
    models: List[torch.nn.Module],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Disable gradient computation to save memory
    with torch.no_grad():

        predictions = torch.zeros(len(models))
        for i, model in enumerate(models):
            predictions[i] = model(
                graph.x,
                graph.edge_index,
                graph.edge_type,
                torch.zeros(1, dtype=torch.long),
                mask,
            )

    return torch.mean(predictions)


def predictBatch(
    data: torch_geometric.loader.DataLoader,
    models: List[torch.nn.Module],
    masks: torch.Tensor | None = None,
):
    # Disable gradient computation to save memory
    with torch.no_grad():
        predictions = torch.zeros(data.batch_size, len(models))

        for i, model in enumerate(models):
            predictions[:, i] = model(
                data.x, data.edge_index, data.edge_type, data.batch, masks
            ).view(-1)

    return torch.mean(predictions, dim=1)
