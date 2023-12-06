from typing import List

import torch
import torch_geometric


def predict(
    graph: torch_geometric.data.Data,
    models: List[torch.nn.Module],
    mask: torch.Tensor | None = None,
    num_classes: int | None = None,
) -> torch.Tensor:
    # Disable gradient computation to save memory
    with torch.no_grad():
        if num_classes is None:
            predictions = torch.zeros(len(models))
        else:
            predictions = torch.zeros(len(models), num_classes)

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
    num_classes: int = 1,
):
    # Disable gradient computation to save memory
    with torch.no_grad():
        predictions = torch.zeros(data.batch_size, len(models), num_classes)

        for i, model in enumerate(models):
            predictions[:, i, :] = model(
                data.x, data.edge_index, data.edge_type, data.batch, masks
            ).view(data.batch_size, -1)

    return torch.mean(predictions, dim=1)
