from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from XAIChem import createDataObjectFromSmiles, predict, predictBatch


def difference(
    models: List[torch.nn.Module],
    molecule_df: pd.DataFrame,
    device,
    method: str = "after",
) -> pd.DataFrame:
    """
    Compute the attribution of a substructure by calculating the difference between
    the prediction of the molecule and the prediction of the molecule where the substructure
    is masked (i.e. is partially not used in the model).

    :param models: list of ML models wherefrom the average prediction is used
        to compute the attribution
    :param molecule_df: pandas dataframe resulting from XAIChem.substructures
    :param method: determines where the mask is applied, can be 'after' or
        'before' (default is 'after')
    """

    data = createDataObjectFromSmiles(molecule_df.molecule_smiles.iloc[0], np.inf)
    prediction = predict(data, models, device=device).item()

    masks = torch.stack(molecule_df["mask"].to_list(), dim=0) ^ 1
    data_batch = DataLoader([data for _ in range(masks.shape[0])], batch_size=256)

    for batch in data_batch:
        molecule_df["masked_prediction"] = predictBatch(
            batch, models, masks.view(-1, 1).to(device), device
        )

    molecule_df["difference"] = prediction - molecule_df.masked_prediction

    return molecule_df
