from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .. import createDataObjectFromSmiles
from ..prediction import predict

tqdm.pandas()


def difference(
    models: List[torch.nn.Module], masks: pd.DataFrame, method: str = "after"
) -> pd.DataFrame:
    """
    Compute the attribution of a substructure by calculating the difference between
    the prediction of the molecule and the prediction of the molecule where the substructure
    is masked (i.e. is partially not used in the model). There are two places where the mask
    can be applied: either before the message passing or after the message passing.

    :param models: list of ML models wherefrom the average prediction is used
        to compute the attribution
    :param masks: pandas DataFrame containing the smiles of the molecules and
        the masks
    :param method: determines where the mask is applied, can be 'after' or
        'before' (default is 'after')
    """

    results = defaultdict(list)

    # Use the models to make predictons of the unique unmasked molecules
    print("unmasked predictions")
    for smiles in masks["molecule_smiles"].unique():
        data = createDataObjectFromSmiles(smiles, np.inf)
        prediction = predict(data, models).item()

        results["molecule_smiles"].append(smiles)
        results["data"].append(data)
        results["prediction"].append(prediction)

    # Convert the predictions of the unmasked molecules to a pandas dataframe
    # and merge it to the 'masks' dataframe.
    print("convert to dataframe")
    results = pd.DataFrame.from_dict(results)
    attribution_df = pd.merge(masks, results, on="molecule_smiles")

    # Now, the apply function can be used to compute the masked predictions
    print("masked predictions")
    attribution_df["prediction_masked"] = attribution_df[
        ["data", "mask"]
    ].progress_apply(
        lambda row: predict(row["data"], models, row["mask"].view(-1, 1)).item(), axis=1
    )

    # Compute the attribution as the difference between the unmasked prediction and the
    # masked prediction
    print("compute difference")
    attribution_df["attribution"] = (
        attribution_df["prediction"] - attribution_df["prediction_masked"]
    )

    return attribution_df
