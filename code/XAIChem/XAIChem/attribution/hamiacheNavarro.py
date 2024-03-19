import time
from functools import lru_cache
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .. import graph
from ..data import createDataObjectFromSmiles
from ..prediction import predictBatch


@lru_cache(maxsize=None)
def _createMc(N, t):
    """
    Creat a matrix representation of an associated game as defined by Hamiache
    (https://doi.org/10.1007/s00182-019-00688-y).

    :param N: a set of players
    :param t: hyperparameter controlling the propotion of surplus
    """

    M = np.zeros(shape=(2 ** len(N) - 1, 2 ** len(N) - 1))
    all_coalitions = list(graph.powerset(N))

    # Diagonal elements
    for i, S in enumerate(all_coalitions):
        M[i, i] = 1 - (len(N) - len(S)) * t

        for j, T in enumerate(all_coalitions):
            if len(T) == 1 and not set(T).issubset(set(S)):
                M[i, j] = -t

            elif len(S) + 1 == len(T) and set(S).issubset(set(T)):
                M[i, j] = t

    return M


def _createPg(N: frozenset | tuple, g: Iterable) -> np.ndarray:
    """
    Create the matrix Pg as defined by Hamiache containing the graphical
    information. (https://doi.org/10.1007/s00182-019-00688-y)

    :param N: a set of players
    :param g: a set of adjacent players
    """

    P = np.zeros(shape=(2 ** len(N) - 1, 2 ** len(N) - 1))
    all_coalitions = list(graph.powerset(N))

    for i, S in enumerate(all_coalitions):
        g_S = graph.inducedGraph(S, g)
        for R in graph.partition(S, g_S):
            P[i, all_coalitions.index(R)] = 1

    return P


def _hamiacheNavarroValue(
    N: frozenset | tuple, v: np.ndarray, g: Iterable, t: float | None = None
) -> np.ndarray:
    """
    Computes a value for a transfer utility game (TU game) with incomplete
    communication as developed by Hamiache and Navarro.

    :param N: set of players
    :param v: characteristic function mapping a coalition (i.e. a subset of N)
        to a real value, represented as a numpy array.
    :param g: set of adjacent players
    :param t: hyperparameter specifying the amount of surplus. (default is
        None)
    """

    Mc = _createMc(N, t)
    Pg = _createPg(N, g)

    Mg = Pg @ Mc @ Pg

    # Converge power series
    Mg_old = np.inf
    while not np.allclose(Mg, Mg_old):
        Mg_old = Mg
        Mg = Mg @ Mg

    # Return the value
    return Mg @ v


def _shapleyValue(
    N: frozenset | tuple, v: np.ndarray, t: float | None = None
) -> np.ndarray:
    """
    Computes a value for a transfer utility game (TU game) with  as developed
    by Shapley () using the associated game formulation developed by Hamiache
    ()

    :param N: set of players
    :param v: characteristic function mapping a coalition (i.e. a subset of N)
        to a real value, represented as a numpy array.
    :param t: hyperparameter specifying the amount of surplus. (default is
        None)
    """

    Mc = _createMc(N, t)
    Pg = np.identity(Mc.shape[0])

    Mg = Pg @ Mc @ Pg

    # Converge power series
    Mg_old = np.inf
    while not np.allclose(Mg, Mg_old):
        Mg_old = Mg
        Mg = Mg @ Mg

    # Return the value
    return Mg @ v


def maskedPredictions(
    models,
    molecule: Data,
    masks: List[torch.Tensor],
    mask_method: str,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
):
    """
    Compute the model predictions of every combination between
    substructures of a molecule defined by the masks.

    :param models: machine learning models
    :param smiles: smiles representation of a molecule
    :param masks: list for tensors, each selecting a substructure in the
        molecule defined by smiles.
    :param mask_method: when to apply the mask, during the RGCN part of the
        model (i.e. 'rgcn') or during the aggregation step (i.e. 'aggregation')
        (optional, default is None)
    """

    num_substructures = len(masks)

    molecule_batch = [molecule for _ in range(2**num_substructures - 1)]
    molecule_batch = DataLoader(molecule_batch, batch_size=batch_size)

    mask_batch = []
    for indices in graph.powerset(list(range(num_substructures))):
        mask_batch.append(torch.stack([masks[i] for i in indices]).sum(dim=0))

    mask_batch = torch.stack(mask_batch, dim=0)

    predictions = np.zeros(2**num_substructures - 1)
    for i, batch in enumerate(molecule_batch):
        batch.to(device)
        mask_batch_subset = mask_batch[i * 256 : (i + 1) * 256, :].to(device)

        predictions[i * batch_size : (i + 1) * batch_size] = predictBatch(
            batch, models, mask_batch_subset.view(-1, 1), mask_method, device
        ).to("cpu")

    return predictions


def hamiacheNavarroValue(
    models,
    molecule_df: pd.DataFrame,
    average_prediction: float,
    shapley: bool = False,
    batch_size: int = 256,
    mask_method: str = "aggregation",
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """
    Computes the Hamiache-Navarro value of the game where the players are defined
    by molecular substructure given in molecule_df. The characteristic function v
    is defined by the model prediction of a subset of the substructures minus the
    average prediction over the entire dataset.

    :param models: machine learning models to explain
    :param molecule_df: a result from XAIChem.substructures defining the substructures
        of interest
    :param average_prediction: average model prediction over the entire data set
    :param shapley: specify if the Shapley value also needs to be computed or not
        (default is False)
    :param batch_size: size of batch used in to compute the predictions (default is 256)
    :param mask_method: when to apply the mask, during the RGCN part of the
        model (i.e. 'rgcn') or during the aggregation step (i.e. 'aggregation')
        (optional, default is None)
    :param device: the device used to compute the predictions, either "cpu" or "cuda"
        (default is "cpu")
    """

    smiles = molecule_df.molecule_smiles.iloc[0]

    t1 = time.time()

    # Compute the vector representing the characteristic function v by calcuation
    # the model predicting of all possible combinations between the substructures
    molecule = createDataObjectFromSmiles(smiles, np.inf)
    masks = molecule_df["mask"].to_list()
    predictions = maskedPredictions(
        models, molecule, masks, mask_method, batch_size, device
    )

    t2 = time.time()

    # Compute the HN-value
    N = tuple(range(len(masks)))
    g = graph.reducedMolecularGraph(
        list(zip(*molecule.cpu().edge_index.numpy())), molecule_df["atom_ids"].to_dict()
    )

    molecule_df["HN_value"] = _hamiacheNavarroValue(
        N, predictions - average_prediction, g, 2 / (len(masks) * 10)
    )[: len(masks)]

    t3 = time.time()

    molecule_df["time_HN_value"] = t3 - t1

    if shapley:
        molecule_df["Shapley_value"] = _shapleyValue(
            N, predictions - average_prediction, 2 / (len(masks) * 10)
        )[: len(masks)]

        t4 = time.time()

        molecule_df["time_Shapley_value"] = t4 - t3 + t2 - t1

    return molecule_df


def shapleyValue(
    models,
    smiles: str,
    molecule_df: pd.DataFrame,
    average_prediction: float,
    batch_size: int = 256,
    mask_method: str = "aggregation",
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """
    Computes the Shapley value of the game where the players are defined
    by molecular substructure given in molecule_df. The characteristic function v
    is defined by the model prediction of a subset of the substructures minus the
    average prediction over the entire dataset.

    :param models: list of machine learning models
    :param smiles: smiles representation of a molecule
    :param molecule_df: a result from XAIChem.substructures defining the substructures
        of interest
    :param average_prediction: average model prediction over the entire data set
    :param batch_size: size of batch used in to compute the predictions (default is 256)
    :param mask_method: when to apply the mask, during the RGCN part of the
        model (i.e. 'rgcn') or during the aggregation step (i.e. 'aggregation')
        (optional, default is None)
    :param device: the device used to compute the predictions, either "cpu" or "cuda"
        (default is "cpu")
    """

    # Compute the vector representing the characteristic function v by calcuation
    # the model predicting of all possible combinations between the substructures
    molecule = createDataObjectFromSmiles(smiles, np.inf)
    masks = molecule_df.masks.to_list()
    predictions = maskedPredictions(
        models, molecule, masks, mask_method, batch_size, device
    )

    # Compute the HN-value
    N = tuple(range(len(masks)))

    molecule_df["Shapley_value"] = _shapleyValue(
        N, predictions - average_prediction, 2 / (len(masks) * 10)
    )

    return molecule_df
