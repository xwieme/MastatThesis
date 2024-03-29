# Evaluation the different attribution methods using different metrics commonly
# used to evaluate graph neural network explanations
import os

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import XAIChem
from dash import Dash, dcc, html
from rdkit import Chem

DATA_DIR = "../../../data/ESOL"


def fidelity(
    group: pd.DataFrame,
    attribution_method: str,
    models: list,
    device: str = "cpu",
    mode: str = "max",
) -> float:
    """
    Compute the difference between the model prediction and the prediction where
    the most or least important features (i.e. the explanation) are masked. This
    metric indicates if the explanation is faithfull to the model as removing the
    most important features should significantly change the model output, whereas
    remove the least important features should not result in a big output change.

    :param group: pandas dataframe containing the substructures and attributions
        of a molecule.
    :param attribution_method: column name of the attributions used to select the
        important substructures.
    :param models: the graph neural network models which are explained.
    :param device: which device to use for model prediction either 'cpu' or 'gpu'
        (default is 'cpu')
    :param mode: remove most important features ('max') or remove least important
        features ('min') (default is 'max')
    """

    if mode == "max":
        most_important_substructs = group.query(
            f"{attribution_method}.abs() == {attribution_method}.abs().max()"
        )
    elif mode == "min":
        most_important_substructs = group.query(
            f"{attribution_method}.abs() == {attribution_method}.abs().min()"
        )
    else:
        raise ValueError(
            f"given mode '{mode}' is not supported, should be either 'min' or 'max'"
        )

    smiles = group.molecule_smiles.iloc[0]
    rdmol = Chem.MolFromSmiles(smiles)

    # All information necessary is available if only one substructure is the
    # most import one.
    if len(most_important_substructs) == 1:
        return (
            most_important_substructs["non_masked_prediction"].iloc[0]
            - most_important_substructs["masked_prediction"].iloc[0]
        )

    atom_ids = [id for ids in most_important_substructs["atom_ids"] for id in ids]
    mask = XAIChem.createMask(Chem.MolFromSmiles(smiles), atom_ids)

    molecule_graph = XAIChem.createDataObjectFromSmiles(smiles, np.nan)
    masked_prediction = XAIChem.predict(molecule_graph, models, mask, device)

    return group.non_masked_prediction.iloc[0] - masked_prediction


if __name__ == "__main__":
    # Get model architecture and configuration
    torch_device = torch.device("cuda")
    model, config = XAIChem.models.PreBuildModels.rgcnWuEtAll(
        "./model_config.yaml", ["seed"], model_id=0
    )

    # Load trained models
    paths = [
        os.path.join(DATA_DIR, f"trained_models/rgcn_model_{i}_early_stop.pt")
        for i in range(10)
    ]
    rgcn_models = XAIChem.loadModels(model, paths, device="cuda")

    test_attributions = pd.read_json(os.path.join(DATA_DIR, "test_attributions.json"))

    print("Fidelity positive")
    fidelities_pos = {
        attribution_method: test_attributions.groupby("molecule_smiles").apply(
            fidelity,
            attribution_method,
            models=rgcn_models,
            device=torch_device,
        )
        for attribution_method in ["SME", "Shapley_value", "HN_value"]
    }

    for attribution_method, fidelity_distr in fidelities_pos.items():
        print(f"{attribution_method}: {np.round(np.mean(fidelity_distr), 2)}")

    print("\nFidelity negative")
    fidelities_neg = {
        attribution_method: test_attributions.groupby("molecule_smiles")
        .apply(
            fidelity,
            attribution_method,
            models=rgcn_models,
            device=torch_device,
            mode="min",
        )
        .abs()
        for attribution_method in ["SME", "Shapley_value", "HN_value"]
    }

    for attribution_method, fidelity_distr in fidelities_neg.items():
        print(f"{attribution_method}: {np.round(np.mean(fidelity_distr), 2)}")

    app = Dash()

    figs_div = []

    for attribution_method, fidelity_distr in fidelities_pos.items():
        fig = px.histogram(x=fidelity_distr)
        fig.update_layout(
            autosize=False, width=800, height=500, template="plotly_white"
        )

        figs_div.append(html.Div([html.H2(attribution_method), dcc.Graph(figure=fig)]))

    app.layout = html.Div([*figs_div], style={"display": "flex"})
    app.run()
