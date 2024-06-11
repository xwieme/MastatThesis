# Evaluation the different attribution methods using different metrics commonly
# used to evaluate graph neural network explanations
import argparse
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import XAIChem
from dash import Dash, dcc, html
from plotly.subplots import make_subplots
from rdkit import Chem

DATA_DIR = "../../../data/ESOL"


def fidelity(
    group: pd.DataFrame,
    attribution_method: str,
    mode: str,
    models: list,
    device: str = "cpu",
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
    :param mode: remove most important features ('max'), remove least important
        features ('min'), remove most important features in absolute values ('abs_max')
        or remove features close to zero ('abs_min')
    :param models: the graph neural network models which are explained.
    :param device: which device to use for model prediction either 'cpu' or 'gpu'
        (default is 'cpu')
    """

    # group = group.query("substruct_smiles != 'scaffold'")

    if len(group) == 0:
        return np.inf

    if mode == "abs_max":
        most_important_substructs = group.query(
            f"{attribution_method}.abs() == {attribution_method}.abs().max()"
        )
    elif mode == "abs_min":
        most_important_substructs = group.query(
            f"{attribution_method}.abs() == {attribution_method}.abs().min()"
        )
    elif mode == "max":
        most_important_substructs = group.query(
            f"{attribution_method} == {attribution_method}.max()"
        )
    elif mode == "min":
        most_important_substructs = group.query(
            f"{attribution_method} == {attribution_method}.min()"
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
    mask = XAIChem.createMask(rdmol, atom_ids).to(device)

    molecule_graph = XAIChem.createDataObjectFromSmiles(smiles, np.nan)
    masked_prediction = float(
        XAIChem.predict(molecule_graph, models, mask.view(-1, 1), "aggregation", device)
    )

    return group.non_masked_prediction.iloc[0] - masked_prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="", dest="suffix")
    args = parser.parse_args()

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

    test_attributions = pd.read_json(
        os.path.join(DATA_DIR, f"test_attributions{args.suffix}.json")
    )

    # esol_df = pd.read_csv(os.path.join(DATA_DIR, "ESOL_test.csv"))
    # attributions = pd.read_json(
    #     os.path.join(DATA_DIR, f"attribution_functional_groups{args.suffix}.json")
    # )

    # test_attributions = pd.merge(
    #     attributions, esol_df, left_on="molecule_smiles", right_on="smiles", how="right"
    # ).dropna()

    print(test_attributions.head())
    print(test_attributions.info())

    fidelities = (
        test_attributions.drop_duplicates("molecule_smiles")[["molecule_smiles"]]
        .set_index("molecule_smiles")
        .copy()
    )

    for attribution_method in ["SME", "Shapley_value", "HN_value"]:
        fidelities[
            f"{attribution_method}_fidelity_positive"
        ] = test_attributions.groupby("molecule_smiles").apply(
            fidelity,
            attribution_method,
            mode="max",
            models=rgcn_models,
            device=torch_device,
        )

        # fidelities[
        #     f"{attribution_method}_fidelity_positive_abs"
        # ] = test_attributions.groupby("molecule_smiles").apply(
        #     fidelity,
        #     attribution_method,
        #     mode="abs_max",
        #     models=rgcn_models,
        #     device=torch_device,
        # )

        fidelities[
            f"{attribution_method}_fidelity_negative"
        ] = test_attributions.groupby("molecule_smiles").apply(
            fidelity,
            attribution_method,
            mode="min",
            models=rgcn_models,
            device=torch_device,
        )

        fidelities[
            f"{attribution_method}_fidelity_negative_abs"
        ] = test_attributions.groupby("molecule_smiles").apply(
            fidelity,
            attribution_method,
            mode="abs_min",
            models=rgcn_models,
            device=torch_device,
        )

    print(fidelities.median())

    data = pd.read_json(f"../../../data/ESOL/test_absolute_error{args.suffix}.json")

    # Condition plots on absolute error
    fidelities = fidelities.merge(data, left_on="molecule_smiles", right_on="smiles")

    fidelities_long = fidelities.melt(
        id_vars=["prediction", "ESOL", "absolute_error", "smiles"],
        var_name="temp",
        value_name="fidelity",
    )
    fidelities_long[
        ["attribution_method", "fidelity_type"]
    ] = fidelities_long.temp.str.split("_fidelity_", regex=True, expand=True)
    fidelities_long = fidelities_long.drop(columns="temp")

    print(fidelities_long.info())
    print(fidelities_long.query("fidelity_type == 'positive' and fidelity == 0"))
    fidelities_long.to_json("../../../data/test_fidelities.json")

    # Plot distributions
    app = Dash()

    # fig = go.Figure()
    fig = make_subplots(
        rows=2,
        shared_xaxes=True,
        shared_yaxes="all",
        row_titles=["AE < 0.6", "AE &#8805; 0.6"],
        y_title="Fidelity (log[mol/L])",
        x_title="Attribution_method",
        vertical_spacing=0.01,
    )
    absolute_error_classes = ["< 0.6", ">= 0.6"]

    for i, ae_class in enumerate(absolute_error_classes):
        fig.add_trace(
            go.Violin(
                x=fidelities_long.query(
                    f"fidelity_type == 'negative' and absolute_error {ae_class}"
                ).attribution_method,
                y=fidelities_long.query(
                    f"fidelity_type == 'negative' and absolute_error {ae_class}"
                ).fidelity,
                legendgroup="negative",
                scalegroup="negative",
                name="negative",
                side="negative",
                line_color="#1f77b4",
                showlegend=i == 0,
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Violin(
                x=fidelities_long.query(
                    f"fidelity_type == 'positive' and absolute_error {ae_class}"
                ).attribution_method,
                y=fidelities_long.query(
                    f"fidelity_type == 'positive' and absolute_error {ae_class}"
                ).fidelity,
                legendgroup="positive",
                scalegroup="positive",
                name="positive",
                side="positive",
                line_color="#ff7f0e",
                showlegend=i == 0,
            ),
            row=i + 1,
            col=1,
        )

        fig.add_trace(
            go.Violin(
                x=fidelities_long.query(
                    f"fidelity_type == 'negative_abs' and absolute_error {ae_class}"
                ).attribution_method,
                y=fidelities_long.query(
                    f"fidelity_type == 'negative_abs' and absolute_error {ae_class}"
                ).fidelity,
                legendgroup="absolute_negative",
                scalegroup="negative",
                name="absolute negative",
                #     side="negative",
                line_color="#d62728",
                #     showlegend=i == 0,
            ),
            row=i + 1,
            col=1,
        )
        # fig.add_trace(
        #     go.Violin(
        #         x=fidelities_long.query(
        #             f"fidelity_type == 'positive_abs' and absolute_error {ae_class}"
        #         ).attribution_method,
        #         y=fidelities_long.query(
        #             f"fidelity_type == 'positive_abs' and absolute_error {ae_class}"
        #         ).fidelity,
        #         legendgroup="absolute_positive",
        #         scalegroup="positive",
        #         name="absolute positive",
        #         side="positive",
        #         line_color="#2ca02c",
        #         showlegend=i == 0,
        #     ),
        #     row=i + 1,
        #     col=1,
        # )

    for annotation in fig["layout"]["annotations"]:
        if annotation["textangle"] == 90:
            annotation["textangle"] = 0

    fig.update_layout(
        autosize=False,
        width=800,
        height=500,
        violingap=0,
        violinmode="overlay",
        template="plotly_white",
        legend_title="Fidelity type",
        legend=dict(yanchor="middle", y=0.5, xanchor="right", x=1.5),
    )
    fig.update_traces(
        meanline_visible=True,
        points=False,
        scalemode="count",
    )
    fig.show()

    app.layout = html.Div(dcc.Graph(figure=fig), style={"display": "flex"})
    app.run()
