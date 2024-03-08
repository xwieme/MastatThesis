"""
This script compares the different attribution methods (i.e. SME, Shapley value 
and HN value) by plotting the prediction RMSE in function of the Spearman rank 
correlation between two methods. Two methods are used to subdivide molecules in 
substructures (i.e. functional groups and BRICS) and their distribution of the 
number of substructures for a molecule is compared. Also, the two substructure 
methods are combined by taking the method producing the most number of substructures,
because most molecules in the data set cannot be split using BRICS.
"""


import os

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

DATA_DIR = "../../../data"


def spearmanRankCorr(data: pd.DataFrame, method1: str, method2: str) -> pd.DataFrame:
    # Indicate if a molecule can be split in substructures or not
    filter_df = data.groupby("molecule_smiles").apply(len)

    return (
        filter_df.to_frame("N_substructures")
        .query("N_substructures > 1")
        .join(data.set_index("molecule_smiles"))
        .groupby("molecule_smiles")[[method1, method2]]
        .corr(method="spearman")[[method2]]
        .reset_index()
        .query("level_1 == @method1")
        .drop(columns="level_1", axis=0)
        .rename(columns={method2: "corr"})
        .set_index("molecule_smiles")
        .join(filter_df.to_frame("N_substructures"))
        .reset_index()
    )


def correlationPlot(df: pd.DataFrame, title: str):
    fig = px.scatter(
        df.set_index("molecule_smiles").join(rmse_df[["RMSE"]]),
        x="corr",
        y="RMSE",
        color="N_substructures",
        color_continuous_scale="inferno",
        title=title,
    )
    fig.update_layout(autosize=False, width=800, height=500, template="seaborn")

    return fig


if __name__ == "__main__":
    app = Dash()

    # Load reference dataset and attribution data
    data = pd.read_csv(os.path.join(DATA_DIR, "ESOL/ESOL.csv"))
    attributions_functional_groups = pd.DataFrame(
        pd.read_json(os.path.join(DATA_DIR, "ESOL/attribution.json"))
    )
    attributions_brics = pd.DataFrame(
        pd.read_json(os.path.join(DATA_DIR, "/ESOL/attribution_brics.json"))
    )

    # Compute the RMSE between the model prediction and experimental data
    rmse_df = data.set_index("smiles").join(
        attributions_brics[["molecule_smiles", "non_masked_prediction"]]
        .drop_duplicates()
        .set_index("molecule_smiles"),
    )
    rmse_df["RMSE"] = np.abs(rmse_df["ESOL"] - rmse_df["non_masked_prediction"])

    # Get the fraction of molecules where the prediction is larger than
    # the experimental RMSE of 0.6 logS
    print(f"{len(rmse_df.query('RMSE > 0.6')) / len(rmse_df) * 100:.2f}%")

    # Compute average number of substructures per molecule
    n_functional_groups = (
        attributions_functional_groups.groupby("molecule_smiles")
        .size()
        .to_frame("N_fg")
    )
    print("\n##### Functional groups #####")
    print(n_functional_groups.N_fg.describe([]))

    n_brics_fragments = (
        attributions_brics.groupby("molecule_smiles").size().to_frame("N_brics")
    )
    print("\n##### BRICS #####")
    print(n_brics_fragments.N_brics.describe([]))

    # Get fraction of molecule that can either be split in at least two
    # functional groups or at least two BRICS fragments
    can_explain = n_functional_groups.join(n_brics_fragments).apply(
        lambda row: 1 if row["N_fg"] > 1 or row["N_brics"] > 1 else 0, axis=1
    )
    print(
        f"\nFraction of molecules that can be explained: {can_explain.sum() / len(can_explain) * 100:.2f}%"
    )

    # # Inspect molecules that cannot be split into functional groups or BRICS fragments
    # smiles = (
    #     can_explain.to_frame("can_explain")
    #     .query("can_explain == 0")
    #     .reset_index()
    #     .molecule_smiles
    # )
    # rdmols = [Chem.MolFromSmiles(smiles_str) for smiles_str in smiles]
    # img = Draw.MolsToGridImage(rdmols, molsPerRow=5, subImgSize=(500, 500))

    # app.layout = html.Div(html.Img(src=img))

    # Compute the Spearman rank correlation between different attribution
    # metrics for each molecule that can be split into substructures
    ##### Functional groups #####
    sme_vs_shapley_corr_fg = spearmanRankCorr(
        attributions_functional_groups, "SME", "Shapley_value"
    )
    shapley_vs_hn_corr_fg = spearmanRankCorr(
        attributions_functional_groups, "Shapley_value", "HN_value"
    )
    sme_vs_hn_corr_fg = spearmanRankCorr(
        attributions_functional_groups, "SME", "HN_value"
    )

    fig_sme_vs_shapley_fg = correlationPlot(sme_vs_shapley_corr_fg, "SME vs Shapley")
    fig_shapley_vs_hn_fg = correlationPlot(shapley_vs_hn_corr_fg, "Shapley vs HN")
    fig_sme_vs_hn_fg = correlationPlot(sme_vs_hn_corr_fg, "SME vs HN")

    ##### BRICS #####
    sme_vs_shapley_corr_brics = spearmanRankCorr(
        attributions_brics, "SME", "Shapley_value"
    )
    shapley_vs_hn_corr_brics = spearmanRankCorr(
        attributions_brics, "Shapley_value", "HN_value"
    )
    sme_vs_hn_corr_brics = spearmanRankCorr(attributions_brics, "SME", "HN_value")

    fig_sme_vs_shapley_brics = correlationPlot(
        sme_vs_shapley_corr_brics, "SME vs Shapley"
    )
    fig_shapley_vs_hn_brics = correlationPlot(shapley_vs_hn_corr_brics, "Shapley vs HN")
    fig_sme_vs_hn_brics = correlationPlot(sme_vs_hn_corr_brics, "SME vs HN")

    fig_dist_brics = px.bar(
        attributions_brics.groupby("molecule_smiles")
        .apply(len)
        .to_frame("N")
        .value_counts()
        .reset_index(),
        x="N",
        y="count",
    )
    fig_dist_brics.update_layout(
        autosize=False, width=800, height=500, template="seaborn"
    )

    fig_dist_fg = px.bar(
        attributions_functional_groups.groupby("molecule_smiles")
        .apply(len)
        .to_frame("N")
        .value_counts()
        .reset_index(),
        x="N",
        y="count",
    )
    fig_dist_fg.update_layout(autosize=False, width=800, height=500, template="seaborn")

    app.layout = html.Div(
        [
            html.Div(
                [dcc.Graph(figure=fig_dist_fg), dcc.Graph(figure=fig_dist_brics)],
                style={"display": "flex"},
            ),
            html.H2("Functional groups"),
            html.Div(
                [
                    dcc.Graph(figure=fig_sme_vs_shapley_fg),
                    dcc.Graph(figure=fig_shapley_vs_hn_fg),
                    dcc.Graph(figure=fig_sme_vs_hn_fg),
                ],
                style={"display": "flex"},
            ),
            html.H2("BRICS"),
            html.Div(
                [
                    dcc.Graph(figure=fig_sme_vs_shapley_brics),
                    dcc.Graph(figure=fig_shapley_vs_hn_brics),
                    dcc.Graph(figure=fig_sme_vs_hn_brics),
                ],
                style={"display": "flex"},
            ),
        ]
    )

    app.run(host="0.0.0.0", port="8888")
