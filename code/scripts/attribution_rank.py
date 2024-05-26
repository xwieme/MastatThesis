import argparse
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import scipy
from dash import Dash, dcc, html
from rdkit import Chem
from rdkit.Chem import Draw

DATA_DIR = "../../data"


def correlationPlot(
    data: pd.DataFrame, rmse_df: pd.DataFrame, method_1: str, method_2: str, title: str
):
    """
    Plot the Spearman rank correlation between attributions of different
    methods in function of the prediction RMSE

    :param corr_df: dataframe containing the correlations
    :param rmse_df: dataframe containing the RMSE between the prediction and
        experimental value
    :param title: plot title
    """

    # Create a dataframe indicating the number of substructures each molecule has
    filter_df = data.groupby("molecule_smiles").apply(len)

    # Compute the Spearman rank correlation between the substructures attributions
    # computed using two different methods per molecule.
    corr_df = (
        filter_df.to_frame("N_sub")
        .query("N_sub > 1")  # Remove molecules that cannot be split into substructures
        .join(data.set_index("molecule_smiles"))
        .round(6)
        .groupby("molecule_smiles")[
            [method_1, method_2]
        ]  # compute the correlation for each molecule
        .corr(method="spearman")[[method_2]]
        .reset_index()
        .query("level_1 == @method1")  # Convert correlation matrix to dataframe
        .drop(columns="level_1", axis=0)
        .rename(columns={method_2: "corr"})
        .set_index("molecule_smiles")
        .join(filter_df.to_frame("N_substructures"))  # Add the number of substructures
    )

    print("#" * 50)
    print(
        f"{method_1} vs {method_2} ({len(corr_df.query('corr == 1')) / len(corr_df) * 100:.2f}% agreement)"
    )
    print("#" * 50)

    print(corr_df.query("corr < 1.0").join(rmse_df.set_index("smiles")[["RMSE"]]))

    # corr_df["N_substructures"] = np.log(corr_df.N_substructures)

    fig = px.scatter(
        corr_df.join(rmse_df.set_index("smiles")[["RMSE"]]),
        x="corr",
        y="RMSE",
        color="N_substructures",
        color_continuous_scale="inferno",
        title=title,
    )

    fig.add_hline(0.6, line_width=3, line_dash="dash", line_color="red")
    fig.update_yaxes(title="Absolute error log(mol/L)")
    fig.update_xaxes(title="Spearman rank correlation")

    fig.update_layout(
        autosize=False,
        width=800,
        height=500,
        template="plotly_white",
        font=dict(family="times new roman", size=22),
    )

    return fig


def inspectMolecules(df: pd.DataFrame):
    """
    Visualize molecules present in the given dataframe.

    :param df: dataframe containing molecule smiles in the column 'smiles'
    """
    rdmols = [Chem.MolFromSmiles(smiles_str) for smiles_str in df.smiles]
    img = Draw.MolsToGridImage(rdmols, molsPerRow=5, subImgSize=(500, 500))

    return img


def numSubstructDistribution(df: pd.DataFrame):
    """
    Create a bar chart of the number of substructures of molecules
    """

    # Compute average number of substructures per molecule
    n_substruct = df.groupby("molecule_smiles").size().to_frame("N_substructs")

    fig = px.bar(
        n_substruct.value_counts().to_frame("count").reset_index(),
        x="N_substructs",
        y="count",
    )
    fig.update_layout(autosize=False, width=800, height=500, template="plotly_white")
    fig.update_xaxes(
        tickvals=np.arange(n_substruct.N_substructs.max()),
        ticktext=np.arange(n_substruct.N_substructs.max()),
    )

    return fig, n_substruct


if __name__ == "__main__":
    app = Dash()

    # Setup argparse to get the subdirectory name of interest in the data directory
    parser = argparse.ArgumentParser()
    parser.add_argument("data_subdir_name")
    parser.add_argument("--suffix", type=str, dest="suffix", default="")
    args = parser.parse_args()

    ###############
    ## Load Data ##
    ###############

    dataset = pd.read_csv(
        os.path.join(DATA_DIR, args.data_subdir_name, f"{args.data_subdir_name}.csv")
    )
    attributions_functional_groups = pd.DataFrame(
        pd.read_json(
            os.path.join(
                DATA_DIR,
                args.data_subdir_name,
                f"attribution_functional_groups{args.suffix}.json",
            )
        )
    )
    attributions_brics = pd.DataFrame(
        pd.read_json(
            os.path.join(
                DATA_DIR, args.data_subdir_name, f"attribution_brics{args.suffix}.json"
            )
        )
    )

    ######################
    ## Prediction Error ##
    ######################

    # Compute the RMSE between the model prediction and experimental data
    dataset = dataset.join(
        attributions_brics[["molecule_smiles", "non_masked_prediction"]]
        .drop_duplicates()
        .set_index("molecule_smiles")
        .rename(columns={"non_masked_prediction": "prediction"}),
        on="smiles",
    )
    dataset["RMSE"] = np.sqrt(
        (dataset[args.data_subdir_name] - dataset["prediction"]) ** 2
    )

    # Get the fraction of molecules where the prediction is larger than
    # the experimental RMSE of 0.6 logS
    print(
        f"Fraction of molecules with a RMSE > 0.6: {len(dataset.query('RMSE > 0.6')) / len(dataset) * 100:.2f}%"
    )

    ################################
    ## substructures Distribution ##
    ################################

    # Create a bar chart of the distribution of number of substructure per molecule
    fig_num_functional_groups, num_functional_groups = numSubstructDistribution(
        attributions_functional_groups
    )
    fig_num_brics, num_brics = numSubstructDistribution(attributions_brics)

    # Combine the different substructure methods by selecting the method
    # which produces the most substructures.
    indicator_df = (
        num_functional_groups.join(
            num_brics,
            lsuffix="_fg",
            rsuffix="_brics",
        )
        .apply(
            lambda row: "fg"
            if row["N_substructs_fg"] >= row["N_substructs_brics"]
            else "brics",
            axis=1,
        )
        .to_frame("method")
    )

    attributions_combined = pd.concat(
        [
            indicator_df.query("method == 'brics'").join(
                attributions_brics.set_index("molecule_smiles")
            ),
            indicator_df.query("method == 'fg'").join(
                attributions_functional_groups.set_index("molecule_smiles")
            ),
        ],
    ).reset_index()

    # Create bar chart of distribution of the number of substructure per molecule
    # in the attributions_combined dataframe
    fig_num_combined, _ = numSubstructDistribution(attributions_combined)

    #############################
    ## Attribution Correlation ##
    #############################

    # Compute the Spearman rank correlation between different attribution
    # metrics for each molecule that can be split into substructures
    correlation_figures = defaultdict(list)
    attributions_dfs = {
        #  "functional groups": attributions_functional_groups,
        #  "brics": attributions_brics,
        "combined": attributions_combined,
    }

    # print(attributions_functional_groups.N_substructures.value_counts())
    # print(attributions_brics.N_substructures.value_counts())
    # print(attributions_combined.N_substructures.value_counts())

    # Plot for each substructure method the Spearman rank correlation between
    # two different attribution techniques per molecule in function of the prediction
    # rmse
    for substruct_method, attribution_df in attributions_dfs.items():
        for method1, method2 in combinations(["SME", "Shapley_value", "HN_value"], 2):
            print()
            print("-" * 50)
            print(f"{substruct_method:>25}")
            print("-" * 50)
            print()
            correlation_figures[substruct_method].append(
                correlationPlot(
                    attribution_df,
                    dataset,
                    method1,
                    method2,
                    f"{method1} vs {method2}",
                )
            )

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Functional groups substructures"),
                            dcc.Graph(figure=fig_num_functional_groups),
                        ]
                    ),
                    html.Div(
                        [
                            html.H3("BRICS substructures"),
                            dcc.Graph(figure=fig_num_brics),
                        ]
                    ),
                    html.Div(
                        [
                            html.H3("Combined substructures"),
                            dcc.Graph(figure=fig_num_combined),
                        ]
                    ),
                ],
                style={"display": "flex"},
            ),
            *[
                html.Div(
                    [
                        substruct_method,
                        html.Div(
                            [dcc.Graph(figure=fig) for fig in figures],
                            style={"display": "flex"},
                        ),
                    ]
                )
                for substruct_method, figures in correlation_figures.items()
            ],
        ]
    )

    app.run(host="0.0.0.0", port="8073")
