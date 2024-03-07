import base64
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dash_table, dcc, html
from rdkit import Chem

import XAIChem

data = pd.read_json("../../data/ESOL/attribution.json")


def pillTob64(image, env_format="png", **kwargs):
    buff = BytesIO()
    image.save(buff, format=env_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded


@callback(Output("dd-output-container", "children"), Input("smiles-dropdown", "value"))
def colorMolecule(value):
    rdmol = Chem.MolFromSmiles(value)
    attributions = data.query("molecule_smiles == @value")
    attributions["atom_ids"] = attributions["atom_ids"].apply(tuple)
    images = []
    for attribution in ["difference", "HN_value", "Shapley_value"]:
        img_str = pillTob64(
            XAIChem.showMolecule(
                rdmol,
                attributions.set_index("atom_ids")[attribution].round(4).to_dict(),
                width=350,
                height=350,
            )
        )

        fig = px.bar(
            attributions,
            x=attribution,
            y="functional_group_smarts",
            barmode="group",
            template="plotly_white",
            width=550,
            height=450,
        )
        fig.add_vline(attributions[attribution].sum())

        images.append(
            html.Div(
                [
                    html.H3(attribution, style={"text-align": "center"}),
                    html.P(f"Sum attributions: {attributions[attribution].sum()}"),
                    html.P(
                        f"Prediction: {attributions['non_masked_prediction'].iloc[0]:.4f}"
                    ),
                    html.P(
                        f"Difference from mean: {attributions['non_masked_prediction'].iloc[0] + 3.1271107:.4f}",
                        style={"padding": "-5px"},
                    ),
                    html.Img(
                        src=f"data:image/png;base64,{img_str}",
                    ),
                    dcc.Graph(figure=fig),
                ],
            )
        )

    return html.Div(images, style={"display": "flex"})


if __name__ == "__main__":
    app = Dash(__name__)

    # Translate smarts to a more readable format
    functional_group_titles = [
        "-tBu",
        "-SH",
        "-CF3",
        "-SCH3",
        "-C#CH",
        "-X",
        "ethoxy",
        "-CN",
        "-N(=O)O",
        "-O-C",
        "-COOCH3",
        "-OH",
        "-C(=O)N",
        "-NH2",
        "-C(=O)CH3",
        "-S(=O)(=O)-NH2",
        "-C(=O)OH",
    ]

    functional_group_titles_dict = dict(
        zip(XAIChem.variables.FUNCTIONAL_GROUPS, functional_group_titles)
    )

    data["functional_group_title"] = data["functional_group_smarts"].apply(
        lambda smarts: functional_group_titles_dict[smarts]
        if smarts in functional_group_titles_dict.keys()
        else smarts
    )

    # Create a datatable to display the raw results
    columns = [
        {"name": "molecule_smiles", "id": "molecule_smiles"},
        {"name": "functional_group_smarts", "id": "functional_group_smarts"},
        {"name": "difference", "id": "difference"},
        {"name": "HN_value", "id": "HN_value"},
        {"name": "Shapley_value", "id": "Shapley_value"},
        {"name": "non_masked_prediction", "id": "non_masked_prediction"},
    ]

    data_table_layout = html.Div(
        [
            dash_table.DataTable(
                id="datatable",
                columns=columns,
                data=data.to_dict("records"),
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_action="native",
                page_current=0,
                page_size=5,
                style_as_list_view=True,
                style_cell={"padding": "5px"},
                style_header={
                    "backgroundColor": "#415a77",
                    "fontWeight": "bold",
                    "color": "#e0e1dd",
                },
            ),
            html.Div(id="datatable-container"),
        ],
    )

    # Visualize the distribution of the different attribution methods for
    # each functional group using a violin plot
    data_long = pd.melt(
        data,
        id_vars=["molecule_smiles", "functional_group_title", "non_masked_prediction"],
        value_vars=["difference", "HN_value", "Shapley_value"],
        value_name="attribution",
        var_name="method",
    )

    data_long = data_long.query("functional_group_title != 'scaffold'")

    figures = []

    for method in data_long.method.unique():
        fig = go.Figure()

        # Split positive and negative attribution scores
        fig.add_traces(
            [
                go.Violin(
                    x=data_long[
                        (data_long.attribution > 0) & (data_long.method == method)
                    ]["attribution"],
                    y=data_long[
                        (data_long.attribution > 0) & (data_long.method == method)
                    ]["functional_group_title"],
                    spanmode="hard",
                    name="positive",
                    side="positive",
                ),
                go.Violin(
                    x=data_long[
                        (data_long.attribution < 0) & (data_long.method == method)
                    ]["attribution"],
                    y=data_long[
                        (data_long.attribution < 0) & (data_long.method == method)
                    ]["functional_group_title"],
                    spanmode="hard",
                    name="negative",
                    side="negative",
                ),
            ]
        )
        fig.update_traces(scalemode="count", width=1.5, orientation="h")
        fig.update_layout(
            width=500,
            height=600,
            violingap=0,
            violinmode="overlay",
            violingroupgap=0,
            template="plotly_white",
            margin={"b": 5, "l": 2, "r": 2, "t": 1},
        )

        figures.append(
            html.Div(
                [html.H3(method), dcc.Graph(figure=fig)], style={"text-align": "center"}
            )
        )

    attribution_distribution = html.Div(figures, style={"display": "flex"})

    # For a selected molecule, color its substructures according to the attribution scores.
    dropdown = html.Div(
        [
            dcc.Dropdown(
                data.molecule_smiles.unique(),
                data.molecule_smiles.iloc[0],
                id="smiles-dropdown",
            ),
            html.Div(id="dd-output-container"),
        ]
    )

    counts = (
        data["functional_group_title"]
        .value_counts()
        .to_frame()
        .reset_index()
        .query("functional_group_title != 'scaffold'")
    )
    bar_plot = px.bar(counts, x="count", y="functional_group_title")
    bar_plot.update_layout(
        width=500,
        height=600,
        template="plotly_white",
        margin={"b": 5, "l": 2, "r": 2, "t": 5},
    )

    num_subgroups_df = (
        data.groupby("molecule_smiles")
        .functional_group_smarts.count()
        .to_frame("num_substructures")
        .reset_index()
    )
    benchmark_df = (
        pd.merge(
            data[
                [
                    "molecule_smiles",
                    "time_difference",
                    "time_HN_value",
                    "time_Shapley_value",
                ]
            ].drop_duplicates(),
            num_subgroups_df,
            on="molecule_smiles",
        )
        .drop("molecule_smiles", axis=1)
        .groupby("num_substructures")
        .mean()
        .reset_index()
    )

    benchmark_df_long = pd.melt(
        benchmark_df,
        id_vars="num_substructures",
        value_name="time",
        var_name="method",
    )

    benchmark_plot = px.bar(
        benchmark_df_long,
        x="num_substructures",
        y="time",
        color="method",
        barmode="group",
        template="plotly_white",
        width=600,
        height=600,
    )

    app.layout = html.Div(
        [
            html.Div(
                [
                    attribution_distribution,
                    html.Div(
                        [dcc.Graph(figure=bar_plot), dcc.Graph(figure=benchmark_plot)],
                        style={"display": "flex"},
                    ),
                ]
            ),
            html.Div([data_table_layout, dropdown]),
        ],
        style={"display": "flex"},
    )

    app.run("0.0.0.0", "8888")
