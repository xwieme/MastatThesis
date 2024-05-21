import dash
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots

from .variables import PLOTLY_COLORS

dash.register_page(__name__, name="Attribution distributions")


layout = html.Div(
    [
        html.Div(id="box-plots"),
        dcc.Dropdown(id="substruct-smiles-dropdown"),
        html.Div(id="histograms"),
    ],
    className="bg-light p-4 m-2 grid-col-1 z-5",
)


@callback(Output("box-plots", "children"), Input("store-data", "data"))
def createBoxPlots(data):
    data = pd.DataFrame(data)

    # Format dataframe from wide to long
    data_long = pd.melt(
        data,
        id_vars=["molecule_smiles", "substruct_smiles"],
        value_vars=["SME", "HN_value", "Shapley_value"],
        value_name="attribution",
        var_name="method",
    )

    # Remove attribution of scaffold to increase visibility of the other distributions
    data_long = (
        data_long.query("substruct_smiles != 'scaffold'")
        .groupby("substruct_smiles")
        .apply(lambda group: group if len(group) > 5 else None)
    )

    selected_groups = [
        "ROH",
        "R-C(=O)OCH3",
        "R-OMe",
        "R-OEt",
        "R-tBu",
    ]

    box_plots = px.box(
        data_long.query("substruct_smiles in @selected_groups"),
        x="substruct_smiles",
        y="attribution",
        color="method",
        color_discrete_sequence=px.colors.qualitative.G10,
        category_orders={"substruct_smiles": selected_groups},
    )

    box_plots.update_layout(
        margin={"b": 5, "l": 2, "r": 2, "t": 35},
    )
    box_plots.update_layout(
        autosize=False,
        width=900,
        height=500,
        font={"size": 20, "family": "Times New Roman"},
        xaxis={"title": ""},
    )

    # box_plots.update_traces(
    #     x=[x[:7] for x in data_long["substruct_smiles"].drop_duplicates()]
    # )

    return dcc.Graph(figure=box_plots)


@callback(
    Output("substruct-smiles-dropdown", "options"),
    Output("substruct-smiles-dropdown", "value"),
    Input("store-data", "data"),
)
def fillFunctionalGroupDrowpdown(data):
    data = pd.DataFrame(data)
    options = data.substruct_smiles.unique()

    return options, options[0]


@callback(
    Output("histograms", "children"),
    Input("store-data", "data"),
    Input("substruct-smiles-dropdown", "value"),
)
def createHistograms(data, substruct_smiles):
    data = pd.DataFrame(data)

    # Format dataframe from wide to long
    data_long = pd.melt(
        data,
        id_vars=["molecule_smiles", "substruct_smiles"],
        value_vars=["SME", "HN_value", "Shapley_value"],
        value_name="attribution",
        var_name="method",
    )

    # Filter data to selected functional group
    data_long = data_long.query("substruct_smiles == @substruct_smiles")

    fig = px.histogram(
        data_long,
        x="attribution",
        facet_col="method",
    )

    #    fig = ff.create_distplot([
    #        data_long.query("method == 'SME'").attribution.to_list(),
    #        data_long.query("method == 'HN_value'").attribution.to_list(),
    #        data_long.query("method == 'Shapley_value'").attribution.to_list()
    #    ], ["SME", "HN_value", "Shapley_value"])

    return dcc.Graph(figure=fig)
    # return dcc.Graph(figure=fig)
