import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots

from .variables import PLOTLY_COLORS

dash.register_page(__name__, name="Attribution distributions")


layout = html.Div(
    [
        html.Div(id="box-plots"),
        dcc.Dropdown(id="functional-group-dropdown"),
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
        id_vars=["molecule_smiles", "functional_group"],
        value_vars=["difference", "HN_value", "Shapley_value"],
        value_name="attribution",
        var_name="method",
    )

    # Remove attribution of scaffold to increase visibility of the other distributions
    data_long = data_long.query("functional_group != 'scaffold'")

    box_plots = px.box(
        data_long,
        x="functional_group",
        y="attribution",
        color="method",
        color_discrete_sequence=px.colors.qualitative.G10,
    )

    box_plots.update_layout(
        margin={"b": 5, "l": 2, "r": 2, "t": 35},
    )

    return dcc.Graph(figure=box_plots)


@callback(
    Output("functional-group-dropdown", "options"),
    Output("functional-group-dropdown", "value"),
    Input("store-data", "data"),
)
def fillFunctionalGroupDrowpdown(data):
    data = pd.DataFrame(data)
    options = data.functional_group.unique()

    return options, options[0]


@callback(
    Output("histograms", "children"),
    Input("store-data", "data"),
    Input("functional-group-dropdown", "value"),
)
def createHistograms(data, functional_group):
    data = pd.DataFrame(data)

    # Format dataframe from wide to long
    data_long = pd.melt(
        data,
        id_vars=["molecule_smiles", "functional_group"],
        value_vars=["difference", "HN_value", "Shapley_value"],
        value_name="attribution",
        var_name="method",
    )

    # Filter data to selected functional group
    data_long = data_long.query("functional_group == @functional_group")

    fig = px.histogram(
        data_long,
        x="attribution",
        facet_col="method",
    )

    # return dcc.Graph(figure=fig)
    return dcc.Graph(figure=fig)
