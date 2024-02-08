import dash
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html

dash.register_page(__name__, name="Data set")


layout = html.Div(
    [
        html.Div(
            [html.Div(id="graph_occurence_functional_groups")],
            className="row",
        ),
    ],
    className="bg-light p-4 m-2",
)


@callback(
    Output("graph_occurence_functional_groups", "children"), Input("store-data", "data")
)
def plotOccurenceFunctionalGroups(data):
    data = pd.DataFrame(data)

    counts_df = (
        data.query("functional_group != 'scaffold'")
        .functional_group.value_counts()
        .reset_index()
    )

    fig = px.bar(counts_df, x="count", y="functional_group")

    return dcc.Graph(figure=fig)
