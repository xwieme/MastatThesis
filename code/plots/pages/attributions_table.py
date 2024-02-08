import dash
import pandas as pd
from dash import Input, Output, callback, dash_table, html

dash.register_page(__name__, name="Attribution table")


layout = html.Div(
    [
        html.Div(id="data-table"),
    ],
    className="bg-light p-4 m-2 border-spacing-y-1",
)


@callback(Output("data-table", "children"), Input("store-data", "data"))
def createTable(data):
    data = pd.DataFrame(data)

    data = data[
        [
            "molecule_smiles",
            "functional_group",
            "difference",
            "HN_value",
            "Shapley_value",
        ]
    ]

    return dash_table.DataTable(
        id="data-table",
        columns=[
            {"name": "molecule_smiles", "id": "molecule_smiles"},
            {"name": "functional_group", "id": "functional_group"},
            {"name": "difference", "id": "difference"},
            {"name": "HN_value", "id": "HN_value"},
            {"name": "Shapley_value", "id": "Shapley_value"},
        ],
        data=data.round(4).to_dict("records"),
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        page_current=0,
        page_size=25,
        # className="border-spacing-y-1",
    )
