import dash
import pandas as pd
import plotly.io as pio
from dash import Dash, Input, Output, callback, dcc, html

pio.templates.default = "plotly_white"

external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]

app = Dash(
    __name__, pages_folder="pages", use_pages=True, external_scripts=external_script
)

app.layout = html.Div(
    [
        dcc.Store(id="store-data", storage_type="memory"),
        dcc.Store(id="store-truevalues", storage_type="memory"),
        html.Br(),
        html.P(
            "Explanation of graph neural networks",
            className="text-center text-2xl font-bold mb-5",
        ),
        dcc.Dropdown(
            id="data-dropdown",
            multi=False,
            value="../../data/ESOL/attribution_no_mean.json",
            options=[
                {
                    "label": "Expected solubility",
                    "value": "../../data/ESOL/attribution_no_mean.json",
                }
            ],
            className="focus:ring-4 focus:outline-none font-medium rounded-lg text-sm px-5 py-2.5 text-center",
        ),
        html.Div(
            children=[
                dcc.Link(
                    page["name"],
                    href=page["relative_path"],
                    className="grid-rows-1 p-2 mx-4 my-2 border-1 rounded-lg bg-[#d2d3db]",
                )
                for page in dash.page_registry.values()
            ]
        ),
        dash.page_container,
    ],
    className="bg-[#fafafa] container mx-auto px-14 py-4",
)


@callback(Output("store-data", "data"), Input("data-dropdown", "value"))
def loadData(value):
    return pd.read_json(value).to_dict("records")


@callback(Output("store-truevalues", "data"), Input("data-dropdown", "value"))
def loadTrueData(value):
    return pd.read_csv("../../data/ESOL/ESOL.csv").to_dict("records")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8888", debug=True)
