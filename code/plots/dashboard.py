import re

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
        dcc.Store(id="store-valuename", storage_type="memory"),
        html.Br(),
        html.P(
            "Explanation of graph neural networks",
            className="text-center text-2xl font-bold mb-5",
        ),
        dcc.Dropdown(
            id="data-dropdown",
            multi=False,
            value="../../data/ESOL/attribution_functional_groups.json::ESOL",
            options=[
                {
                    "label": "Expected solubility",
                    "value": "../../data/ESOL/attribution_functional_groups.json::ESOL",
                },
                {
                    "label": "Expected solubility BRICS",
                    "value": "../../data/ESOL/attribution_brics.json::ESOL",
                },
                {
                    "label": "Expected solubility around mean",
                    "value": "../../data/ESOL/attribution_functional_groups_around_mean.json::ESOL",
                },
                {
                    "label": "Expected solubility BRICS around mean",
                    "value": "../../data/ESOL/attribution_brics_around_mean.json::ESOL",
                },
                {
                    "label": "Expected solubility BRICS aqsoldb A",
                    "value": "../../data/aqsoldb_A/attribution_brics.json::aqsoldb_A",
                },
                {
                    "label": "Expected solubility functional groups aqsoldb A",
                    "value": "../../data/aqsoldb_A/attribution_functional_groups.json::aqsoldb_A",
                },
                #                {
                #                    "label": "Expected solubility BRICS aqsoldb B",
                #                    "value": "../../data/aqsoldb_B/attribution_brics.json",
                #                },
                #                {
                #                    "label": "Expected solubility functional groups aqsoldb B",
                #                    "value": "../../data/aqsoldb_B/attribution_functional_groups.json",
                #                },
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


@callback(
    Output("store-data", "data"),
    Output("store-valuename", "data"),
    Output("store-truevalues", "data"),
    Input("data-dropdown", "value"),
)
def loadData(value):
    filename, value_name = re.split("::", value)

    # Get the data directory where the attribution file is saved
    filepath_parts = re.split("/", filename)

    true_values = pd.read_csv(
        f"{'/'.join(filepath_parts[:-1])}/{value_name}.csv"
    ).to_dict("records")
    attributions = pd.read_json(filename).to_dict("records")

    return attributions, value_name, true_values


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
