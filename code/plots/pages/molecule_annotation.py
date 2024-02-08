import base64
from io import BytesIO

import dash
import pandas as pd
import plotly.express as px
import XAIChem
from dash import Input, Output, callback, dcc, html
from rdkit import Chem

dash.register_page(__name__, name="Annotate molecules")


layout = html.Div(
    [dcc.Dropdown(id="smiles-dropdown"), html.Div(id="smiles-dropown-container")],
    className="bg-light p-4 m-2",
)


def pillTob64(image, env_format="png", **kwargs):
    buff = BytesIO()
    image.save(buff, format=env_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded


@callback(
    Output("smiles-dropdown", "options"),
    Output("smiles-dropdown", "value"),
    Input("store-data", "data"),
)
def getDropdownOptions(data):
    data = pd.DataFrame(data)
    options = data.molecule_smiles.unique()

    return options, options[0]


@callback(
    Output("smiles-dropown-container", "children"),
    Input("smiles-dropdown", "value"),
    Input("store-data", "data"),
)
def annotateMolecule(smiles, data):
    data = pd.DataFrame(data)

    attributions = data.query("molecule_smiles == @smiles").copy()

    # Each set of atom ids forming a functional group is mapped to
    # its corresponding attribution using a dictionairy. Since keys
    # in a dictionairy must be hashable use tuples instead of lists
    # for the atom ids
    attributions["atom_ids_tuple"] = attributions.atom_ids.apply(tuple)
    images = []
    for attribution in ["difference", "HN_value", "Shapley_value"]:
        img_str = XAIChem.showMolecule(
            Chem.MolFromSmiles(smiles),
            attributions.set_index("atom_ids_tuple")[attribution].round(4).to_dict(),
            width=550,
            height=350,
        )

        fig = px.bar(
            attributions,
            x=attribution,
            y="functional_group",
            barmode="group",
            width=550,
            height=450,
        )
        fig.add_vline(attributions[attribution].sum())
        fig.update_traces(width=0.1)
        fig.update_layout(margin={"b": 20, "l": 25, "t": 0, "r": 0})

        images.append(
            html.Center(
                [
                    html.H3(attribution, style={"text-align": "center"}),
                    html.P(
                        [
                            f"Sum attributions: {attributions[attribution].sum():.4f}",
                            html.Br(),
                            f"Prediction: {attributions['non_masked_prediction'].iloc[0]:.4f}",
                            html.Br(),
                            f"Difference from mean: {attributions['non_masked_prediction'].iloc[0] + 3.1271107:.4f}",
                        ]
                    ),
                    html.Img(src=img_str),
                    dcc.Graph(figure=fig),
                ],
                className="bg-light m-5",
                style={"align": "center", "justifyContent": "center"},
            )
        )

    return html.Div(images, style={"display": "flex"}, className="bg-light")
