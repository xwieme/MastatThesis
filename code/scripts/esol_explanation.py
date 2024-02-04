import base64
import time
from io import BytesIO
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import XAIChem
from dash import Dash, dcc, html
from rdkit import Chem
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def pillTob64(image, env_format="png", **kwargs):
    buff = BytesIO()
    image.save(buff, format=env_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded


if __name__ == "__main__":
    app = Dash(__name__)

    device = torch.device("cpu")
    # Get model architecture and configuration
    model, config = XAIChem.models.PreBuildModels.rgcnWuEtAll(
        "esol_reproduction.yaml", ["seed"], model_id=0
    )

    # Load trained models
    paths = [
        f"../../data/ESOL/trained_models/ESOL_rgcn_model_{i}_early_stop.pt"
        for i in range(10)
    ]
    models = XAIChem.loadModels(model, paths, device="cpu")

    # Load data for explanation
    molecules = pd.read_csv("../../data/ESOL/ESOL.csv")
    # masks = XAIChem.substructures.functionalGroupMasks(molecules.smiles, inverse=True)

    # # Compute mean prediction
    # molecules_data = [
    #     XAIChem.createDataObjectFromSmiles(smiles, np.inf)
    #     for smiles in molecules.smiles
    #     if len(smiles) > 1
    # ]
    # predictions = [
    #     XAIChem.predict(data, models=models) for data in tqdm(molecules_data)
    # ]
    # print(np.mean(predictions))

    t1 = time.time()

    # molecule_smiles = "CC1CCC(C(C1)O)C(C)C"
    molecule_smiles = molecules["smiles"].iloc[0]
    # masks = XAIChem.substructures.functionalGroupMasks(molecule_smiles)
    inverse_masks = XAIChem.substructures.functionalGroupMasks(
        molecule_smiles, inverse=True
    )

    attributions = XAIChem.attribution.difference(models, inverse_masks, device)
    attributions_gametheory = XAIChem.attribution.hamiacheNavarroValue(
        models, molecule_smiles, inverse_masks, -3.1271107, shapley=True, device=device
    )

    rdmol = Chem.MolFromSmiles(molecule_smiles)
    molecule = XAIChem.createDataObjectFromRdMol(rdmol, np.inf)

    prediction = XAIChem.predict(molecule, models, device=device)
    print(f"Estimated solubility: {prediction}")
    print(f"Difference from mean expected solubility: {prediction + 3.1271107}")

    print(time.time() - t1)

    # mol_masks = inverse_masks["mask"].tolist()
    # mol_masks = torch.stack(mol_masks, dim=0)

    # # for atom in rdmol.GetAtoms():
    # #     print(f"{atom.GetSymbol()}: {atom.GetIdx()}")

    # # v_OH = XAIChem.predict(molecule, models, mol_masks[0, :].view(-1, 1))
    # # v_scaffold = XAIChem.predict(molecule, models, mol_masks[1, :].view(-1, 1))
    # # v_molecule = XAIChem.predict(molecule, models)
    # # sh_OH = 1 / 2 * (v_molecule - v_scaffold + v_OH)
    # # sh_scaffold = 1 / 2 * (v_molecule - v_OH + v_scaffold)

    # # print(f"v(OH): {v_OH}")
    # # print(f"v(scaffold): {v_scaffold}")
    # # print(f"v(N): {v_molecule}")
    # # print("#####")
    # # print(f"sh(OH): {sh_OH}")
    # # print(f"sh(scaffold): {sh_scaffold}")
    # # print(f"sh_OH + sh_scaffold = {sh_OH + sh_scaffold}")

    # molecule_batch = [molecule for _ in range(2 ** len(mol_masks) - 1)]
    # molecule_batch = DataLoader(molecule_batch, batch_size=256)

    # mask_batch = []
    # for indices in tqdm(list(XAIChem.graph.powerset(list(range(len(mol_masks)))))):
    #     mask = torch.stack([mol_masks[i, :] for i in indices]).sum(dim=0)
    #     mask_batch.append(mask)

    # mask_batch = torch.stack(mask_batch, dim=0).t()
    # # mask_batch.to(device)

    # for model in models:
    #     model.to(device)

    # predictions = []
    # for i, batch in enumerate(molecule_batch):
    #     batch.to(device)
    #     mask_batch_subset = mask_batch[:, i * 256 : (i + 1) * 256].to(device)

    #     predictions.extend(
    #         XAIChem.predictBatch(
    #             batch,
    #             models,
    #             mask_batch_subset.t().contiguous().view(-1, 1),
    #             device,
    #         ).to("cpu")
    #     )

    # N = tuple(range(len(mol_masks)))
    # g = reducedMolecularGraph(
    #     list(zip(*molecule.cpu().edge_index.numpy())),
    #     inverse_masks["atom_ids"].to_dict(),
    # )

    # print(N)
    # print(g)

    # HN_value = XAIChem.attribution.hamiacheNavarroValue(
    #     N, np.asarray(predictions) + 3.1271107, g, 2 / (len(mol_masks) * 10)
    # )

    # Shapley_value = XAIChem.attribution.shapleyValue(
    #     N, np.asarray(predictions) + 3.1271107, 2 / (len(mol_masks) * 10)
    # )

    print(f"difference: {np.round(attributions['difference'], 4)}")
    print(f"Sum difference: {np.round(sum(attributions['difference']), 4)}")

    num_substructures = attributions_gametheory["mask"].shape[0]
    print(
        f"HN-value: {np.round(attributions_gametheory.HN_value[:num_substructures], 4)}"
    )
    print(
        f"Sum HN-values: {sum(attributions_gametheory.HN_value[:num_substructures]):.4f}"
    )

    print(
        f"Shapley-value: {np.round(attributions_gametheory.Shapley_value[:num_substructures], 4)}"
    )
    print(
        f"Sum Shapley-values: {sum(attributions_gametheory.Shapley_value[:num_substructures]):.4f}"
    )

    attributions["HN_value"] = attributions_gametheory.HN_value[:num_substructures]
    attributions["Shapley_value"] = attributions_gametheory.Shapley_value[
        :num_substructures
    ]

    print(attributions)
    print(attributions.set_index("atom_ids").difference.round(4).to_dict())

    images = []
    for attribution in ["difference", "HN_value", "Shapley_value"]:
        img_str = pillTob64(
            XAIChem.showMolecule(
                rdmol,
                attributions.set_index("atom_ids")[attribution].round(4).to_dict(),
                width=800,
                height=600,
            )
        )

        fig = px.bar(
            attributions,
            x=attribution,
            y="functional_group_smarts",
            barmode="group",
            template="plotly_white",
        )
        fig.add_vline(attributions[attribution].sum())

        images.append(
            html.Div(
                [
                    html.H1(attribution, style={"text-align": "center"}),
                    html.Img(src=f"data:image/png;base64,{img_str}"),
                    dcc.Graph(figure=fig),
                ]
            )
        )

    molecules["solubility"] = molecules["ESOL"].apply(np.exp)
    fig = px.histogram(molecules, x="solubility")
    app.layout = html.Div([*images, dcc.Graph(figure=fig)], style={"display": "flex"})

    app.run("0.0.0.0", "8888")
