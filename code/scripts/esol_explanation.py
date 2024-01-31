import base64
import sys
import time
from io import BytesIO
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import XAIChem
from dash import Dash, html
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
    models = XAIChem.loadModels(model, paths, device="cuda")

    for model in models:
        model.to(device)

    # Load data for explanation
    molecules = pd.read_csv("../../data/ESOL/ESOL.csv")
    masks = XAIChem.substructures.functionalGroupMasks(molecules.smiles, inverse=True)

    molecule_smiles = "CC1CCC(C(C1)O)C(C)C"
    masks = XAIChem.substructures.functionalGroupMasks(molecule_smiles, inverse=True)

    rdmol = Chem.MolFromSmiles(molecule_smiles)
    molecule = XAIChem.createDataObjectFromRdMol(rdmol, np.inf)
    # img_str = pillTob64(XAIChem.showMolecule(rdmol))
    # app.layout = html.Div([html.Img(src=f"data:image/png;base64,{img_str}")])

    print(XAIChem.predict(molecule, models))

    mol_masks = masks["mask"].tolist()
    scaffold_mask = torch.stack(mol_masks, dim=0).sum(dim=0) ^ 1
    mol_masks.append(scaffold_mask)
    mol_masks = torch.stack(mol_masks, dim=0)

    molecule_batch = [molecule for _ in range(2 ** len(mol_masks) - 1)]
    molecule_batch = DataLoader(molecule_batch, batch_size=256)

    mask_batch = []
    for indices in tqdm(list(XAIChem.graph.powerset(list(range(len(mol_masks)))))):
        mask = torch.stack([mol_masks[i, :] for i in indices]).sum(dim=0)
        mask_batch.append(mask)

    mask_batch = torch.stack(mask_batch, dim=0).t()
    mask_batch.to(device)

    print(mask_batch.shape)

    predictions = []
    for i, batch in enumerate(molecule_batch):
        batch.to(device)
        mask_batch_subset = mask_batch[:, i * 256 : (i + 1) * 256]
        predictions.extend(
            XAIChem.predictBatch(
                batch,
                models,
                mask_batch_subset.t().contiguous().view(-1, 1),
                device,
            ).to("cpu")
        )
    N = tuple(range(len(mol_masks)))
    # g = list(zip(*molecule.edge_index))
    g = {(0, 1)}

    print(N)
    print(g)

    HN_value = XAIChem.attribution.hamiacheNavarroValue(
        N, np.asarray(predictions), g, 2 / len(mol_masks)
    )

    print(predictions)
    print(HN_value)
    print(sum(HN_value[: len(mol_masks)]))

    #    attributions = XAIChem.attribution.difference(models, masks)
    #
    #    print(attributions.head())

    # app.run("0.0.0.0", "8888")
