import base64
from io import BytesIO
from typing import Dict, List, Set, Tuple

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


def reducedMolecularGraph(
    molecular_graph: List[Tuple[int]], groups: Dict[int, Tuple[int]]
) -> Set[Tuple[int]]:
    """
    Create a graph of molecular features (i.e. functional groups or other substructures)
    from the full molecular graph representation of a molecule.

    :param molecular_graph: list of bonded atoms
    :param groups: a dictionairy specifying which atoms are grouped together as a substructure,
        keys represent the group id and the value is a tuple of atom ids.
    """

    reduced_graph = set()

    # Unpack groups dictionairy to map each atom id to its corresponding group id
    group = {
        atom_id: group_id
        for group_id, atom_ids in groups.items()
        for atom_id in atom_ids
    }

    # If a bond has atoms from different groups, there exists an edge between those
    # corresponding groups. By convention use the smallest group number is the first
    # index.
    for i, j in molecular_graph:
        if group[i] != group[j]:
            reduced_graph.add(
                (
                    min(group[i], group[j]),
                    max(group[i], group[j]),
                )
            )

    return reduced_graph


if __name__ == "__main__":
    app = Dash(__name__)

    device = torch.device("cuda")
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

    # Load data for explanation
    molecules = pd.read_csv("../../data/ESOL/ESOL.csv")
    # masks = XAIChem.substructures.functionalGroupMasks(molecules.smiles, inverse=True)

    # molecule_smiles = "CC1CCC(C(C1)O)C(C)C"
    molecule_smiles = molecules["smiles"].iloc[0]
    masks = XAIChem.substructures.functionalGroupMasks(molecule_smiles, inverse=True)

    print(masks)

    difference = XAIChem.attribution.difference(models, masks)

    rdmol = Chem.MolFromSmiles(molecule_smiles)
    molecule = XAIChem.createDataObjectFromRdMol(rdmol, np.inf)
    img_str = pillTob64(XAIChem.showMolecule(rdmol, show_atom_indices=True))
    app.layout = html.Div([html.Img(src=f"data:image/png;base64,{img_str}")])

    print(XAIChem.predict(molecule, models))
    mol_masks = masks["mask"].tolist()
    mol_masks = torch.stack(mol_masks, dim=0)

    for atom in rdmol.GetAtoms():
        print(f"{atom.GetSymbol()}: {atom.GetIdx()}")

    # v_OH = XAIChem.predict(molecule, models, mol_masks[0, :].view(-1, 1))
    # v_scaffold = XAIChem.predict(molecule, models, mol_masks[1, :].view(-1, 1))
    # v_molecule = XAIChem.predict(molecule, models)
    # sh_OH = 1 / 2 * (v_molecule - v_scaffold + v_OH)
    # sh_scaffold = 1 / 2 * (v_molecule - v_OH + v_scaffold)

    # print(f"v(OH): {v_OH}")
    # print(f"v(scaffold): {v_scaffold}")
    # print(f"v(N): {v_molecule}")
    # print("#####")
    # print(f"sh(OH): {sh_OH}")
    # print(f"sh(scaffold): {sh_scaffold}")
    # print(f"sh_OH + sh_scaffold = {sh_OH + sh_scaffold}")

    molecule_batch = [molecule for _ in range(2 ** len(mol_masks) - 1)]
    molecule_batch = DataLoader(molecule_batch, batch_size=256)

    mask_batch = []
    for indices in tqdm(list(XAIChem.graph.powerset(list(range(len(mol_masks)))))):
        mask = torch.stack([mol_masks[i, :] for i in indices]).sum(dim=0)
        mask_batch.append(mask)

    mask_batch = torch.stack(mask_batch, dim=0).t()
    # mask_batch.to(device)

    for model in models:
        model.to(device)

    predictions = []
    for i, batch in enumerate(molecule_batch):
        batch.to(device)
        mask_batch_subset = mask_batch[:, i * 256 : (i + 1) * 256].to(device)

        predictions.extend(
            XAIChem.predictBatch(
                batch,
                models,
                mask_batch_subset.t().contiguous().view(-1, 1),
                device,
            ).to("cpu")
        )

    N = tuple(range(len(mol_masks)))
    g = reducedMolecularGraph(
        list(zip(*molecule.cpu().edge_index.numpy())), masks["atom_ids"].to_dict()
    )

    print(N)
    print(g)

    HN_value = XAIChem.attribution.hamiacheNavarroValue(
        N, np.asarray(predictions), g, 2 / (len(mol_masks) * 10)
    )

    Shapley_value = XAIChem.attribution.shapleyValue(
        N, np.asarray(predictions), 2 / (len(mol_masks) * 10)
    )

    print(
        f"Predictions: {[round(val.item(), 4) for val in predictions[:len(mol_masks)]]}"
    )

    print(f"difference: {np.round(difference['attribution'], 4)}")
    print(f"Sum difference: {np.round(sum(difference['attribution']), 4)}")

    print(f"HN-value: {np.round(HN_value[:len(mol_masks)], 4)}")
    print(f"Sum HN-values: {sum(HN_value[:len(mol_masks)]):.4f}")

    print(f"Shapley-value: {np.round(Shapley_value[:len(mol_masks)], 4)}")
    print(f"Sum Shapley-values: {sum(Shapley_value[:len(mol_masks)]):.4f}")

    difference["HN_value"] = HN_value[: len(mol_masks)]
    difference["Shapley_value"] = Shapley_value[: len(mol_masks)]

    print(difference)

    XAIChem.showMolecule(molecule)

    #    attributions = XAIChem.attribution.difference(models, masks)
    #
    #    print(attributions.head())

    # app.run("0.0.0.0", "8888")
