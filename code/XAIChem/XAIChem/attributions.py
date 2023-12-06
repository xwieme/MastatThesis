from typing import List

import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.loader import DataLoader

from . import variables, prediction
from .data import createDataObjectFromRdMol
from .masks import createMask, removeAtoms


def functionalGroupAttributionScores(
    smiles: str,
    models: List[torch.nn.Module],
    method: str,
    functional_groups: List[str] = variables.FUNCTIONAL_GROUPS,
) -> pd.DataFrame:
    """
    Compute the attribution scores for all present functional groups.

    :param smiles: smiles represenation of the molecule
    :param functional_groups: list of functional groups to search for in the molecule
    :param method: determines the method to compute the attribution, either 'mask' to
        apply a mask after the molecular embedding, or 'structure' to modify the input
        structure
    """

    molecule = Chem.MolFromSmiles(smiles)
    graph = createDataObjectFromRdMol(molecule, -1)

    # Get atom indices of present functional groups
    functional_groups_matches = dict()
    for functional_group in functional_groups:
        for match in molecule.GetSubstructMatches(Chem.MolFromSmarts(functional_group)):
            functional_groups_matches[match] = functional_group

    # Quit if no matches are found
    if len(functional_groups_matches) == 0:
        return

    if method == "mask":
        masks = torch.cat(
            [
                createMask(molecule, substructure)
                for substructure in functional_groups_matches.keys()
            ]
        )

        # Create a batch with size of the number of functional groups that are present
        graphs = DataLoader(
            [graph for _ in range(len(functional_groups_matches))],
            batch_size=len(functional_groups_matches),
        )

        for data in graphs:
            pred = prediction.predictBatch(data, models)
            pred_masked = prediction.predictBatch(
                data, models, masks.view(-1, 1)
            )

    elif method == "structure":
        modified_molecules = list()
        substructures = list(functional_groups_matches.keys())

        for atom_ids in substructures:
            modified_molecule = removeAtoms(molecule, atom_ids)

            if len(modified_molecule.GetAtoms()) > 1:
                modified_molecules.append(modified_molecule)
            else:
                del functional_groups_matches[atom_ids]

        # Stop if remaining structure is a single atom
        if len(modified_molecules) == 0:
            return

        graphs = DataLoader(
            [createDataObjectFromRdMol(mol, -1) for mol in modified_molecules],
            batch_size=len(modified_molecules),
        )

        pred = prediction.predict(graph, models)
        for data in graphs:
            pred_masked = prediction.predictBatch(data, models)

    results = pd.DataFrame(
        {
            "smiles": [smiles for _ in range(len(functional_groups_matches))],
            "functional_group": functional_groups_matches.values(),
            "substructure": functional_groups_matches.keys(),
            "attribution": pred - pred_masked,
            "prediction": pred,
            "prediction_masked": pred_masked,
        }
    )

    return results
