import re
from collections import defaultdict, deque
from typing import Iterable, List

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import BRICS

from ..masks import createMask


def BRICSMasks(smiles: str):
    """
    Given a list of atom ids which are are part of a substructure, return a set
    of the bond ids between those atoms.

    :param smiles: smiles representation of a molecule
    :param atom_ids: rdkit ids of atoms in the substructure
    """
    molecule = Chem.MolFromSmiles(smiles)

    # Break the molecule on all BRICS bond and retreiv their molecule
    # objects and list of atom ids of each substructure
    broken_mol = BRICS.BreakBRICSBonds(molecule)
    substructures = Chem.GetMolFrags(broken_mol, asMols=True)
    substructures_atom_ids = Chem.GetMolFrags(broken_mol)

    masks = defaultdict(list)
    for i, substruct_atom_ids in enumerate(substructures_atom_ids):
        # Get the substructure smiles representation where the linking
        # atom is removed
        substruct_smiles = Chem.MolToSmiles(substructures[i])
        substruct_smiles = re.sub(r"\[\d+\*\]|\(\[\d+\*\]\)", "", substruct_smiles)

        # Remove the atom id of the linking atom
        substruct_atom_ids = tuple(
            atom_id
            for atom_id in substruct_atom_ids
            if broken_mol.GetAtomWithIdx(atom_id).GetSymbol() != "*"
        )

        masks["molecule_smiles"].append(smiles)
        masks["substruct_smiles"].append(substruct_smiles)
        masks["atom_ids"].append(substruct_atom_ids)
        masks["mask"].append(createMask(molecule, substruct_atom_ids))

    return pd.DataFrame.from_dict(masks)


# def BRICSMasks(smiles: str):
#     """
#     Given a list of atom ids which are are part of a substructure, return a set
#     of the bond ids between those atoms.
#
#     :param smiles: smiles representation of a molecule
#     :param atom_ids: rdkit ids of atoms in the substructure
#     """
#
#     molecule = Chem.MolFromSmiles(smiles)
#
#     bricks_masks = defaultdict(list)
#
#     for brics_bond in BRICS.FindBRICSBonds(molecule):
#         # Divide the molecule into two parts by breaking the BRICS bond
#         substructures = breakBRICKSBond(molecule, brics_bond[0])
#
#         # Create a mask for both substructures, by taking into account that
#         # they are complementary to eachother
#         substructure_mask = createMask(molecule, substructures[0])
#         complementary_substructure_mask = (
#             torch.ones(substructure_mask.shape) - substructure_mask
#         ).int()
#
#         bricks_masks["molecule_smiles"].append(smiles)
#         bricks_masks["atom_ids"].append(substructures[-1])
#         bricks_masks["mask"].append(substructure_mask)
#         # bricks_masks["molecule_smiles"].extend([smiles, smiles])
#         # bricks_masks["atom_ids"].extend(substructures[::-1])
#         # bricks_masks["mask"].extend(
#         #     [substructure_mask, complementary_substructure_mask]
#         # )
#
#     return pd.DataFrame.from_dict(bricks_masks)


def breakBRICKSBond(molecule: Chem.rdchem.Mol, bond: Iterable[int]) -> List[List[int]]:
    """
    Create two lists each containing the atom ids of the substructures resulting
    when the given BRICKS bond is broken.

    :param molecule: molecule where the BRICKS bond needs to be broken
    :param bond: tuple of the atom ids of the BRICKS bond
    """
    substructures = list()

    # Breadth-first search (BFS) is used to find all atom ids that are part of
    # the substructure when the BRICKS bond is broken. BFS starts at the atom
    # ids of the BRICK bond and ignores the other one to break the molecular
    # graph into two substructures.
    for break_atom_id in bond:
        substructure_atom_ids = list()
        substructure_atom_ids.append(break_atom_id)
        break_atom = molecule.GetAtomWithIdx(break_atom_id)

        atoms_to_visit = deque()
        atoms_to_visit.append(break_atom)

        while len(atoms_to_visit) != 0:
            current_atom = atoms_to_visit.popleft()

            for neighbor_atom in current_atom.GetNeighbors():
                neightbor_atom_id = neighbor_atom.GetIdx()
                if neightbor_atom_id not in [*bond, *substructure_atom_ids]:
                    substructure_atom_ids.append(neightbor_atom_id)
                    atoms_to_visit.append(neighbor_atom)

        substructures.append(substructure_atom_ids)

    return substructures
