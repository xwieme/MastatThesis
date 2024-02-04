from collections import defaultdict
from typing import List

import pandas as pd
import torch
from rdkit.Chem import MolFromSmarts, MolFromSmiles

import XAIChem


def functionalGroupMasks(
    smiles: str,
    functional_groups: List[str] | None = None,
    inverse: bool = False,
) -> pd.DataFrame:
    """
    Get all functional groups of the given molecule in smiles represenation and
    create a mask for each functional group. The mask isolates a functional group
    from the rest of the molecule. To use the mask to select the corresponding
    functional group use inverse = True.

    :param smiles: smiles represenation of the molecule
    :param functional_groups: optional list of functional groups to search for in the molecule.
        (default is None, i.e. a default set will be used)
    :param inverse: Set to true to select the corresponding functional group instead
        of masking it. (default is False)
    """

    if functional_groups is None:
        functional_groups = XAIChem.variables.FUNCTIONAL_GROUPS

    molecule = MolFromSmiles(smiles)

    # Keep track of atoms that are note included in a mask to generate
    # a mask for the scaffold, i.e. main structure of the molecule
    not_masked_atom_ids = [atom.GetIdx() for atom in molecule.GetAtoms()]
    scaffold_mask = torch.ones(len(not_masked_atom_ids)).int()

    # Create a dictionairy of lists to store the atom ids of functional groups
    # together with their functional group smarts and mask
    functional_group_masks = defaultdict(list)

    for functional_group in functional_groups:
        for matched_atom_ids in molecule.GetSubstructMatches(
            MolFromSmarts(functional_group)
        ):
            mask = XAIChem.createMask(molecule, matched_atom_ids)

            # Remove atom id of masked atoms to create mask of scaffold
            for atom_id in matched_atom_ids:
                not_masked_atom_ids.remove(atom_id)

            # Use xor to invert the mask. This mask represents the atoms of a
            # functional group.
            scaffold_mask -= mask ^ 1

            functional_group_masks["molecule_smiles"].append(smiles)
            functional_group_masks["atom_ids"].append(matched_atom_ids)
            functional_group_masks["functional_group_smarts"].append(functional_group)
            functional_group_masks["mask"].append(mask ^ 1 if inverse else mask)

    # Add the scaffold mask
    functional_group_masks["molecule_smiles"].append(smiles)
    functional_group_masks["atom_ids"].append(tuple(not_masked_atom_ids))
    functional_group_masks["functional_group_smarts"].append("scaffold")
    functional_group_masks["mask"].append(
        scaffold_mask if inverse else scaffold_mask ^ 1
    )

    return pd.DataFrame.from_dict(functional_group_masks)
