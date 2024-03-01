import os
from collections import defaultdict
from typing import Dict

import pandas as pd
import torch
from rdkit.Chem import FragmentCatalog, MolFromSmarts, MolFromSmiles, RDConfig

from ..masks import createMask
from ..variables import FUNCTIONAL_GROUPS_DICT


def functionalGroupMasks(
    smiles: str,
    functional_groups_dict: Dict[str, str] | None = None,
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

    if functional_groups_dict is None:
        # functional_groups_dict = FUNCTIONAL_GROUPS_DICT
        # Get path to list of functional groups from RDKit and extract their molecule objects
        fname = os.path.join(RDConfig.RDDataDir, "FunctionalGroups.txt")
        fparams = FragmentCatalog.FragCatParams(1, 6, fname)
        functional_groups_dict = {
            fparams.GetFuncGroup(i).GetProp("_Name"): fparams.GetFuncGroup(i)
            for i in range(fparams.GetNumFuncGroups())
        }

    molecule = MolFromSmiles(smiles)

    # Keep track of atoms that are note included in a mask to generate
    # a mask for the scaffold, i.e. main structure of the molecule
    not_masked_atom_ids = [atom.GetIdx() for atom in molecule.GetAtoms()]
    scaffold_mask = torch.ones(len(not_masked_atom_ids)).int()

    # Create a dictionary of lists to store the atom ids of functional groups
    # together with their functional group and mask
    masks = defaultdict(list)

    for functional_group, smarts in functional_groups_dict.items():
        for matched_atom_ids in molecule.GetSubstructMatches(smarts):
            # Remove connection Carbon atom
            matched_atom_ids = matched_atom_ids[1:]

            # Check if matched atoms were already matched by another functional group.
            # If this is the case, skip the match to avoid overlap
            if not set(matched_atom_ids).issubset(not_masked_atom_ids):
                continue

            # Remove atom id of masked atoms to create mask of scaffold
            for atom_id in matched_atom_ids:
                not_masked_atom_ids.remove(atom_id)

            mask = createMask(molecule, matched_atom_ids)

            # Use xor to invert the mask. This mask selects the atoms of a
            # functional group.
            scaffold_mask -= mask ^ 1

            masks["molecule_smiles"].append(smiles)
            masks["atom_ids"].append(matched_atom_ids)
            masks["functional_group"].append(functional_group)
            masks["mask"].append(mask ^ 1 if inverse else mask)

    # Add the scaffold mask if necessairy
    if len(not_masked_atom_ids) > 0:
        masks["molecule_smiles"].append(smiles)
        masks["atom_ids"].append(tuple(not_masked_atom_ids))
        masks["functional_group"].append("scaffold")
        masks["mask"].append(scaffold_mask if inverse else scaffold_mask ^ 1)

    return pd.DataFrame.from_dict(masks)
