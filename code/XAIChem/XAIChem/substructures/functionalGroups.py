from collections import defaultdict
from typing import List

import pandas as pd
from rdkit.Chem import MolFromSmarts, MolFromSmiles
from tqdm import tqdm

from .. import createMask, variables


def functionalGroupMasks(
    smiles: str | List[str],
    functional_groups: List[str] | None = None,
    inverse: bool = False,
) -> pd.DataFrame:
    """
    Get all functional groups of the given molecule in smiles represenation and
    create a mask for each functional group. The mask isolates a functional group
    from the rest of the molecule.

    :param smiles: smiles represenation of the molecule
    :param functional_groups: optional list of functional groups to search for in the molecule.
        (default is None, i.e. a default set will be used)
    :param inverse: Set to true to select the corresponding functional group instead
        of masking it. (default is False)
    """

    # Convert smiles to a one element list if a single string is given
    if isinstance(smiles, str):
        smiles = [smiles]

    if functional_groups is None:
        functional_groups = variables.FUNCTIONAL_GROUPS

    dataframes = []

    for molecule_smiles in tqdm(smiles):
        molecule = MolFromSmiles(molecule_smiles)

        # Create a dictionairy of lists to store the atom ids of functional groups
        # together with their functional group smarts and mask
        functional_group_masks = defaultdict(list)

        for functional_group in functional_groups:
            for matched_atom_ids in molecule.GetSubstructMatches(
                MolFromSmarts(functional_group)
            ):

                functional_group_masks["molecule_smiles"].append(molecule_smiles)
                functional_group_masks["atom_ids"].append(matched_atom_ids)
                functional_group_masks["functional_group_smarts"].append(
                    functional_group
                )

                if inverse:
                    functional_group_masks["mask"].append(
                        createMask(molecule, matched_atom_ids) ^ 1
                    )
                else:
                    functional_group_masks["mask"].append(
                        createMask(molecule, matched_atom_ids)
                    )

        dataframes.append(pd.DataFrame.from_dict(functional_group_masks))

    return pd.concat(dataframes, ignore_index=True)
