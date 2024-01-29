from typing import List
from collections import defaultdict

from tqdm import tqdm
import pandas as pd 
from rdkit.Chem import MolFromSmiles, MolFromSmarts 

from .. import variables
from .. import createMask


def functionalGroupMasks(
    smiles: str | List[str],
    functional_groups: List[str] = variables.FUNCTIONAL_GROUPS,
) -> pd.DataFrame:
    """
    Get all functional groups of the given molecule in smiles represenation and 
    create a mask for each functional group. The mask isolates a functional group 
    from the rest of the molecule.

    :param smiles: smiles represenation of the molecule
    :param functional_groups: list of functional groups to search for in the molecule
    """

    # Convert smiles to a one element list if a single string is given
    if isinstance(smiles, str):
        smiles = [smiles]

    dataframes = list()

    for molecule_smiles in tqdm(smiles): 
        molecule = MolFromSmiles(molecule_smiles)

        # Create a dictionairy of lists to store the atom ids of functional groups 
        # together with their functional group smarts and mask
        functional_group_masks = defaultdict(list)

        for functional_group in functional_groups:
            for matched_atom_ids in molecule.GetSubstructMatches(MolFromSmarts(functional_group)):
                
                functional_group_masks["molecule_smiles"].append(molecule_smiles)
                functional_group_masks["atom_ids"].append(matched_atom_ids)
                functional_group_masks["functional_group_smarts"].append(functional_group)
                functional_group_masks["mask"].append(createMask(molecule, matched_atom_ids))

        dataframes.append(pd.DataFrame.from_dict(functional_group_masks))

    return pd.concat(dataframes, ignore_index=True) 

