from typing import List, Set

from rdkit import Chem


def getSubstructureBondIds(molecule: Chem.rdchem.Mol, atom_ids: List[int]) -> Set[int]:
    """
    Given a list of atom ids which are are part of a substructure, return a set
    of the bond ids between those atoms.

    :param molecule: rdkit molecule that containers the substructure
    :param atom_ids: rdkit ids of atoms in the substructure
    """

    substructure_bonds = set()

    for atom_id in atom_ids:
        bonds = molecule.GetAtomWithIdx(atom_id).GetBonds()

        for bond in bonds:
            begin_atom_id = bond.GetBeginAtomIdx()
            end_atom_id = bond.GetEndAtomIdx()

            # Either the current bond is part of the substructure or it connects
            # the substructure to the rest of the molecule. Only the bonds that
            # part of the substructure are of interest.
            if (begin_atom_id != atom_id and begin_atom_id in atom_ids) or (
                end_atom_id != atom_id and end_atom_id in atom_ids
            ):
                substructure_bonds.add(bond.GetIdx())

    return substructure_bonds
