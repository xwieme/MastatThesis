from typing import Set, List, Iterable
from collections import deque

from rdkit import Chem


def getSubstructureBondIds(
    molecule: Chem.rdchem.Mol, 
    atom_ids: List[int]
) -> Set[int]:
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
            if (begin_atom_id != atom_id and begin_atom_id in atom_ids) or\
                (end_atom_id != atom_id and end_atom_id in atom_ids):
                substructure_bonds.add(bond.GetIdx())

    return substructure_bonds


def breakBRICKSBond(
    molecule: Chem.rdchem.Mol, 
    bond: Iterable[int]
) -> List[List[int]]:
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






























