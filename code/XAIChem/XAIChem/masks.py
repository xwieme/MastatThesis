from typing import Iterable

import torch
from rdkit import Chem


def createMask(molecule: Chem.rdchem.Mol, substructure: Iterable[int]) -> torch.Tensor:
    """
    Create a binary vector with length equal to the number of atoms in the given molecule.
    A zero indicates the corresponding atom is masked.

    :param molecule: rdkit molecule where a masked is applied on
    :param substructure: ids of the atoms that are masked
    """

    mask = torch.zeros(molecule.GetNumAtoms(), dtype=int)
    for i, atom in enumerate(molecule.GetAtoms()):
        mask[i] = atom.GetIdx() not in substructure

    # mask must be a 2D tensor to mach dimensions in graph neural network model
    return mask.view(-1, 1)


def removeAtoms(molecule: Chem.rdchem.Mol, atom_ids: Iterable[int]) -> Chem.rdchem.Mol:
    try:
        edited_molecule = Chem.RWMol(molecule)
        edited_molecule.BeginBatchEdit()

        for atom_id in atom_ids:
            edited_molecule.RemoveAtom(atom_id)

        edited_molecule.CommitBatchEdit()
        Chem.SanitizeMol(edited_molecule)

        return edited_molecule

    except:
        # Changes in aromatic carbon or nitrogen atoms need to be manually specified

        edited_molecule = Chem.RWMol(molecule)
        edited_molecule.BeginBatchEdit()

        edited_molecule.RemoveAtom(atom_ids[0])
        atom = edited_molecule.GetAtomWithIdx(atom_ids[0] - 1)
        edited_molecule.GetAtomWithIdx(atom_ids[0] - 1).SetNumExplicitHs(
            1 if atom.GetSymbol() == "N" else 2
        )

        edited_molecule.CommitBatchEdit()
        Chem.SanitizeMol(edited_molecule)

        return edited_molecule
