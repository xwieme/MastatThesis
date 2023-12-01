import numpy as np
import torch
from rdkit import Chem


ATOMS = [
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "As",
    "Se",
    "Br",
    "Te",
    "I",
    "At",
    "other",
]

HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
    "other",
]

CHIRALITY_TYPES = [
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
]

BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

STEREO_TYPES = [
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOANY,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
]


def oneHotEncoding(x, values: list | None = None, length: int | None = None):
    """
    Creates a one-hot encoding of x, either by a given list of possible values
    that x can take or given the length of the resulting vector if x is an integer.

    :param x: value to encode, must be int if length is not None
    :param values: list of all possible value x could take (default is None)
    :param length: length of the resulting vector
    """

    if values is None:
        assert length is not None, "Either a list of values or a length must be given."
        assert isinstance(x, int), "x must be of type int if a length is given."

        out = torch.zeros(length)
        out[x] = 1

        return out

    else:
        # Use the last element of values if x is not present in the list
        if x not in values:
            x = values[-1]

        return torch.tensor([int(x == value) for value in values])


def getAtomFeatureVector(atom):
    return [
        *oneHotEncoding(atom.GetSymbol(), values=ATOMS),
        *oneHotEncoding(atom.GetDegree(), length=6),
        atom.GetFormalCharge(),
        *oneHotEncoding(atom.GetHybridization(), values=HYBRIDIZATIONS),
        int(atom.GetIsAromatic()),
        *oneHotEncoding(atom.GetTotalNumHs(), length=5),
        int(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
        *oneHotEncoding(atom.GetChiralTag(), values=CHIRALITY_TYPES),
    ]


def getBondFeatureVector(bond):
    return [
        *oneHotEncoding(bond.GetBondType(), values=BOND_TYPES),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        *oneHotEncoding(bond.GetStereo(), values=STEREO_TYPES),
    ]


def getNumAtomFeatures():
    return len(getAtomFeatureVector(Chem.MolFromSmiles("C").GetAtoms()[0]))


def getNumBondFeatures():
    return len(getBondFeatureVector(Chem.MolFromSmiles("CC").GetBonds()[0]))
