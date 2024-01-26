import os.path as osp
from typing import Callable, Optional

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

from . import features, utils


def createDataObjectFromRdMol(
    molecule: Chem.rdchem.Mol, y: float, num_classes: int | None = None
):
    x = torch.tensor(
        [features.getAtomFeatureVector(atom) for atom in molecule.GetAtoms()],
        dtype=torch.float,
    )

    edge_indices, edge_attrs, edge_types = [], [], []
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_features = features.getBondFeatureVector(bond)
        bond_type = utils.getEdgeType(torch.tensor(bond_features))

        # Treat molecules as undirected graphs
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [bond_features, bond_features]
        edge_types += [bond_type, bond_type]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(
        -1, features.getNumBondFeatures()
    )
    edge_type = torch.tensor(edge_types, dtype=torch.long).view(-1, 1)

    return Data(
        x=x,
        y=float(y),        
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        smiles=Chem.MolToSmiles(molecule),
    )


def createDataObjectFromSmiles(smiles: str, y: float , num_classes: int | None = None):
    molecule = Chem.MolFromSmiles(smiles)
    return createDataObjectFromRdMol(molecule, y, num_classes)


class Dataset(InMemoryDataset):
    """
    Creat a pytorch graph dataset from a csv file.

    :param root: directory containing data folders
    :param name: folder name of dataset
    :param tag: indicate type of data, is equal to train, test or val
    :param num_classes: if specified, encode target as a one hot vector
    """

    def __init__(
        self,
        root: str,
        name: str,
        tag: str,
        num_classes: int | None = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = name
        self.tag = tag
        self.nclasses = num_classes

        super(Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def raw_file_names(self) -> str:
        return f"{self.name}_{self.tag}.csv"

    @property
    def processed_file_names(self) -> str:
        return f"{self.name}_{self.tag}.pt"

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        data_list = (
            df[["smiles", self.name]]
            .apply(
                lambda row: createDataObjectFromSmiles(*row, num_classes=self.nclasses),
                axis=1,
            )
            .values
            .tolist()
        )

        data, slices = self.collate(data_list)
        torch.save(
            (data, slices), osp.join(self.processed_dir, self.processed_file_names)
        )
