from typing import Optional, Callable
import os.path as osp

from rdkit import Chem
import pandas as pd

import torch
from torch_geometric.data import Data, InMemoryDataset

from. import features, utils


def createDataObjectFromSmiles(smiles: str, y: any):

    rd_mol = Chem.MolFromSmiles(smiles)

    x = torch.tensor([
            features.getAtomFeatureVector(atom)
            for atom in rd_mol.GetAtoms()
        ], dtype=torch.float)

    edge_indices, edge_attrs, edge_types = [], [], []
    for bond in rd_mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_features = features.getBondFeatureVector(bond)
        bond_type = utils.getEdgeType(torch.tensor(bond_features))

        # Treat molecules as undirected graphs
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [bond_features, bond_features]
        edge_types += [bond_type, bond_type]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 10)
    edge_type = torch.tensor(edge_types, dtype=torch.long).view(-1, 1)

    return Data(
        x=x, 
        y=y, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        edge_type=edge_type, 
        smiles=smiles
    )


class Dataset(InMemoryDataset):
    """
    Creat a pytorch graph dataset from a csv file.

    :param root: directory containing data folders
    :param name: folder name of dataset
    :param tag: indicate type of data, is equal to train,
        test or val
    """

    def __init__(
        self,
        root: str,
        name: str,
        tag: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = name
        self.tag = tag 

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
        data_list = df[["smiles", self.name]].apply(
            lambda row: createDataObjectFromSmiles(*row),
            axis=1
        ).tolist()

        data, slices = self.collate(data_list)
        torch.save((data, slices), osp.join(self.processed_dir, self.processed_file_names))































