from typing import Tuple
from torch_geometric.nn import MLP, FastRGCNConv

import XAIChem


def buildEsolModel(model_id: int) -> Tuple[XAIChem.models.MolecularPropertyPredictor, dict]:
    """
    Create a relational graph neural network to predict
    the expected solubility of molecules. The model 
    architecture and hyperparameters are taken from 
    Wu et. al. (https://doi.org/10.1038/s41467-023-38192-3).

    :param model_id: id of trained model, determines the random 
        seed.
    :return: a tuple of the RGCN model and its configuration details.
    """
    # Define model configuration
    config = {
        "seed": 2022 + model_id * 10,
        "batch_size": 32,
        "learning_rate": 0.003,
        "epochs": 500,
        "patience": 30,
        "RGCN_num_layers": 2,
        "RGCN_num_units": [256, 256],
        "RGCN_dropout_rate": 0.5,
        "RGCN_use_batch_norm": True,
        "RGCN_num_bases": 65, # Equal to the number of different edge types
        "RGCN_loop": False,
        "MLP_num_units": [256, 64, 64, 64, 1],
        "MLP_dropout_rate": 0.1
    }

    XAIChem.set_seed(config["seed"])

    # Construct a MolecularPropertiePredicor
    gnn = XAIChem.models.RGCN(
        XAIChem.features.getNumAtomFeatures(),
        config["RGCN_num_layers"],
        config["RGCN_num_units"],
        rgcn_conv=FastRGCNConv,
        dropout_rate=config["RGCN_dropout_rate"],
        use_batch_norm=config["RGCN_use_batch_norm"],
        num_bases=config["RGCN_num_bases"],
        loop=config["RGCN_loop"]
    )

    molecular_embedder = XAIChem.models.WeightedSum(config["RGCN_num_units"][-1])

    mlp = MLP(
        config["MLP_num_units"], 
        dropout=config["MLP_dropout_rate"]
    )

    # Combine all parts and return the final model
    return XAIChem.models.MolecularPropertyPredictor(gnn, molecular_embedder, mlp), config

