from torch.nn import functional as F
from torch_geometric.nn import MLP, FastRGCNConv

import XAIChem


def buildMutagModel(model_id):

    # Specify model details
    config = {
        "epochs": 500,
        "random_seed": 2022 + model_id * 10,
        "learning_rate": 0.001,
        "patience": 30,
        "RGCN_num_layers": 3,
        "RGCN_hidden_units": [256, 256, 256],
        "RGCN_dropout_rate": 0.4,
        "RGCN_use_batch_norm": True, 
        "RGCN_num_bases": 65, # Use the number of different edge types
        "RGCN_loop": False,
        "MLP_hidden_units": [256, 128, 128, 128, 1],
        "MLP_dropout_rate": 0.0,
    }

    XAIChem.set_seed(config["random_seed"])
    
    print("Building model")
    gnn = XAIChem.models.RGCN(
        XAIChem.features.getNumAtomFeatures(),
        config["RGCN_num_layers"],
        config["RGCN_hidden_units"],
        rgcn_conv=FastRGCNConv,
        dropout_rate=config["RGCN_dropout_rate"],
        use_batch_norm=config["RGCN_use_batch_norm"],
        num_bases=config["RGCN_num_bases"],
        loop=config["RGCN_loop"],
    )

    molecular_embedder = XAIChem.models.WeightedSum(config["RGCN_hidden_units"][-1])

    mlp = MLP(
        config["MLP_hidden_units"],
        dropout=config["MLP_dropout_rate"]
    )

    model = XAIChem.models.MolecularPropertyPredictor(
        gnn,
        molecular_embedder,
        mlp,
        F.sigmoid
    )

    return model, config

