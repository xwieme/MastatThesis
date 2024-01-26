from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP
import torch
from torch.optim import Adam 

import XAIChem


if __name__ == "__main__":

    # Define model configuration
    config = {
        "seed": 2022 + 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 500,
        "patience": 30,
        "RGCN_num_layers": 2,
        "RGCN_num_units": [XAIChem.features.getNumAtomFeatures(), 256, 256],
        "RGCN_dropout_rate": 0.5,
        "RGCN_use_batch_norm": True,
        "RGCN_num_bases": None,
        "RGCN_loop": False,
        "MLP_num_layers": 3,
        "MLP_num_units": [256, 256, 256, 256, 1],
        "MLP_dropout_rate": 0.5
    }

    # Load data
    print("Loading data ...")
    train_data = XAIChem.Dataset("../../data", "ESOL", "train")
    test_data = XAIChem.Dataset("../../data", "ESOL", "test")
    val_data = XAIChem.Dataset("../../data", "ESOL", "val")

    # Batch data 
    train_loader = DataLoader(train_data, batch_size = config["batch_size"])
    test_loader = DataLoader(test_data, batch_size = config["batch_size"])
    val_loader = DataLoader(val_data, batch_size = config["batch_size"])

    data = {
        "train": train_loader,
        "test": test_loader,
        "validation": val_loader
    }

    print("Model setup ...")
    # Construct a MolecularPropertiePredicor
    gnn = XAIChem.models.RGCN(
        XAIChem.features.getNumAtomFeatures(),
        config["RGCN_num_layers"],
        config["RGCN_num_units"],
        dropout_rate = config["RGCN_dropout_rate"],
        use_batch_norm = config["RGCN_use_batch_norm"],
        num_bases = config["RGCN_num_bases"],
        loop = config["RGCN_loop"]
    )

    molecular_embedder = XAIChem.models.WeightedSum(config["RGCN_num_units"][-1])

    mlp = MLP(
        config["MLP_num_units"], 
        dropout = config["MLP_dropout_rate"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XAIChem.models.MolecularPropertyPredictor(gnn, molecular_embedder, mlp)
    # Transfer model to gpu
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), config["learning_rate"])
    early_stopper = XAIChem.EarlyStopping(
        "../../data/ESOL/trained_models",
        "ESOL_reproduction",
        config["patience"]
    )

    print("start training")
    trainer = XAIChem.models.ModelTrainer(model, device)
    trainer.train(
        data,
        criterion,
        optimizer,
        config["epochs"],
        "../../data/ESOL/model_1.pt",
        early_stop = early_stopper,
        wandb_project = "ESOL_reproduction",
        wandb_group = "RUN_1",
        wandb_name = "model_1",
        log = True
    )

