import argparse

import torch
from torch.optim import Adam 
from torch_geometric.loader import DataLoader
from sklearn import metrics

import XAIChem

# from esol_rgcn_model import buildEsolModel  


if __name__ == "__main__":
    
    # Get model id from user input. The model id determines the random seed.
    parser = argparse.ArgumentParser(
        prog="ESOL_training",
        description="Train a GNN to predict expected solubility of molecules"
    )
    parser.add_argument("model_id")
    args = parser.parse_args()
    model_id = int(args.model_id)

    model, config = XAIChem.models.PreBuildModels.rgcnWuEtAll(
        "esol_reproduction.yaml", 
        ["seed"], 
        model_id=model_id
    )

    print("Loading data")
    train_data = XAIChem.Dataset("../../data", "ESOL", "train")
    test_data = XAIChem.Dataset("../../data", "ESOL", "test")
    val_data = XAIChem.Dataset("../../data", "ESOL", "val")

    # Batch data 
    data = {
        "train": DataLoader(train_data, batch_size=config["batch_size"], shuffle=True),
        "test": DataLoader(test_data, batch_size=config["batch_size"]),
        "validation": DataLoader(val_data, batch_size=config["batch_size"])
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Transfer model to gpu

    # Setup training
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), config["learning_rate"])
    early_stopper = XAIChem.EarlyStopping(
        "../../data/ESOL/trained_models",
        f"ESOL_rgcn_model_{model_id}",
        config["early_stop"]["patience"],
        config["early_stop"]["mode"]
    )

    # Specify evaluation metrics
    # Loss is logged automatically
    metrics_dict = {"r2": metrics.r2_score}

    print("start training")
    trainer = XAIChem.models.ModelTrainer(model, device, config)
    trainer.train(
        data,
        criterion,
        optimizer,
        config["epochs"],
        f"../../data/ESOL/model_{model_id}.pt",
        early_stop=early_stopper,
        metrics=metrics_dict,
        wandb_project="ESOL_reproduction",
        wandb_group="RUN_3",
        wandb_name=f"model_{model_id}",
        log=True
    )

