"""
This scripts trains a graph neural network using the architecture proposed 
by wu et al. Training results are uploaded to wandb to inspect the history. 
"""

import argparse
import os

import torch
import XAIChem
from sklearn import metrics
from torch.optim import Adam
from torch_geometric.loader import DataLoader

if __name__ == "__main__":

    # Get model id from user input. The model id determines the random seed.
    parser = argparse.ArgumentParser(
        prog="train_model",
        description="Train a GNN to predict molecular properties",
    )
    parser.add_argument("DATA_DIR")
    parser.add_argument("wandb_project_name")
    parser.add_argument("model_id")
    args = parser.parse_args()
    MODEL_ID = int(args.model_id)

    model, config = XAIChem.models.PreBuildModels.rgcnWuEtAll(
        "./model_config.yaml", ["seed"], model_id=MODEL_ID
    )

    print("Loading data")
    train_data = XAIChem.Dataset(args.DATA_DIR, "train")
    test_data = XAIChem.Dataset(args.DATA_DIR, "test")
    val_data = XAIChem.Dataset(args.DATA_DIR, "val")

    # Batch data
    data = {
        "train": DataLoader(train_data, batch_size=config["batch_size"], shuffle=True),
        "test": DataLoader(test_data, batch_size=config["batch_size"]),
        "validation": DataLoader(val_data, batch_size=config["batch_size"]),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Transfer model to gpu

    # Setup training
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), config["learning_rate"])
    early_stopper = XAIChem.EarlyStopping(
        os.path.join(args.DATA_DIR, "trained_models"),
        f"rgcn_model_{MODEL_ID}",
        config["early_stop"]["patience"],
        config["early_stop"]["mode"],
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
        os.path.join(args.DATA_DIR, f"trained_models/rgcn_model_{MODEL_ID}.pt"),
        early_stop=early_stopper,
        metrics=metrics_dict,
        wandb_project=args.wandb_project_name,
        wandb_group="RUN_A",
        wandb_name=f"model_{MODEL_ID}",
        log=True,
    )
