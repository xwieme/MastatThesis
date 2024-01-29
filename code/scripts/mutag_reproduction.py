import argparse

import torch
from torch_geometric.loader import DataLoader
from sklearn import metrics
import pandas as pd
import numpy as np

import XAIChem


def computePosWeight(train_labels):

    num_pos = torch.sum(train_labels == 1)
    num_neg = torch.sum(train_labels == 0)

    weight = num_neg / (num_pos + 1e-8)

    return weight


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Mutagenicity_training",
        description="Train an graph neural network to predict mutagenicity class",
    )
    parser.add_argument("model_id")
    args = parser.parse_args()
    model_id = int(args.model_id)

    model, config = XAIChem.models.PreBuildModels.rgcnWuEtAll(
        "mutag_reproduction.yaml",
        ["seed"],
        model_id=model_id
    )

    print("Processing data") 
    data = pd.read_csv("../../data/Mutagenicity/Mutagenicity.csv")

    for tag in ["training", "test", "valid"]:
        df = data.query(f"group == '{tag}'").to_csv(
            f"../../data/Mutagenicity/Mutagenicity_{tag}.csv"
        )

    # Create pytorch geometric data objects
    train_data = XAIChem.Dataset(root="../../data", name="Mutagenicity", tag="training")
    test_data = XAIChem.Dataset(root="../../data", name="Mutagenicity", tag="test")
    val_data = XAIChem.Dataset(root="../../data", name="Mutagenicity", tag="valid")

    # Create pytorch graph batches
    data_loaders = {
        "train": DataLoader(train_data, batch_size=config["batch_size"], shuffle=True),
        "test": DataLoader(test_data, batch_size=config["batch_size"]),
        "validation": DataLoader(val_data, batch_size=config["batch_size"])
    }

    # Define evaluation metrics 
    metrics_dict = {
        "roc_auc": metrics.roc_auc_score,
        "F1": metrics.f1_score,
        "accuracy": metrics.accuracy_score,
        "recall": metrics.recall_score,
        "precision": metrics.precision_score,
        "mcc": metrics.matthews_corrcoef,
    }

    # Use gpu if available, otherwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device) # Transfer model to gpu

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    pos_weight = computePosWeight(train_data.y)
    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="mean", pos_weight=pos_weight.to(device)
    )

    early_stop = XAIChem.EarlyStopping(
        "../../data/Mutagenicity/trained_models", 
        f"Mutagenicity_rgcn_model_{model_id}", 
        config["early_stop"]["patience"],
        config["early_stop"]["mode"]
    )

    print("Start taining model")
    trainer = XAIChem.models.ModelTrainer(model, device, config)
    trainer.train(
        data_loaders,
        criterion,
        optimizer,
        config["epochs"],
        f"../../data/Mutagenicity/trained_models/Mutagenicity_rgcn_model_{model_id}",
        metrics_dict,
        early_stop,
        wandb_project="Mutagenicity_reproduction",
        wandb_group="RUN_1",
        wandb_name=f"model_{model_id}",
    )

