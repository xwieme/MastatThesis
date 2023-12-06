import os
import argparse
import pathlib

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn import metrics
import wandb

import XAIChem


def train(dataloader, model, criterion, optimizer, device):
    model.train()

    for data in dataloader:
        data.to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_type, data.batch)
        loss = criterion(out.view(-1), data.y)
        loss.backward()
        optimizer.step()


def evaluate(dataloader, model, criterion, device):
    model.eval()

    # Disable gradients computation to save GPU memory
    with torch.no_grad():
        predictions = list()
        labels = list()
        losses = 0
        for data in dataloader:
            data.to(device)

            out = model(data.x, data.edge_index, data.edge_type, data.batch)
            losses += criterion(out.view(-1), data.y)
            predictions.extend(out.to("cpu").numpy())
            labels.extend(data.y.to("cpu").numpy())

    return metrics.roc_auc_score(labels, predictions), losses / len(dataloader)


def pos_weight(train_labels):
    task_pos_weight_list = []
    num_pos = torch.sum(train_labels == 1)
    num_neg = torch.sum(train_labels == 0)

    weight = num_neg / (num_pos + 0.00000001)

    return weight.view(
        1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Mutagenicity_training",
        description="Train an graph neural network to predict mutagenicity class",
    )
    parser.add_argument("--data_dir")
    args = parser.parse_args()

    data = pd.read_csv(os.path.join(args.data_dir, "Mutagenicity/Mutagenicity.csv"))

    for tag in ["training", "test", "valid"]:
        df = data.query(f"group == '{tag}'").to_csv(
            os.path.join(args.data_dir, f"Mutagenicity/Mutagenicity_{tag}.csv")
        )

    # Create pytorch geometric data objects
    train_data = XAIChem.Dataset(
        root=args.data_dir, name="Mutagenicity", tag="training"
    )
    test_data = XAIChem.Dataset(root=args.data_dir, name="Mutagenicity", tag="test")
    val_data = XAIChem.Dataset(root=args.data_dir, name="Mutagenicity", tag="valid")

    # Create pytorch graph batches
    train_loader = DataLoader(train_data, batch_size=256)
    test_loader = DataLoader(test_data, batch_size=256)
    val_loader = DataLoader(val_data, batch_size=256)

    # Create directory where model is saved
    save_model_dir = os.path.join(args.data_dir, "trained_models/Mutagenicity")
    pathlib.Path(save_model_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_id in range(10):
        config = {
            "architecture": "RGCN by Wu et al.",
            "dataset": "Mutagenicity from Wu et al.",
            "epochs": 500,
            "random_seed": 2022 + model_id * 10,
            "learning_rate": 0.001,
            "rgcn_dropout_rate": 0.4,
            "num_rgcn_layers": 3,
            "mlp_dropout_rate": 0.0,
            "num_mlp_hidden_units": 128,
        }

        wandb.init(
            project="Mutagenicity_reproduction", name=f"model_{model_id}", config=config
        )
        XAIChem.set_seed(config["random_seed"])

        model = XAIChem.RGCN(
            num_node_features=XAIChem.getNumAtomFeatures(),
            num_rgcn_layers=config["num_rgcn_layers"],
            num_mlp_hidden_units=config["num_mlp_hidden_units"],
            rgcn_dropout_rate=config["rgcn_dropout_rate"],
            mlp_dropout_rate=config["mlp_dropout_rate"],
            use_fastrgcn=True,
            bclassification=True,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        pos_weight = pos_weight(train_data.y)
        criterion = torch.nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=pos_weight.to(device)
        )

        early_stop = XAIChem.EarlyStopping(
            save_model_dir, f"Mutagenicity_rgcn_model_{model_id}", patience=30
        )

        epoch = 0
        val_loss = np.inf
        while epoch <= config["epochs"] and not early_stop(val_loss, model):
            epoch += 1

            train(train_loader, model, criterion, optimizer, device)

            train_acc, train_loss = evaluate(train_loader, model, criterion, device)
            test_acc, test_loss = evaluate(test_loader, model, criterion, device)
            val_acc, val_loss = evaluate(val_loader, model, criterion, device)

            wandb.log(
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }
            )

        torch.save(
            model.state_dict(),
            os.path.join(save_model_dir, f"Mutagenicity_rgcn_model_{model_id}.pt"),
        )
        wandb.finish()
