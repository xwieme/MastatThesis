import os
import argparse

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import wandb

import XAIChem


def train(dataloader, model, criterion, optimizer, device):
    model.train()

    for data in dataloader:
        data.to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_type, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()


def evaluate(dataloader, model, criterion, device):
    model.eval()

    # Disable gradients computation to save GPU memory
    with torch.no_grad():
        correct = 0
        losses = 0
        for data in dataloader:
            data.to(device)

            out = model(data.x, data.edge_index, data.edge_type, data.batch)
            losses += criterion(out, data.y)
            pred = out.argmax(dim=1)
            correct = int((pred == data.y).sum())

    return correct / len(dataloader.dataset), losses / len(dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Mutagenicity_training",
        description="Train an graph neural network to predict mutagenicity class",
    )
    parser.add_argument("--data_dir")
    args = parser.parse_args()

    data = pd.read_csv(os.path.join(args.data_dir, "Mutagenicity/Mutagenicity.csv"))

    data.query("group == 'training'").to_csv(
        os.path.join(args.data_dir, "Mutagenicity/Mutagenicity_train.csv")
    )
    data.query("group == 'test'").to_csv(
        os.path.join(args.data_dir, "Mutagenicity/Mutagenicity_test.csv")
    )
    data.query("group == 'valid'").to_csv(
        os.path.join(args.data_dir, "Mutagenicity/Mutagenicity_val.csv")
    )

    # Create pytorch geometric data objects
    train_data = XAIChem.Dataset(root=args.data_dir, name="Mutagenicity", tag="train")
    test_data = XAIChem.Dataset(root=args.data_dir, name="Mutagenicity", tag="test")
    val_data = XAIChem.Dataset(root=args.data_dir, name="Mutagenicity", tag="val")

    # Create pytorch graph batches
    train_loader = DataLoader(train_data, batch_size=256)
    test_loader = DataLoader(test_data, batch_size=256)
    val_loader = DataLoader(val_data, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_id in range(10):
        config = {
            "architecture": "RGCN by Wu et al.",
            "dataset": "Mutagenicity from Wu et al.",
            "epchos": 500,
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
            num_mlp_output_units=2,
            rgcn_dropout_rate=config["rgcn_dropout_rate"],
            mlp_dropout_rate=config["mlp_dropout_rate"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, 500):
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
            os.path.join(args.data_dir, f"trained_models/Mutagenicity/Mutagenicity_rgcn_model_{model_id}.pt"),
        )
        wandb.finish()
