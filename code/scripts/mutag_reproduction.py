import argparse

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP
from sklearn import metrics
import pandas as pd

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
        "train": DataLoader(train_data, batch_size=256),
        "test": DataLoader(test_data, batch_size=256),
        "validation": DataLoader(val_data, batch_size=256)
    }

    # Use gpu if available, otherwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify model details
    config = {
        "epochs": 500,
        "random_seed": 2022 + model_id * 10,
        "learning_rate": 0.001,
        "patience": 30,
        "RGCN_num_layers": 3,
        "RGCN_hidden_units": [
            256,
            256,
            256,
        ],
        "RGCN_dropout_rate": 0.4,
        "RGCN_use_batch_norm": True,
        "RGCN_num_bases": None,
        "RGCN_loop": True,
        "MLP_hidden_units": [256, 256, 256, 1],
        "MLP_dropout_rate": 0.0,
    }

    # Define evaluation metrics 
    metrics_dict = {
        "roc_auc": metrics.roc_auc_score,
        "F1": metrics.f1_score,
        "accuracy": metrics.accuracy_score,
        "recall": metrics.recall_score
    }

    XAIChem.set_seed(config["random_seed"])
    
    print("Building model")
    gnn = XAIChem.models.RGCN(
        XAIChem.features.getNumAtomFeatures(),
        config["RGCN_num_layers"],
        config["RGCN_hidden_units"],
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

    # Transfer model to gpu
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    pos_weight = computePosWeight(train_data.y)
    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="mean", pos_weight=pos_weight.to(device)
    )

    early_stop = XAIChem.EarlyStopping(
        "../../data/Mutagenicity/trained_models", 
        f"Mutagenicity_rgcn_model_{model_id}", 
        config["patience"]
    )

    print("Start taining model")
    trainer = XAIChem.models.ModelTrainer(model, device)
    trainer.train(
        data_loaders,
        criterion,
        optimizer,
        config["epochs"],
        f"../../data/Mutagenicity/trained_models/Mutagenicity_rgcn_model_{model_id}",
        metrics_dict,
        early_stop,
        log=True,
    )

