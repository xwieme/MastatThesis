import os
import argparse
from pathlib import Path
import pandas as pd
import torch
import XAIChem


DATA_DIR = "../../../data"
OUT_DIR = os.path.join(DATA_DIR, "ESOL/attributions_brics")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ESOL_explanation_brics",
        description="Computation of different attribution measures to explain GNN model",
    )
    parser.add_argument("sample_id")
    args = parser.parse_args()
    sample_id = int(args.sample_id)

    if not Path(OUT_DIR).exists():
        Path(OUT_DIR).mkdir(parents=True)

    device = torch.device("cuda")
    # Get model architecture and configuration
    model, config = XAIChem.models.PreBuildModels.rgcnWuEtAll(
        "esol_reproduction.yaml", ["seed"], model_id=0
    )

    # Load trained models
    paths = [
        os.path.join(DATA_DIR, f"ESOL/trained_models/ESOL_rgcn_model_{i}_early_stop.pt")
        for i in range(10)
    ]
    models = XAIChem.loadModels(model, paths, device="cuda")

    # Load data for explanation and select a molecule
    molecules = pd.read_csv(os.path.join(DATA_DIR, "ESOL/ESOL.csv"))
    smiles = molecules.smiles.iloc[sample_id]

    # Explain the model prediction by breaking the molecule in parts
    # using BRICS bonds
    explanation_df = XAIChem.substructures.BRICSMasks(smiles)

    explanation_df = XAIChem.attribution.hamiacheNavarroValue(
        models, explanation_df, 0, device=device, shapley=True
    )
    explanation_df = XAIChem.attribution.substructureMaskExploration(
        models, explanation_df, device
    )

    print(explanation_df.drop(["molecule_smiles", "mask"], axis=1))

    explanation_df.drop("mask", axis=1).to_json(
        f"{OUT_DIR}/attribution_{sample_id}.json"
    )
