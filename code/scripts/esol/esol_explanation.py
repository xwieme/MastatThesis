import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import tqdm
import XAIChem

DATA_DIR = "../../../data"
OUT_DIR = os.path.join(DATA_DIR, "ESOL/attributions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ESOL_explanation",
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

    # Load data for explanation
    molecules = pd.read_csv(os.path.join("ESOL/ESOL.csv"))

    # # Compute mean prediction: -3.1271107
    # molecules_data = [
    #     XAIChem.createDataObjectFromSmiles(smiles, np.inf)
    #     for smiles in molecules.smiles
    #     if len(smiles) > 1
    # ]
    # predictions = [
    #     XAIChem.predict(data, models=models) for data in tqdm(molecules_data)
    # ]
    # print(torch.mean(predictions))

    molecule_smiles = molecules["smiles"].iloc[sample_id]

    try:
        masks = XAIChem.substructures.functionalGroupMasks(molecule_smiles, inverse=True)

        attributions, prediction = XAIChem.attribution.difference(
            models, masks, device, return_prediction=True
        )
        attributions = XAIChem.attribution.hamiacheNavarroValue(
            models, molecule_smiles, attributions, -3.1271107, shapley=True, device=device
        )

        attributions.drop("mask", axis=1).to_json(
            f"{OUT_DIR}/attribution_{sample_id}.json"
        )

    except Exception as e:
        print(e)
        print(molecule_smiles)
        print("#"*25)
