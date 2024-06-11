"""
This script computes the SME, Shapley value and HN value attributions 
of a graph neural network by subdividing molecules into functional groups 
or BRICS substructures.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import XAIChem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GNN_explanation",
        description="Computation of different attribution measures to explain GNN model",
    )
    parser.add_argument("DATA_DIR")
    parser.add_argument("sample_id")
    parser.add_argument("mean", type=float)

    args = parser.parse_args()
    # Assume that the data file has the same name as the directory and is in csv format
    DATA_FILENAME = f"{args.DATA_DIR.split('/')[-1]}.csv"
    SAMPLE_ID = int(args.sample_id)
    OUT_DIR_FUNC_GROUP = os.path.join(
        args.DATA_DIR, "attributions_functional_groups_attention"
    )
    OUT_DIR_BRICS = os.path.join(args.DATA_DIR, "attributions_brics_attention")

    if not Path(OUT_DIR_FUNC_GROUP).exists():
        Path(OUT_DIR_FUNC_GROUP).mkdir(parents=True)

    if not Path(OUT_DIR_BRICS).exists():
        Path(OUT_DIR_BRICS).mkdir(parents=True)

    # Get model architecture and configuration
    device = torch.device("cuda")
    model, config = XAIChem.models.PreBuildModels.rgcnWuEtAll(
        "./model_config.yaml", ["seed"], model_id=0
    )

    # Load trained models
    paths = [
        os.path.join(args.DATA_DIR, f"trained_models/rgcn_model_{i}_early_stop.pt")
        for i in range(10)
    ]
    models = XAIChem.loadModels(model, paths, device="cuda")

    # Load data for explanation and select the current molecule
    molecules = pd.read_csv(os.path.join(args.DATA_DIR, DATA_FILENAME))
    molecule_smiles = molecules["smiles"].iloc[SAMPLE_ID]

    try:
        # Explain using functional group substructures
        masks_functional_groups = XAIChem.substructures.functionalGroupMasks(
            molecule_smiles, inverse=True, multiplier=1
        )

        attributions_functional_groups = (
            XAIChem.attribution.substructureMaskExploration(
                models, masks_functional_groups, device
            )
        )

        if len(masks_functional_groups) > 12:
            raise Exception("Too many substructures")

        attributions_functional_groups = XAIChem.attribution.hamiacheNavarroValue(
            models,
            attributions_functional_groups,
            args.mean,
            shapley=True,
            device=device,
        )

        attributions_functional_groups.drop("mask", axis=1).to_json(
            os.path.join(OUT_DIR_FUNC_GROUP, f"attribution_{SAMPLE_ID}.json")
        )

        print(
            attributions_functional_groups[
                [
                    "molecule_smiles",
                    "functional_group",
                    "atom_ids",
                    "SME",
                    "HN_value",
                    "Shapley_value",
                ]
            ]
        )

        # Explain prediction by breaking the molecule in parts using BRICS bonds
        explanation_df = XAIChem.substructures.BRICSMasks(molecule_smiles, multiplier=1)

        explanation_df = XAIChem.attribution.substructureMaskExploration(
            models, explanation_df, device
        )

        if len(masks_functional_groups) > 12:
            raise Exception("Too many substructures")

        explanation_df = XAIChem.attribution.hamiacheNavarroValue(
            models, explanation_df, args.mean, device=device, shapley=True
        )

        explanation_df.drop("mask", axis=1).to_json(
            os.path.join(OUT_DIR_BRICS, f"attribution_{SAMPLE_ID}.json")
        )

        print(
            explanation_df[
                [
                    "molecule_smiles",
                    "substruct_smiles",
                    "SME",
                    "HN_value",
                    "Shapley_value",
                ]
            ]
        )

    # Write failures to log file
    except Exception as e:
        raise e
        print(e)
        print(molecule_smiles)
        print("#" * 25)
