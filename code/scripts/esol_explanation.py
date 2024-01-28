import pandas as pd

import XAIChem 

from esol_rgcn_model import buildEsolModel


if __name__ == "__main__":

    # Get model architecture and configuration
    model, config = buildEsolModel(0)

    # Load trained models 
    paths = [
        f"../../data/ESOL/trained_models/ESOL_rgcn_model_{i}.pt"
        for i in range(10)
    ]
    models = XAIChem.loadModels(model, paths)

    # Load data for explanation
    molecules = pd.read_csv("../../data/ESOL/ESOL.csv")

    masks = XAIChem.substructures.functionalGroupMasks(molecules.smiles)
    attributions = XAIChem.attribution.difference(models, masks)

    print(attributions.head())
