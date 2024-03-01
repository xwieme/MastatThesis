import pandas as pd

import XAIChem 

from mutag_rgcn_model import buildMutagModel


if __name__ == "__main__":

    # Get model architecture and configuration
    model, config = buildMutagModel(0)

    # Load trained models 
    paths = [
        f"../../data/Mutagenicity/trained_models/Mutagenicity_rgcn_model_{i}.pt"
        for i in range(10)
    ]
    models = XAIChem.loadModels(model, paths)

    # Load data for explanation
    molecules = pd.read_csv("../../data/Mutagenicity/Mutagenicity.csv")

    masks = XAIChem.substructures.functionalGroupMasks(molecules.smiles)
    attributions = XAIChem.attribution.difference(models, masks)

    print(attributions.head())
