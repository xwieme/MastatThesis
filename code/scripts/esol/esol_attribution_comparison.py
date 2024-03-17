import os

import pandas as pd

DATA_DIR = "../../../data"


if __name__ == "__main__":
    # Load reference dataset and attribution data
    data = pd.read_csv(os.path.join(DATA_DIR, "ESOL/ESOL.csv"))
    attributions_functional_groups = pd.DataFrame(
        pd.read_json(os.path.join(DATA_DIR, "ESOL/attribution.json"))
    )
    attributions_brics = pd.DataFrame(
        pd.read_json(os.path.join(DATA_DIR, "/ESOL/attribution_brics.json"))
    )
    aqsoldb = pd.read_csv("../../data/ESOL/aqsoldb.csv")[
        ["Name", "SMILES", "Solubility"]
    ]

    # Get the molecules where all its BRICS substructures have an experimental
    # solubility present in the AqSolDB.
    comparable_attributions_df = attributions_brics.join(
        aqsoldb.set_index("SMILES"), on="substruct_smiles", how="inner"
    )

    print("#Molecules: ", end="")
    print(
        comparable_attributions_df.groupby("molecule_smiles")
        .size()
        .to_frame("N_check")
        .join(
            attributions_brics.groupby("molecule_smiles")
            .size()
            .to_frame("N_attribution")
        )
        .apply(lambda row: 1 if row.iloc[0] == row.iloc[1] else 0, axis=1)
        .sum()
    )
