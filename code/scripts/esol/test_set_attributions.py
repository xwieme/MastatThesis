# Create a DataFrame containing the test set molecules with there corresponding
# attributions using the substructure method which produces the most substructures
import os

import numpy as np
import pandas as pd

DATA_DIR = "../../../data/ESOL"


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(DATA_DIR, "ESOL_test.csv"))[["smiles", "ESOL"]]
    attributions_fg = pd.read_json(
        os.path.join(DATA_DIR, "attribution_functional_groups.json")
    )
    attributions_brics = pd.read_json(os.path.join(DATA_DIR, "attribution_brics.json"))

    # Compute for each molecule the number of substructures obtained from functional
    # groups and BRICS and create column 'group' indicating which method create the
    # most substructures
    indicator_df = (
        attributions_fg.groupby("molecule_smiles")
        .size()
        .to_frame("N_fg")
        .join(attributions_brics.groupby("molecule_smiles").size().to_frame("N_brics"))
        .apply(lambda row: "fg" if row.N_fg > row.N_brics else "brics", axis=1)
        .to_frame("group")
    )

    # Get the attributions from the method that generates the most substructers.
    # Subsequently filter the molecules of the test set.
    test_attributions = (
        pd.concat(
            [
                indicator_df.query("group == 'fg'").join(
                    attributions_fg.set_index("molecule_smiles")
                ),
                indicator_df.query("group == 'brics'").join(
                    attributions_brics.set_index("molecule_smiles")
                ),
            ]
        )
        .join(data.set_index("smiles"), how="right")
        .reset_index()
    )

    # Display how many molecules cannot be split into substructures. These molecules
    # are removed since their rank correlation will be one no matter their prediction
    # since the rank of one number is always one.
    smiles_no_substructures = (
        test_attributions.groupby("molecule_smiles")
        .size()
        .to_frame("N")
        .query("N == 1")
        .index
    )

    test_attributions = test_attributions.query(
        "molecule_smiles not in @smiles_no_substructures"
    )

    # Compute absolute error
    data = (
        test_attributions[["molecule_smiles", "non_masked_prediction", "ESOL"]]
        .drop_duplicates()
        .rename(
            columns={"non_masked_prediction": "prediction", "molecule_smiles": "smiles"}
        )
    )

    data["absolute_error"] = np.abs(data["prediction"] - data["ESOL"])

    data.to_json(os.path.join(DATA_DIR, "test_absolute_error.json"))
    test_attributions.to_json(os.path.join(DATA_DIR, "test_attributions.json"))
