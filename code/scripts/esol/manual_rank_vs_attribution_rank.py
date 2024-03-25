import re
from ast import literal_eval
from collections import defaultdict
from itertools import zip_longest

import numpy as np
import pandas as pd

if __name__ == "__main__":
    #############################
    # Manual rank preprocessing #
    #############################

    # Load data and convert it to a pandas data frame
    with open(
        "../../../data/ESOL/manual_rank_labeling.txt", "r", encoding="utf-8"
    ) as f:
        manual_ranks = f.read()

    manual_ranks = re.sub(r"\s+", " ", manual_ranks)

    data = defaultdict(list)

    for entry in re.split(r"\},", manual_ranks)[:-1]:
        # Complete dictionary string and convert to python dictionary
        molecule_ranks = literal_eval("{" + entry + "}}")
        smiles = list(molecule_ranks.keys())[0]

        # Convert dictionary into a generator of lists where each list
        # contains smiles, atom_ids and manual rank in that order
        substructures_ranks = zip_longest(
            [smiles],
            list(molecule_ranks.values())[0].keys(),
            list(molecule_ranks.values())[0].values(),
            fillvalue=smiles,
        )

        # substruct is a tuple that contains molecule smiles, atom_ids and
        # manual rank. Each part of the list is added to a dictionary to create
        # a pandas data frame.
        for substruct in substructures_ranks:
            data["molecule_smiles"].append(substruct[0])
            data["atom_ids"].append(substruct[1])
            data["manual_rank"].append(substruct[2])

    manual_ranks = pd.DataFrame.from_dict(data)
    print(f"Number of molecules: {len(manual_ranks.molecule_smiles.unique())}")

    #######################################
    # Test set attributions preprocessing #
    #######################################

    data = pd.read_csv("../../../data/ESOL/ESOL_test.csv")[["smiles", "ESOL"]]
    attributions_fg = pd.read_json(
        "../../../data/ESOL/attribution_functional_groups.json"
    )
    attributions_brics = pd.read_json("../../../data/ESOL/attribution_brics.json")

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

    # Compute Spearman Rank correlation between attributions and chemical
    # intuitive ranks
    test_attributions["atom_ids"] = test_attributions.atom_ids.apply(tuple)
    test_attributions = test_attributions[
        ["molecule_smiles", "atom_ids", "SME", "Shapley_value", "HN_value"]
    ].join(
        manual_ranks.set_index(["molecule_smiles", "atom_ids"]),
        how="left",
        on=["molecule_smiles", "atom_ids"],
    )

    attribution_rank_correlation = (
        test_attributions.groupby("molecule_smiles")
        .corr(method="spearman", numeric_only=True)
        .iloc[3::4, 0:-1]
        .reset_index()
        .drop(columns=["level_1"])
        .rename(
            columns={
                "SME": "SME_rank_corr",
                "Shapley_value": "Shapley_rank_corr",
                "HN_value": "HN_rank_corr",
            }
        )
    )

    data["absolute_error_class"] = data.absolute_error.apply(
        lambda value: "< 0.6" if value < 0.6 else ">= 0.6"
    )

    print(data.groupby("absolute_error_class").size())

    print(
        data.join(
            attribution_rank_correlation.set_index("molecule_smiles"),
            on="smiles",
        )
        .groupby("absolute_error_class")[
            ["SME_rank_corr", "Shapley_rank_corr", "HN_rank_corr"]
        ]
        .mean()
    )
