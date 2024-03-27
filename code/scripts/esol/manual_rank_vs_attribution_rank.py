import os
import re
from ast import literal_eval
from collections import defaultdict
from itertools import zip_longest

import numpy as np
import pandas as pd

DATA_DIR = "../../../data/ESOL"

if __name__ == "__main__":
    #############################
    # Manual rank preprocessing #
    #############################

    # Load data and convert it to a pandas data frame
    with open(
        os.path.join(DATA_DIR, "manual_rank_labeling.txt"), "r", encoding="utf-8"
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

    ######################
    # Read test set data #
    ######################
    data = pd.read_json(os.path.join(DATA_DIR, "test_absolute_error.json"))
    test_attributions = pd.read_json(os.path.join(DATA_DIR, "test_attributions.json"))

    #############################
    # Spearman rank correlation #
    #############################
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

    print(data.absolute_error_class.value_counts())

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
