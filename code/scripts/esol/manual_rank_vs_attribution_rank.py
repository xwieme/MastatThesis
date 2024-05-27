import argparse
import os
import re
from ast import literal_eval
from collections import defaultdict
from itertools import zip_longest

import numpy as np
import pandas as pd

DATA_DIR = "../../../data/ESOL"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="", dest="suffix")
    args = parser.parse_args()

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
    data = pd.read_json(
        os.path.join(DATA_DIR, f"test_absolute_error{args.suffix}.json")
    )
    test_attributions = pd.read_json(
        os.path.join(DATA_DIR, f"test_attributions{args.suffix}.json")
    )

    # print(data.info())
    # print("molecules", len(test_attributions.molecule_smiles.unique()))

    # print(
    #     test_attributions.query(
    #         "molecule_smiles == 'CCC(C)n1c(=O)[nH]c(C)c(Br)c1=O'"
    #     ).atom_ids
    # )
    # print(
    #     manual_ranks.query(
    #         "molecule_smiles == 'CCC(C)n1c(=O)[nH]c(C)c(Br)c1=O'"
    #     ).atom_ids
    # )

    #############################
    # Spearman rank correlation #
    #############################
    test_attributions["atom_ids"] = test_attributions.atom_ids.apply(tuple)

    # Consider attribution values up to six decimal digits, as more precission
    # is chemically irrelevant
    test_attributions[["SME", "Shapley_value", "HN_value"]] = test_attributions[
        ["SME", "Shapley_value", "HN_value"]
    ].round(6)

    test_attributions = test_attributions[
        ["molecule_smiles", "atom_ids", "SME", "Shapley_value", "HN_value"]
    ].join(
        manual_ranks.set_index(["molecule_smiles", "atom_ids"]),
        how="left",
        on=["molecule_smiles", "atom_ids"],
    )

    # Compute Spearman Rank correlation between attributions and chemical
    # intuitive ranks
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

    print(attribution_rank_correlation.info())
    print(
        attribution_rank_correlation[attribution_rank_correlation.SME_rank_corr.isna()]
    )

    data["absolute_error_class"] = data.absolute_error.apply(
        lambda value: "< 0.6" if value < 0.6 else ">= 0.6"
    )

    print(data.absolute_error_class.value_counts())

    rank_df = data.join(
        attribution_rank_correlation.set_index("molecule_smiles"),
        on="smiles",
    )

    print(
        rank_df.groupby("absolute_error_class")[
            ["SME_rank_corr", "Shapley_rank_corr", "HN_rank_corr"]
        ].mean()
    )

    rank_df.to_csv(
        os.path.join(
            DATA_DIR, f"manual_vs_attribution_rank_correlations{args.suffix}.csv"
        )
    )
