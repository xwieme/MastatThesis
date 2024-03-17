import os

import pandas as pd
import plotly.express as px

DATA_DIR = "../../../data/aqsoldb_B"


if __name__ == "__main__":
    aqsoldb = pd.read_csv(os.path.join(DATA_DIR, "aqsoldb_B_raw.csv"))[
        ["Name", "SMILES", "Solubility"]
    ]

    print(f"Raw data set size: {len(aqsoldb)}")

    # Remove disconnected compounds and filter for carbon compounds
    aqsoldb["SMILES"] = aqsoldb.SMILES.str.split(".", expand=True)[0]
    aqsoldb = aqsoldb.query(
        "SMILES.str.contains('C|c') and not SMILES.str.contains(r'Co|Ca|^\[.*\]$')"
    )

    # Check the frequency of how many times a molecule occurs
    print("#####\nFrequency of occerence of the same molecule\n#####")
    print(aqsoldb.groupby(["Name", "SMILES"]).size().value_counts())
    print()

    # Use the mean value if the same compound occurs multiple times
    aqsoldb = aqsoldb.groupby(["Name", "SMILES"]).mean().reset_index()

    # Rename columns to be compatable with XAIChem python package
    aqsoldb = aqsoldb.rename(columns={"SMILES": "smiles", "Solubility": "aqsoldb_B"})

    print(f"Filtered data set size: {len(aqsoldb)}\n")
    aqsoldb.to_csv(os.path.join(DATA_DIR, "aqsoldb_B.csv"))

    # Shuffle and split data into train, validation and test set
    aqsoldb_shuffled = aqsoldb.sample(frac=1, random_state=42)

    train_size = int(0.75 * len(aqsoldb_shuffled))
    validation_size = int(0.15 * len(aqsoldb_shuffled))

    aqsoldb_train = aqsoldb_shuffled.iloc[:train_size]
    aqsoldb_validation = aqsoldb_shuffled.iloc[
        train_size : (train_size + validation_size)
    ]
    aqsoldb_test = aqsoldb_shuffled.iloc[(train_size + validation_size) :]

    aqsoldb_train.to_csv(os.path.join(DATA_DIR, "aqsoldb_B_train.csv"))
    aqsoldb_validation.to_csv(os.path.join(DATA_DIR, "aqsoldb_B_val.csv"))
    aqsoldb_test.to_csv(os.path.join(DATA_DIR, "aqsoldb_B_test.csv"))
