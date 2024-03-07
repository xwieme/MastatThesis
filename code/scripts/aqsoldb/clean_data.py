import pandas as pd 


if __name__ == "__main__":

    aqsoldb = pd.read_csv("aqsoldb_A/aqsoldb_A_raw.csv")[["Name", "SMILES", "Solubility"]]

    print(f"Raw data set size: {len(aqsoldb)}")

    # Remove disconnected compounds and filter for carbon compounds 
    aqsoldb["SMILES"] = aqsoldb.SMILES.str.split(".", expand=True)[0]
    aqsoldb = aqsoldb.query("SMILES.str.contains('C|c') and not SMILES.str.contains(r'Co|Ca|^\[.*\]$')")

    # Check the frequency of how many times a molecule occurs
    print("#####\nFrequency of occerence of the same molecule\n#####")
    print(aqsoldb.groupby(["Name", "SMILES"]).size().value_counts())
    print()

    # Use the mean value if the same compound occurs multiple times
    aqsoldb = aqsoldb.groupby(["Name", "SMILES"]).mean().reset_index()
    
    print(f"Filtered data set size: {len(aqsoldb)}\n")
    aqsoldb.to_csv("aqsoldb_A/aqsoldb_A.csv")
