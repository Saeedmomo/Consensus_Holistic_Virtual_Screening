import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs

# Read the CSV file containing the compounds
csv_path = # Set CSV input file
df = pd.read_csv(csv_path, header=None)

# Select the compound for comparison (change the index as needed)
compare_index = 0  # Index of the compound to compare

# Extract the SMILES data for the compound to compare
compare_smiles = df.iloc[compare_index, 0]

# Calculate the Tanimoto similarity with the rest of the compounds
similarities = []
try:
    compare_mol = Chem.MolFromSmiles(compare_smiles)
    if compare_mol is None:
        raise ValueError("Invalid SMILES")
except Exception as e:
    print(f"Error: Invalid SMILES - {str(e)}")
    compare_mol = None

for index, row in df.iterrows():
    if index == compare_index:
        similarity = 1.0  # Set similarity to 1 for the compound itself
    else:
        try:
            mol = Chem.MolFromSmiles(row[0])
            if mol is None:
                raise ValueError("Invalid SMILES")
            similarity = DataStructs.FingerprintSimilarity(
                Chem.RDKFingerprint(compare_mol),
                Chem.RDKFingerprint(mol)
            )
        except Exception as e:
            print(f"Error: Invalid SMILES - {str(e)}")
            similarity = None
    similarities.append(similarity)

# Create a new DataFrame to store the results
result_df = pd.DataFrame({'Tanimoto Similarity': similarities})

# Save the results to a new CSV file
output_file = # Set the CSV output file
result_df.to_csv(output_file, index=False)

# Print a message indicating the successful completion
print(f"Tanimoto similarities calculated and saved to '{output_file}'.")
