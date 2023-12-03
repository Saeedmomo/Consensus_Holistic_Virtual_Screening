import csv
from rdkit import Chem
from rdkit.Chem import MACCSkeys

input_file = #CSV SMILES file path
output_file = # output CSV file path
batch_size = 100  # Number of molecules to process in each batch

# Function to process a batch of SMILES strings
def process_batch(batch):
    mols = []
    fingerprints = []

    # Convert SMILES strings to molecules and calculate fingerprints
    for smiles in batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
            fingerprints.append(MACCSkeys.GenMACCSKeys(mol))
        else:
            print(f"Invalid SMILES string: {smiles}")

    # Prepare the output data
    output_data = []
    for smiles, fp in zip(batch, fingerprints):
        if fp is not None:
            binary_values = [int(bit) for bit in fp.ToBitString()]
            output_data.append([smiles] + binary_values)
        else:
            print(f"Failed to calculate MACCS key for: {smiles}")

    # Write the output data to the CSV file
    with open(output_file, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(output_data)

# Open the input CSV file
with open(input_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    smiles_list = [row[0] for row in reader]

# Process the molecules in batches
batch = []
for smiles in smiles_list:
    batch.append(smiles)

    if len(batch) == batch_size:
        # Process the current batch
        process_batch(batch)

        # Reset the batch
        batch = []

# Process the remaining SMILES in the last batch
if batch:
    process_batch(batch)
