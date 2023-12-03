from rdkit import Chem
from rdkit.Chem import AllChem
import csv

# Read the file containing SMILES strings
file_path = # SMILES CSV file path
batch_size = 100  # Set the desired batch size
output_file =  # Set the desired output CSV file name

def process_batch(batch):
    # Convert SMILES strings to molecules and calculate fingerprints
    mols = []
    fingerprints = []
    for smiles in batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
            fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
        else:
            print(f"Invalid SMILES string: {smiles}")
    
    # Prepare the output data
    output_data = []
    for fingerprint in fingerprints:
        # Convert the fingerprint to a list of binary values
        binary_values = [int(bit) for bit in fingerprint.ToBitString()]
        output_data.append(binary_values)
    
    # Write the output data to the CSV file
    with open(output_file, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(output_data)

with open(file_path, 'r') as file:
    batch = []
    for line in file:
        smiles = line.strip()
        batch.append(smiles)
        
        if len(batch) == batch_size:
            # Process the current batch
            process_batch(batch)
            
            # Reset the batch
            batch = []
    
    # Process the remaining SMILES in the last batch
    if batch:
        process_batch(batch)
