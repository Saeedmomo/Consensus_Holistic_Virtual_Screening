from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
import csv

# Read the file containing SMILES strings
file_path = # set CSV SMILES input file
batch_size = 100  # Set the desired batch size
output_file =  # Set the desired output CSV file name

def process_batch(batch):
    # Convert SMILES strings to molecules
    mols = [Chem.MolFromSmiles(smiles) for smiles in batch]

    # Calculate topological torsion fingerprints
    fingerprints = [rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048) for mol in mols]

    # Prepare the output data
    output_data = []
    for fp in fingerprints:
        binary_values = [int(bit) for bit in fp.ToBitString()]
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


-------------
# Different batch processing

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import csv

def process_batch(batch):
    output_data = []
    for smiles in batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
            binary_values = list(fp.ToBitString())
            output_data.append(binary_values)
    return output_data

# Read the file containing SMILES strings
file_path = # set CSV SMILES input file
output_file =  # Set the desired output CSV file name

# Process the SMILES in batches
batch_size = 1000
batch = []
output_data = []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    
    for row in reader:
        smiles = row[0]  # Assuming the SMILES is in the first column
        batch.append(smiles)
        
        if len(batch) >= batch_size:
            output_data.extend(process_batch(batch))
            batch = []
    
    # Process the remaining SMILES in the last batch
    if batch:
        output_data.extend(process_batch(batch))

# Write the output data to the CSV file
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(output_data)
