from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import csv

# Read the file containing SMILES strings
file_path = # set CSV SMILES input file
batch_size = 100  # Set the desired batch size
output_file =  # Set the desired output CSV file name

def process_batch(batch):
    # Convert SMILES strings to molecules
    mols = [Chem.MolFromSmiles(smiles) for smiles in batch]

    # Create the atom-pair fingerprint generator
    fpgen = AllChem.GetAtomPairGenerator(countSimulation=False)

    # Calculate atom-pair fingerprints
    fps = [fpgen.GetFingerprint(mol) for mol in mols]

    # Prepare the output data
    output_data = []
    for fp in fps:
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


-----------
# Different batch processing

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import csv

# Read the file containing SMILES strings
file_path = # set CSV SMILES input file
output_file =  # Set the desired output CSV file name

def process_smiles(smiles):
    # Convert SMILES string to molecule
    mol = Chem.MolFromSmiles(smiles)
    
    # Skip invalid SMILES
    if mol is None:
        return None
    
    # Calculate atom-pair fingerprint
    fp = rdMolDescriptors.GetHashedAtomPairFingerprint(mol)
    
    # Determine the number of bits based on the maximum index
    num_bits = max(fp.GetNonzeroElements().keys()) + 1
    binary_values = [0] * num_bits
    
    for idx, val in fp.GetNonzeroElements().items():
        binary_values[idx] = val
    
    return binary_values

with open(file_path, 'r') as file, open(output_file, 'w', newline='') as csv_file:
    reader = csv.reader(file)
    writer = csv.writer(csv_file)
    
    for row in reader:
        smiles = row[0]  # Assuming the SMILES is in the first column
        binary_values = process_smiles(smiles)
        
        # Skip invalid SMILES
        if binary_values is None:
            continue
        
        writer.writerow(binary_values)
