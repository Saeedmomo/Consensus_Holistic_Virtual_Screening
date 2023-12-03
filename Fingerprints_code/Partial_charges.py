import csv
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

input_file = # Set CSV input SMILES file
output_file = # Set CSV output file

# Open the input CSV file
with open(input_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    smiles_list = [row[0] for row in reader]

# Initialize the output data
output_data = [['SMILES', 'PartialCharges']]

# Process each SMILES string
for smiles in smiles_list:
    # Create a molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)

    # Assign Gasteiger charges to the molecule
    rdPartialCharges.ComputeGasteigerCharges(mol)

    # Access the partial charges of atoms
    charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]

    # Append the SMILES and charges to the output data
    output_data.append([smiles, charges])

# Write the output data to the CSV file
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(output_data)


-------------------
# Different Batches processing


import csv
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

input_file = # Set CSV input SMILES file
output_file = # Set CSV output file

# Open the input CSV file
with open(input_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    smiles_list = [row[0] for row in reader]

# Initialize the output data
output_data = [['SMILES', 'PartialCharges']]

# Process each SMILES string
for smiles in smiles_list:
    # Create a molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        # Assign Gasteiger charges to the molecule
        rdPartialCharges.ComputeGasteigerCharges(mol)

        # Access the partial charges of atoms
        charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]

        # Append the SMILES and charges to the output data
        output_data.append([smiles, charges])

# Write the output data to the CSV file
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(output_data)
