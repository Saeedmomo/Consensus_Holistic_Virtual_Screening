import csv
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem

input_file = # set CSV SMILES input file
output_file = # Set CSV output file
smiles_column_index = 0  # Modify this to match the index of the SMILES column in your input file (zero-based)

# Collect the SMILES strings from the input file
smiles_list = []
with open(input_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        if len(row) > smiles_column_index:
            smiles = row[smiles_column_index]
            smiles_list.append(smiles)

# Calculate descriptors for each molecule
descriptors_2d = []
descriptors_3d = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # Add hydrogens to the molecule
    AllChem.EmbedMolecule(mol)  # Generate conformers
    AllChem.MMFFOptimizeMolecule(mol)  # Optimize conformers
    descriptor_values_2d = [Descriptors.descList[i][1](mol) for i in range(len(Descriptors.descList))]
    descriptor_values_3d = [rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcTPSA(mol)]
    descriptors_2d.append(descriptor_values_2d)
    descriptors_3d.append(descriptor_values_3d)

# Write the descriptor values to the output CSV file
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the descriptor names as the header row
    header_row = ['SMILES']
    for descriptor_name, _ in Descriptors.descList:
        header_row.append(descriptor_name)
    header_row.extend(['NumRotatableBonds', 'TPSA'])
    writer.writerow(header_row)

    # Write the descriptor values for each molecule
    for smiles, descriptor_values_2d, descriptor_values_3d in zip(smiles_list, descriptors_2d, descriptors_3d):
        row = [smiles] + descriptor_values_2d + descriptor_values_3d
        writer.writerow(row)


------------
# Different Batch processing (continous)


import csv
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem

input_file = # set CSV SMILES input file
output_file = # Set CSV output file
smiles_column_index = 0  # Modify this to match the index of the SMILES column in your input file (zero-based)

# Collect the SMILES strings from the input file
smiles_list = []
with open(input_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        if len(row) > smiles_column_index:
            smiles = row[smiles_column_index]
            smiles_list.append(smiles)

# Calculate descriptors for each molecule
descriptors_2d = []
descriptors_3d = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)  # Add hydrogens to the molecule
        AllChem.EmbedMolecule(mol)  # Generate conformers
        AllChem.MMFFOptimizeMolecule(mol)  # Optimize conformers
        descriptor_values_2d = [Descriptors.descList[i][1](mol) for i in range(len(Descriptors.descList))]
        descriptor_values_3d = [rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcTPSA(mol)]
        descriptors_2d.append(descriptor_values_2d)
        descriptors_3d.append(descriptor_values_3d)

# Write the descriptor values to the output CSV file
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the descriptor names as the header row
    header_row = ['SMILES']
    for descriptor_name, _ in Descriptors.descList:
        header_row.append(descriptor_name)
    header_row.extend(['NumRotatableBonds', 'TPSA'])
    writer.writerow(header_row)

    # Write the descriptor values for each molecule
    for smiles, descriptor_values_2d, descriptor_values_3d in zip(smiles_list, descriptors_2d, descriptors_3d):
        row = [smiles] + descriptor_values_2d + descriptor_values_3d
        writer.writerow(row)
