import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans

# Define the path to the CSV dataset file
dataset_path = # Set CSV file input source

# Read the CSV file into a pandas DataFrame
dataset = pd.read_csv(dataset_path)

# Initialize a list to store the molecules
molecules = []

# Iterate over the rows in the DataFrame and store the molecules
for _, row in dataset.iterrows():
    smiles = row['SMILES']  # Assuming the SMILES column is named 'SMILES'
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        molecules.append(mol)

# Calculate the fingerprints for each molecule
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in molecules]

# Define the number of diverse compounds to select
num_to_select = 5

# Perform k-means clustering on the fingerprints
kmeans = KMeans(n_clusters=num_to_select)
cluster_labels = kmeans.fit_predict(fingerprints)

# Get the representative compounds from each cluster
selected_indices = []
for cluster_id in range(num_to_select):
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    representative_index = max(cluster_indices, key=lambda i: len(Chem.MolToSmiles(molecules[i])))
    selected_indices.append(representative_index)

# Retrieve the selected compounds
selected_compounds = [molecules[i] for i in selected_indices]

# Print the selected compounds
for compound in selected_compounds:
    print(Chem.MolToSmiles(compound))
