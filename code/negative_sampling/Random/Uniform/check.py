import numpy as np
import os


from torch_geometric.nn import ComplEx


import torch

# Replace these paths with the actual file paths
path_to_entity_dict = '/home/ubuntu/.cache/Capstone/code/negative_sampling/Random/Uniform/MetaQA/raw/entities.dict'
path_to_relation_dict = '/home/ubuntu/.cache/Capstone/code/negative_sampling/Random/Uniform/MetaQA/raw/relations.dict'

# Load entity and relation dictionaries
# These dictionaries should map entity/relation names to unique integer IDs
with open(path_to_entity_dict, 'r') as f:
    entity_dict = f.read().splitlines()

with open(path_to_relation_dict, 'r') as f:
    relation_dict = f.read().splitlines()

# Calculate the number of unique entities and relations
num_entities = len(entity_dict)
num_relations = len(relation_dict)

# Print the numbers to verify
print(f"Number of entities: {num_entities}")
print(f"Number of relations: {num_relations}")

# Load the model's state dictionary if it contains the original embedding dimension
# Assuming you have saved the entity and relation embeddings in .npy format
entity_embeddings = np.load('E.npy')
relation_embeddings = np.load('R.npy')

# The embeddings consist of a real and an imaginary concatenated part
# so the original dimension for each part is half of the total embedding size
original_dim = entity_embeddings.shape[1] // 2

print(f"Original embedding dimension: {original_dim}")


# Load the embeddings
entity_embeddings = np.load('E.npy')

# Verify if the embeddings are concatenated by comparing dimensions
if entity_embeddings.shape[1] == 2 * original_dim:
    print(f"The embeddings are concatenated. Each part's dimension is {original_dim}.")
else:
    print("The embeddings may not be in concatenated form, or there is a mismatch in the dimensions.")



# Check the files
if os.path.exists('E.npy') and os.path.exists('R.npy'):
    print('Embedding files found!')
else:
    print('Embedding files not found.')

# Load and print shapes
entity_embeddings = np.load('E.npy')
relation_embeddings = np.load('R.npy')

print('Entity embeddings shape:', entity_embeddings.shape)
print('Relation embeddings shape:', relation_embeddings.shape)

# Print sample data
print('Sample entity embeddings:', entity_embeddings[:5])
print('Sample relation embeddings:', relation_embeddings[:5])
