import numpy as np
import os

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
