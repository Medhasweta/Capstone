import os

with open(os.path.join('/Users/medhaswetasen/Documents/GitHub/Capstone/code/Tutorial Code/Temp Data/raw', 'entities.dict'), 'r') as f:
    lines = [row.split('\t') for row in f.read().split('\n') if row]  # Add `if row` to filter out empty strings
    entities_dict = {key: int(value) for value, key in lines if len(key) > 0 and len(value) > 0}  # Ensure both key and value are present

print(entities_dict)

for split in ['train']:
    with open(os.path.join('/Users/medhaswetasen/Documents/GitHub/Capstone/code/Tutorial Code', f'{split}.txt'), 'r') as f:
        lines = [row.split('\t') for row in f.read().split('\n')]
        heads = [entities_dict[row[0]] for row in lines]