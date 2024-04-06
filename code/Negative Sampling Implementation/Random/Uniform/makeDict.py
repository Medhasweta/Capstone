import torch
import torch_geometric
import os
import numpy as np
import pickle

with open('/home/ubuntu/capstone/data/MetaQA/raw/entities.dict', 'r') as f:
    lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
    entities_dict = {key: int(value) for key, value in lines}

entity_embeddings = np.load('/home/ubuntu/capstone/code/Negative Sampling Implementation/Random/Uniform/E.npy')

with open('entities_dict_tensors.pkl', 'rb') as pickle_file:
    e = pickle.load(pickle_file)