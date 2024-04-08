#%%
import torch
import torch_geometric
import os
from Readourdata import MetaQADataset


# # print(os.getcwd())
# class MetaQADataset(torch_geometric.data.InMemoryDataset):
#     r"""FB15-237 dataset from Freebase.
#     Follows similar structure to torch_geometric.datasets.rel_link_pred_dataset

#     Args:
#       root (string): Root directory where the dataset should be saved.
#       transform (callable, optional): A function/transform that takes in an
#           :obj:`torch_geometric.data.Data` object and returns a transformed
#           version. The data object will be transformed before every access.
#           (default: :obj:`None`)
#       pre_transform (callable, optional): A function/transform that takes in
#           an :obj:`torch_geometric.data.Data` object and returns a
#           transformed version. The data object will be transformed before
#           being saved to disk. (default: :obj:`None`)
#     """
#     # data_path = 'https://raw.githubusercontent.com/DeepGraphLearning/' \
#     #             'KnowledgeGraphEmbedding/master/data/FB15k-237'

#     def __init__(self, root, transform=None, pre_transform=None):
#         super().__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return ['train.txt', 'valid.txt', 'test.txt',
#                 'entities.dict', 'relations.dict']

#     @property
#     def processed_file_names(self):
#         return ['data.pt'] # returns the data file that has read the KG

#     @property
#     def raw_dir(self):
#         return os.path.join(self.root, 'raw')

#     def download(self):
#         # for file_name in self.raw_file_names:
#         #     torch_geometric.data.download_url(f'{self.data_path}/{file_name}',
#         #                                       self.raw_dir)
#         pass

#     def process(self):
#         with open(os.path.join(self.raw_dir, 'entities.dict'), 'r') as f:
#             lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
#             entities_dict = {key: int(value) for key,value in lines}

#         with open(os.path.join(self.raw_dir, 'relations.dict'), 'r') as f:
#             lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
#             relations_dict = {key: int(value) for key,value in lines}

#         kwargs = {}
#         for split in ['train', 'valid', 'test']:
#             with open(os.path.join(self.raw_dir, f'{split}.txt'), 'r') as f:
#                 lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
#                 heads = [entities_dict[row[0]] for row in lines]
#                 relations = [relations_dict[row[1]] for row in lines]
#                 tails = [entities_dict[row[2]] for row in lines]
#                 kwargs[f'{split}_edge_index'] = torch.tensor([heads, tails])
#                 kwargs[f'{split}_edge_type'] = torch.tensor(relations)

#         _data = torch_geometric.data.Data(num_entities=len(entities_dict),
#                                           num_relations=len(relations_dict),
#                                           **kwargs)

#         if self.pre_transform is not None:
#             _data = self.pre_transform(_data)

#         data, slices = self.collate([_data])

#         torch.save((data, slices), self.processed_paths[0])


device = 'cuda' if torch.cuda.is_available() else 'cpu'


FB15k_dset = MetaQADataset(root='/Users/yoninayoni/Documents/GitHub/Capstone/code/Negative Sampling Implementation/Random/Uniform/MetaQA')
data = FB15k_dset[0].to(device)

print(f'The graph has a total of {data.num_entities} entities and {data.num_relations} relations.') #43234
print(f'The train split has {data.train_edge_type.size()[0]} relation triples.') # The train split has 133582 relation triples.
print(f'The valid split has {data.valid_edge_type.size()[0]} relation triples.') # The valid split has 4052 relation triples.
print(f'The test split has {data.test_edge_type.size()[0]} relation triples.') # The test split has 4053 relation triples.




#%%
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # Define a GCN-based entity embedding model
# class EntityEmbedding(nn.Module):
#     def __init__(self, num_entities, num_relations, embedding_dim):
#         super(EntityEmbedding, self).__init__()
#         self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
#         self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

#     def forward(self, head, relation, tail):
#         # Lookup embeddings for head and tail entities
#         head_emb = self.entity_embedding(edge_index[0])
#         tail_emb = self.entity_embedding(edge_index[1])

#         # Lookup embeddings for relations
#         relation_emb = self.relation_embedding(edge_type)

#         # Compute the composite embedding for each triple
#         composite_embed = head_emb + relation_emb

#         return composite_embed


# # Prepare data
# data = data.to(device)
# num_entities = data.num_entities
# num_relations = data.num_relations
# edge_index = data.train_edge_index.to(device)
# edge_type = data.train_edge_type.to(device)

# # Define model and optimizer
# model = EntityEmbedding(num_entities, num_relations, embedding_dim=100).to(device)




#%%
