import os.path as osp
import torch
import torch.optim as optim
from Readourdata import MetaQADataset
# from ReadingDatasetUrl import FB15kDataset
from torch_geometric.nn import ComplEx, DistMult, RotatE # from torch_geometric.datasets import FB15k_237


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# FB15k_dset = MetaQADataset(root='packageData')
FB15k_dset = MetaQADataset(root='/home/ubuntu/.cache/Capstone/code/negative_sampling/Random/Uniform/MetaQA')

# FB15k_dset = MetaQADataset(root='/home/ubuntu/capstone/data/fbwq')
data = FB15k_dset[0].to(device)

model_map = {

    'complex': ComplEx,
    'distmult': DistMult,
    'rotate': RotatE

}
# model_name = 'complex'
model_name  = 'distmult'
# model_name = 'rotate'

# parser = argparse.ArgumentParser()
# parser.add_argument('--model', choices=model_map.keys(), type=str.lower,
#                     required=True)
# args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# path = 'packageData'

# train_data = FB15k_237(path, split='train')[0].to(device)
# val_data = FB15k_237(path, split='val')[0].to(device)
# test_data = FB15k_237(path, split='test')[0].to(device)
# print(len(train_data))
#
# model_arg_map = {'rotate': {'margin': 9.0}}
# model = model_map[args.model](
#     num_nodes=train_data.num_nodes,
#     num_relations=train_data.num_edge_types,
#     hidden_channels=50,
#     **model_arg_map.get(args.model, {}),
# ).to(device)
#
model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[model_name](
    num_nodes=data.num_entities,
    num_relations=data.num_relations,
    hidden_channels=50, #each entity and each relation in the graph is represented as a 50-dimensional vector.
    **model_arg_map.get(model_name, {}),
).to(device)

loader = model.loader(
    head_index=data.train_edge_index[0], #tensor with two rows, columns reps edge, first row contains the indices of the head entities, second row contains the tail entities 
    rel_type=data.train_edge_type, #This tells the model what kind of relation connects the head and tail entities.
    tail_index=data.train_edge_index[1],
    batch_size=1000, #Specifies how many edges (head, relation type, tail triples) to process in each batch. A batch size of 1000 means that the model will process 1000 edges at a time during training. 
    shuffle=True,
)
#
optimizer_map = {

    'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6), #default lr applied and small L2 regulariation pressure applied,  Given the sparse nature of knowledge graphs, Adagrad's mechanism of adapting learning rates to the frequency of features helps ensure that less frequent relations and entities still receive sufficient attention during the training process
    'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
    'rotate': optim.Adam(model.parameters(), lr=1e-3)

}

optimizer = optimizer_map[model_name]

#
#
def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples
#
#

@torch.no_grad()
def val(data):
    model.eval()
    return model.test(
        head_index=data.valid_edge_index[0],
        rel_type=data.valid_edge_type,
        tail_index=data.valid_edge_index[1],
        batch_size=20000,
        k=10,
    )

@torch.no_grad()
def testing(data):
    model.eval()
    return model.test(
        head_index=data.test_edge_index[0],
        rel_type=data.test_edge_type,
        tail_index=data.test_edge_index[1],
        batch_size=20000,
        k=10,
    )

#
for epoch in range(1, 51):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch % 25 == 0:
        rank, mrr, hits = val(data)
        print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
              f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')
#
rank, mrr, hits_at_10 = testing(data)
print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
      f'Test Hits@10: {hits_at_10:.4f}')


# Mean Rank
# Mean Rank is the average rank of the true entities (either head or tail) among all entities in the dataset ranked by the model's predicted scores. For each test triple (head, relation, tail), the model predicts a score for every possible entity replacing the head or tail. Then, it ranks these entities based on their scores. The rank of the correct entity is recorded, and the mean rank is calculated across all test triples. A lower mean rank indicates better model performance, as it means the true entities are, on average, ranked closer to the top.
# Mean Reciprocal Rank (MRR)
# Mean Reciprocal Rank (MRR) is another metric to evaluate the model's performance. It is the average of the reciprocal ranks of the true entities for all the test triples. The reciprocal rank is the inverse of the rank of the correct entity. If the correct entity is ranked first, the reciprocal rank is 1; if it's ranked second, the reciprocal rank is 0.5, and so on. MRR gives a higher weight to the top-ranked entities, making it a more stringent metric than the mean rank. Higher MRR values indicate better performance.
# Hits@k
# Hits@k measures the proportion of correct entities that are ranked within the top-k positions by the model. For instance, Hits@1 measures the percentage of times the correct entity is ranked first, Hits@3 measures the percentage of times the correct entity is among the top 3, and so on. This metric is useful for understanding how often the model's predictions are highly accurate. Higher Hits@k values indicate better performance, and it's common to report this metric for various values of k (e.g., 1, 3, 10) to show the model's performance at different levels of ranking precision.

# dimensions
import numpy as np

for name, param in model.named_parameters():
    print(name)

# Assuming 'bn0' is your batch normalization layer
bn0_state = {
    'weight': model.bn0.weight.data.cpu().numpy(),
    'bias': model.bn0.bias.data.cpu().numpy(),
    'running_mean': model.bn0.running_mean.data.cpu().numpy(),
    'running_var': model.bn0.running_var.data.cpu().numpy()
}

# Now save it
np.save('bn0.npy', bn0_state)


# Correct key for entity embeddings
entity_embeddings = model.state_dict()['node_emb.weight'].cpu().numpy()
# Correct key for relation embeddings
relation_embeddings = model.state_dict()['rel_emb.weight'].cpu().numpy()

# Save embeddings
np.save('E.npy', entity_embeddings)  # Contains both real and imaginary parts
np.save('R.npy', relation_embeddings)

# Loading the embeddings
entity_embeddings = np.load('E.npy')
relation_embeddings = np.load('R.npy')

# Assuming you know the original dimension of the real (or imaginary) parts
embedding_dim = entity_embeddings.shape[1] // 2  # Assuming the second dimension contains concatenated parts

# Splitting back into real and imaginary parts if needed
real_part_entities = entity_embeddings[:, :embedding_dim]
imaginary_part_entities = entity_embeddings[:, embedding_dim:]

real_part_relations = relation_embeddings[:, :embedding_dim]
imaginary_part_relations = relation_embeddings[:, embedding_dim:]


import numpy as np

# Function to save the batch normalization state
def save_bn_state(bn_layer, filename):
    bn_state = {
        'weight': bn_layer.weight.data.cpu().numpy(),
        'bias': bn_layer.bias.data.cpu().numpy(),
        'running_mean': bn_layer.running_mean.data.cpu().numpy(),
        'running_var': bn_layer.running_var.data.cpu().numpy()
    }
    np.save(filename, bn_state)

# Example usage for saving bn0 state
save_bn_state(model.bn0, 'bn0_state.npy')