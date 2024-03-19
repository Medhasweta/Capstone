import os.path as osp
import torch
import torch.optim as optim
from Readourdata import MetaQADataset
# from ReadingDatasetUrl import FB15kDataset
from random_corrupt_complex import ComplEx #, DistMult
# from torch_geometric.datasets import FB15k_237


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# FB15k_dset = FB15kDataset(root='FB15k')
FB15k_dset = MetaQADataset(root='/Users/medhaswetasen/Documents/GitHub/Capstone Data/__Parent__/EmbedKGQA/data/MetaQA')
data = FB15k_dset[0].to(device)

model_map = {

    'complex': ComplEx
#     'distmult': DistMult
}
model_name = 'complex'
# model_name  = 'distmult'

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

model = model_map[model_name](
    num_nodes=data.num_entities,
    num_relations=data.num_relations,
    hidden_channels=50).to(device)

loader = model.loader(
    head_index=data.train_edge_index[0],
    rel_type=data.train_edge_type,
    tail_index=data.train_edge_index[1],
    batch_size=1000,
    shuffle=True,
)
#
optimizer_map = {
    'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
    'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
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
def val(vdata):
    model.eval()
    return model.test(
        head_index=vdata.valid_edge_index[0],
        rel_type=vdata.valid_edge_type,
        tail_index=vdata.valid_edge_index[1],
        batch_size=20000,
        k=10,
    )

@torch.no_grad()
def testing(tdata):
    model.eval()
    return model.test(
        head_index=tdata.test_edge_index[0],
        rel_type=tdata.test_edge_type,
        tail_index=tdata.test_edge_index[1],
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