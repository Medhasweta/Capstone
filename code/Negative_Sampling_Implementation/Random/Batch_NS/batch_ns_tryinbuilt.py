#%%

import torch
import torch.optim as optim

from torch_geometric.datasets import FB15k_237
from batch_ns_complex import ComplEx #,
from batch_ns_distmult import DistMult
# DistMult

#%%

model_map = {

    'complex': ComplEx,
    'distmult': DistMult

}

#%%

# model_name = 'complex'
model_name  = 'distmult'

#%%

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = 'packageData'

#%%

train_data = FB15k_237(path, split='train')[0].to(device)
val_data = FB15k_237(path, split='val')[0].to(device)
test_data = FB15k_237(path, split='test')[0].to(device)

#%%

# model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[model_name](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=50
 #   **model_arg_map.get(model_name, {}),
).to(device)

#%%

loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=1000,
    shuffle=True,
)

#%%

optimizer_map = {

    'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
    'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

}

optimizer = optimizer_map[model_name]

#%%

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

#%%
@torch.no_grad()
def test(data):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=20000,
        k=10,
    )

#%%

for epoch in range(1, 51):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch % 25 == 0:
        rank, mrr, hits = test(val_data)
        print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
              f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')

#%%

rank, mrr, hits_at_10 = test(test_data)
print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
      f'Test Hits@10: {hits_at_10:.4f}')

# input(dimension)/model(nodes and edges)/model output/loss/learnable parameter