import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gnn_model import GraphSAGEPathModel

torch.manual_seed(42)

labels_file = ROOT / "results" / "gnn_training_labels.csv"
if not labels_file.exists():
    raise FileNotFoundError(
        "Labels file not found. Run scripts/generate_gnn_labels.py first."
    )

labels = pd.read_csv(labels_file)

graph = nx.Graph()
graph.add_weighted_edges_from([
    (0, 1, 1.0),
    (1, 2, 1.0),
    (2, 3, 1.0),
    (3, 4, 1.0),
    (4, 5, 1.0),
    (0, 2, 3.0),
    (1, 4, 4.0),
    (2, 5, 5.0),
])

edge_pairs = []
edge_weights = []

for u, v, attributes in graph.edges(data=True):
    edge_pairs.append((u, v))
    edge_pairs.append((v, u))
    edge_weights.extend([attributes["weight"], attributes["weight"]])

edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

num_nodes = graph.number_of_nodes()
degree = torch.tensor(
    [graph.degree(node) for node in range(num_nodes)],
    dtype=torch.float32
).view(-1, 1)

node_ids = torch.arange(num_nodes, dtype=torch.float32).view(-1, 1)
x = torch.cat([degree, node_ids / max(num_nodes - 1, 1)], dim=1)

data = Data(x=x, edge_index=edge_index)

model = GraphSAGEPathModel(input_dim=x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

source = torch.tensor(labels["source"].to_numpy(), dtype=torch.long)
destination = torch.tensor(labels["destination"].to_numpy(), dtype=torch.long)
candidate = torch.tensor(labels["candidate"].to_numpy(), dtype=torch.long)
target = torch.tensor(labels["label"].to_numpy(), dtype=torch.float32)

print("Starting GraphSAGE training...")
print("Training rows:", len(labels))

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()

    embeddings = model(data)
    logits = []

    for i in range(len(labels)):
        scores = model.score_candidates(
            embeddings,
            int(source[i]),
            int(destination[i]),
            candidate[i].view(1)
        )
        logits.append(scores.squeeze(0))

    logits = torch.stack(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target)

    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 10 == 0:
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = (predictions == target).float().mean().item()
        print(
            f"Epoch {epoch:03d} | Loss: {loss.item():.4f} "
            f"| Training accuracy: {accuracy:.3f}"
        )

model_dir = ROOT / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "graphsage_path_model.pt"

torch.save(model.state_dict(), model_path)

print("Training completed.")
print("Model saved to:", model_path)
