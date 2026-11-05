import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gnn_model import GraphSAGEPathModel

torch.manual_seed(42)

MODEL_PATH = ROOT / "models" / "graphsage_path_model.pt"
LABELS_PATH = ROOT / "results" / "gnn_training_labels.csv"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model missing. Run scripts/train_gnn.py first.")

if not LABELS_PATH.exists():
    raise FileNotFoundError("Labels missing. Generate labels first.")

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
for u, v in graph.edges():
    edge_pairs.extend([(u, v), (v, u)])

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
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

labels = pd.read_csv(LABELS_PATH)
embeddings = model(data)

correct = 0
total = 0

print("\nGNN next-hop evaluation")
print("-" * 55)

with torch.no_grad():
    for (source, destination), group in labels.groupby(["source", "destination"]):
        candidates = torch.tensor(
            group["candidate"].tolist(),
            dtype=torch.long
        )

        scores = model.score_candidates(
            embeddings,
            int(source),
            int(destination),
            candidates
        )

        predicted_candidate = int(candidates[torch.argmax(scores)].item())

        correct_candidates = group.loc[group["label"] == 1, "candidate"].tolist()
        expected_candidate = int(correct_candidates[0])

        is_correct = predicted_candidate == expected_candidate
        correct += int(is_correct)
        total += 1

        print(
            f"Source {source} -> Destination {destination} | "
            f"Expected next hop: {expected_candidate} | "
            f"Predicted: {predicted_candidate} | "
            f"{'CORRECT' if is_correct else 'INCORRECT'}"
        )

accuracy = correct / total if total else 0.0

print("-" * 55)
print(f"Correct predictions: {correct}/{total}")
print(f"Next-hop accuracy: {accuracy:.2%}")
print("\nNote: This is evaluation on the same synthetic examples used for training.")
print("It does not demonstrate generalization to unseen road networks.")
