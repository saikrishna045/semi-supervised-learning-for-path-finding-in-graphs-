import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from src.gnn_model import GraphSAGEPathModel

x = torch.randn(6, 5)

edge_index = torch.tensor(
    [
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
    ],
    dtype=torch.long,
)

model = GraphSAGEPathModel(input_dim=5)
model.eval()

with torch.no_grad():
    embeddings = model.encode(x, edge_index)
    candidates = torch.tensor([1, 3])
    scores = model.score_candidates(
        embeddings=embeddings,
        source=0,
        destination=5,
        candidates=candidates,
    )

print("Number of nodes:", x.shape[0])
print("Embedding shape:", tuple(embeddings.shape))
print("Candidate scores:", scores.tolist())

assert embeddings.shape == (6, 32)
assert scores.shape == (2,)

print("GNN model test passed!")
