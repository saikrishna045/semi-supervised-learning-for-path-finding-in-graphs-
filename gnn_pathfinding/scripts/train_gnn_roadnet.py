import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import GraphData
from src.gnn_model import GraphSAGEPathModel

torch.manual_seed(42)
np.random.seed(42)

GRAPH_PATH = ROOT / "data" / "processed" / "demo_road" / "graph.npz"
LABEL_PATH = ROOT / "data" / "processed" / "demo_road" / "splits" / "labeled_10pct.csv"
MODEL_PATH = ROOT / "models" / "graphsage_roadnet.pt"

def main():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")

    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Labels not found: {LABEL_PATH}")

    graph = GraphData.load(GRAPH_PATH)
    G = graph.to_networkx()
    data = graph.to_pyg()

    labels = pd.read_csv(LABEL_PATH)

    # Build one training example per path step.
    # The correct next node is the next node in the known shortest path.
    examples = []

    for row in labels.itertuples(index=False):
        path = [int(n) for n in str(row.path).split()]

        if len(path) < 2:
            continue

        destination = path[-1]

        for i in range(len(path) - 1):
            current = path[i]
            correct_next = path[i + 1]

            if current not in G or correct_next not in G[current]:
                continue

            candidates = sorted(G.neighbors(current))

            if correct_next not in candidates:
                continue

            examples.append(
                (current, destination, candidates, correct_next)
            )

    if not examples:
        raise ValueError("No valid next-hop training examples were created.")

    # Limit training examples to keep the first experiment manageable.
    rng = np.random.default_rng(42)
    if len(examples) > 2500:
        selected = rng.choice(len(examples), size=2500, replace=False)
        examples = [examples[i] for i in sorted(selected)]

    print("Dataset:", graph.name)
    print("Nodes:", graph.num_nodes)
    print("Edges:", graph.num_edges)
    print("Node features:", data.x.shape[1])
    print("Labeled source-destination pairs:", len(labels))
    print("Next-hop training examples:", len(examples))

    model = GraphSAGEPathModel(input_dim=data.x.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.003,
        weight_decay=1e-4,
    )

    for epoch in range(1, 21):
        model.train()
        optimizer.zero_grad()

        embeddings = model(data)
        losses = []
        correct = 0

        for source, destination, candidates, target in examples:
            candidate_tensor = torch.tensor(
                candidates, dtype=torch.long
            )

            scores = model.score_candidates(
                embeddings,
                source,
                destination,
                candidate_tensor,
            )

            target_position = candidates.index(target)
            target_tensor = torch.tensor([target_position])

            losses.append(
                F.cross_entropy(scores.unsqueeze(0), target_tensor)
            )

            if int(candidate_tensor[torch.argmax(scores)]) == target:
                correct += 1

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0:
            accuracy = correct / len(examples)
            print(
                f"Epoch {epoch:02d}/20 | "
                f"Loss: {loss.item():.4f} | "
                f"Training next-hop accuracy: {accuracy:.2%}"
            )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(data.x.shape[1]),
            "hidden_dim": 64,
            "embedding_dim": 32,
            "dropout": 0.2,
            "dataset": graph.name,
            "seed": 42,
        },
        MODEL_PATH,
    )

    print("\nTraining completed.")
    print("Model saved to:", MODEL_PATH)
    print("Note: Training accuracy is not test accuracy.")

if __name__ == "__main__":
    main()
