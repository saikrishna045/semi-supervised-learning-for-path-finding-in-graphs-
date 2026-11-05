import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import GraphData
from src.gnn_model import GraphSAGEPathModel

GRAPH_PATH = ROOT / "data" / "processed" / "demo_road" / "graph.npz"
LABEL_PATH = ROOT / "data" / "processed" / "demo_road" / "splits" / "labeled_10pct.csv"
MODEL_PATH = ROOT / "models" / "graphsage_roadnet.pt"


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")

    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Labels not found: {LABEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. Run train_gnn_roadnet.py first."
        )

    graph = GraphData.load(GRAPH_PATH)
    G = graph.to_networkx()
    data = graph.to_pyg()
    labels = pd.read_csv(LABEL_PATH)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = GraphSAGEPathModel(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        embedding_dim=checkpoint["embedding_dim"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    examples = []

    # Build next-hop examples from the stored shortest paths.
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

            if correct_next in candidates:
                examples.append(
                    (current, destination, candidates, correct_next)
                )

    if not examples:
        raise ValueError("No evaluation examples were created.")

    # Evaluate on a reproducible subset to keep runtime manageable.
    rng = np.random.default_rng(42)

    if len(examples) > 1000:
        selected = rng.choice(len(examples), size=1000, replace=False)
        examples = [examples[i] for i in sorted(selected)]

    correct = 0
    total = 0

    with torch.no_grad():
        embeddings = model(data)

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

            predicted = candidates[int(torch.argmax(scores).item())]

            if predicted == target:
                correct += 1

            total += 1

    accuracy = correct / total

    print("\n===== ROAD-NETWORK GNN EVALUATION =====")
    print("Dataset:", graph.name)
    print("Graph nodes:", graph.num_nodes)
    print("Graph edges:", graph.num_edges)
    print("Evaluation examples:", total)
    print(f"Next-hop accuracy: {accuracy:.2%}")
    print("\nImportant: These examples come from the labeled training paths.")
    print("This is a preliminary evaluation, not an independent test score.")


if __name__ == "__main__":
    main()
