from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data_loader import GraphData
from src.gnn_model import GraphSAGEPathModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "demo_road"
SPLIT_DIR = DATA_DIR / "splits"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

GRAPH_PATH = DATA_DIR / "graph.npz"
LABELED_PATH = SPLIT_DIR / "labeled_10pct.csv"
UNLABELED_PATH = SPLIT_DIR / "unlabeled_10pct.csv"
VAL_PATH = SPLIT_DIR / "val.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

SEED = 42
EPOCHS = 15
MAX_LABELED_EXAMPLES = 2500
MAX_PSEUDO_EXAMPLES = 2500
PSEUDO_WEIGHT = 0.5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")


def read_path_examples(graph, df):
    """Convert known paths into next-hop classification examples."""
    examples = []

    for row in df.itertuples(index=False):
        path_text = str(row.path).strip()

        try:
            path = [int(v) for v in path_text.split()]
        except ValueError:
            continue

        if len(path) < 2:
            continue

        src = int(row.src)
        dst = int(row.dst)

        if path[0] != src or path[-1] != dst:
            continue

        valid = True
        for u, v in zip(path, path[1:]):
            if not graph.has_edge(u, v):
                valid = False
                break

        if not valid:
            continue

        for current, next_node in zip(path, path[1:]):
            candidates = sorted(graph.neighbors(current))

            if next_node not in candidates:
                continue

            examples.append(
                (current, dst, candidates, next_node)
            )

    return examples


def make_dijkstra_examples(graph, df):
    """Generate reference next-hop examples for unlabeled pairs."""
    examples = []
    skipped = 0

    for row in df.itertuples(index=False):
        src, dst = int(row.src), int(row.dst)

        try:
            path = nx.shortest_path(
                graph, source=src, target=dst, weight="weight"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            skipped += 1
            continue

        for current, next_node in zip(path, path[1:]):
            candidates = sorted(graph.neighbors(current))

            examples.append(
                (current, dst, candidates, next_node)
            )

    print(f"Unlabeled pairs without a path: {skipped}")
    return examples


def limit_examples(examples, maximum, seed):
    """Reproducibly sample examples when the dataset is large."""
    if len(examples) <= maximum:
        return examples

    rng = random.Random(seed)
    return rng.sample(examples, maximum)


def train_model(data, labeled_examples, pseudo_examples, model_name):
    model = GraphSAGEPathModel(
        input_dim=data.x.shape[1],
        hidden_dim=64,
        embedding_dim=32,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.003,
        weight_decay=1e-4,
    )

    labeled_examples = limit_examples(
        labeled_examples, MAX_LABELED_EXAMPLES, SEED
    )

    if pseudo_examples:
        pseudo_examples = limit_examples(
            pseudo_examples, MAX_PSEUDO_EXAMPLES, SEED + 1
        )

    print(f"\nTraining: {model_name}")
    print(f"Labeled examples: {len(labeled_examples)}")
    print(f"Pseudo-labeled examples: {len(pseudo_examples)}")

    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        embeddings = model(data)
        losses = []

        for current, destination, candidates, correct_next in labeled_examples:
            candidate_tensor = torch.tensor(
                candidates, dtype=torch.long, device=device
            )
            scores = model.score_candidates(
                embeddings, current, destination, candidate_tensor
            )
            target = torch.tensor(
                [candidates.index(correct_next)],
                dtype=torch.long,
                device=device,
            )
            losses.append(F.cross_entropy(scores.unsqueeze(0), target))

        labeled_loss = torch.stack(losses).mean()

        if pseudo_examples:
            pseudo_losses = []

            for current, destination, candidates, correct_next in pseudo_examples:
                candidate_tensor = torch.tensor(
                    candidates, dtype=torch.long, device=device
                )
                scores = model.score_candidates(
                    embeddings, current, destination, candidate_tensor
                )
                target = torch.tensor(
                    [candidates.index(correct_next)],
                    dtype=torch.long,
                    device=device,
                )
                pseudo_losses.append(
                    F.cross_entropy(scores.unsqueeze(0), target)
                )

            pseudo_loss = torch.stack(pseudo_losses).mean()
            loss = labeled_loss + PSEUDO_WEIGHT * pseudo_loss
        else:
            loss = labeled_loss

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"labeled loss={labeled_loss.item():.4f} | "
                f"total loss={loss.item():.4f}"
            )

    elapsed = time.perf_counter() - start

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / f"{model_name}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(data.x.shape[1]),
            "hidden_dim": 64,
            "embedding_dim": 32,
            "dropout": 0.2,
            "seed": SEED,
            "training_type": model_name,
        },
        checkpoint_path,
    )

    print(f"Saved: {checkpoint_path}")
    print(f"Training time: {elapsed:.2f} seconds")

    return model, elapsed


def main():
    print("Loading graph and datasets...")

    graph_data = GraphData.load(GRAPH_PATH)
    graph = graph_data.to_networkx()
    data = graph_data.to_pyg().to(device)

    labeled_df = pd.read_csv(LABELED_PATH)
    unlabeled_df = pd.read_csv(UNLABELED_PATH)

    # Load validation and test sets only for reporting their sizes.
    # Their paths are never used to train either model.
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Node features: {data.x.shape[1]}")
    print(f"Labeled pairs: {len(labeled_df)}")
    print(f"Unlabeled pairs: {len(unlabeled_df)}")
    print(f"Validation pairs reserved: {len(val_df)}")
    print(f"Test pairs reserved: {len(test_df)}")

    labeled_examples = read_path_examples(graph, labeled_df)

    if not labeled_examples:
        raise RuntimeError(
            "No labeled training examples were parsed. "
            "Check the path column format."
        )

    print(f"Next-hop examples from labeled paths: {len(labeled_examples)}")

    # Experiment A: supervised training only.
    supervised_model, supervised_time = train_model(
        data,
        labeled_examples,
        [],
        "graphsage_supervised_10pct",
    )

    # Experiment B: add Dijkstra-generated labels for unlabeled pairs.
    print("\nGenerating reference paths for unlabeled pairs...")
    pseudo_examples = make_dijkstra_examples(graph, unlabeled_df)

    print(f"Generated pseudo-labeled next-hop examples: {len(pseudo_examples)}")

    semi_model, semi_time = train_model(
        data,
        labeled_examples,
        pseudo_examples,
        "graphsage_pseudolabeled_10pct",
    )

    summary = pd.DataFrame(
        [
            {
                "experiment": "Supervised GraphSAGE",
                "labeled_pairs": len(labeled_df),
                "unlabeled_pairs_used": 0,
                "training_seconds": round(supervised_time, 2),
                "checkpoint": "models/graphsage_supervised_10pct.pt",
            },
            {
                "experiment": "Dijkstra-pseudo-labeled GraphSAGE",
                "labeled_pairs": len(labeled_df),
                "unlabeled_pairs_used": len(unlabeled_df),
                "training_seconds": round(semi_time, 2),
                "checkpoint": "models/graphsage_pseudolabeled_10pct.pt",
            },
        ]
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULT_DIR / "semi_supervised_training_summary.csv"
    summary.to_csv(output, index=False)

    print("\n--- TRAINING SUMMARY ---")
    print(summary.to_string(index=False))
    print(f"\nSummary saved to: {output}")
    print(
        "\nTraining completed. These results compare training configurations, "
        "not test-set routing accuracy. Evaluate both checkpoints separately."
    )


if __name__ == "__main__":
    main()


