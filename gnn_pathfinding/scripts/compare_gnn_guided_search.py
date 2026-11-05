import sys
import time
import heapq
from pathlib import Path
from itertools import count

import numpy as np
import pandas as pd
import torch
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import GraphData
from src.gnn_model import GraphSAGEPathModel


DATA_DIR = ROOT / "data" / "processed" / "demo_road"
GRAPH_PATH = DATA_DIR / "graph.npz"
TEST_PATH = DATA_DIR / "splits" / "test.csv"
MODEL_PATH = ROOT / "models" / "graphsage_roadnet.pt"
OUTPUT_PATH = ROOT / "results" / "gnn_guided_search_comparison.csv"

DEVICE = torch.device("cpu")
MAX_EXPANSIONS = 10000


def load_model_and_graph():
    graph_data = GraphData.load(GRAPH_PATH)
    graph = graph_data.to_networkx()

    pyg_data = graph_data.to_pyg().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = GraphSAGEPathModel(input_dim=pyg_data.x.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        embeddings = model(pyg_data)

    embeddings = embeddings.detach()

    return graph, model, embeddings


def reconstruct_path(parent, destination):
    if destination not in parent:
        return None

    path = []
    current = destination

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path


def path_cost(graph, path):
    if path is None:
        return float("inf")

    return sum(
        float(graph[u][v].get("weight", 1.0))
        for u, v in zip(path[:-1], path[1:])
    )


def dijkstra_search(graph, source, destination):
    try:
        return nx.shortest_path(
            graph,
            source=source,
            target=destination,
            weight="weight",
            method="dijkstra",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def astar_zero_search(graph, source, destination):
    try:
        return nx.astar_path(
            graph,
            source=source,
            target=destination,
            heuristic=lambda _u, _v: 0,
            weight="weight",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def gnn_guided_search(graph, model, embeddings, source, destination):
    """
    Best-first search using GNN scores as a search-ordering signal.

    Actual accumulated edge cost is tracked separately. GNN scores do not
    replace edge weights, and this method does not guarantee optimality.
    """

    if source == destination:
        return [source]

    if source not in graph or destination not in graph:
        return None

    # Higher model scores are preferred. Convert them into a priority
    # ordering while keeping the real route cost separately.
    serial = count()
    frontier = []
    heapq.heappush(frontier, (0.0, next(serial), source))

    best_cost = {source: 0.0}
    parent = {source: None}
    closed = set()
    expansions = 0

    while frontier and expansions < MAX_EXPANSIONS:
        _, _, current = heapq.heappop(frontier)

        if current in closed:
            continue

        if current == destination:
            return reconstruct_path(parent, destination)

        closed.add(current)
        expansions += 1

        neighbors = list(graph.neighbors(current))
        if not neighbors:
            continue

        candidates = torch.tensor(
            neighbors,
            dtype=torch.long,
            device=embeddings.device,
        )

        with torch.no_grad():
            scores = model.score_candidates(
                embeddings,
                source,
                destination,
                candidates,
            )

        # Sort neighbors by descending GNN score.
        score_order = torch.argsort(scores, descending=True).tolist()

        # The score controls exploration order; accumulated real edge
        # costs still determine the cost recorded for each discovered node.
        for index in score_order:
            neighbor = neighbors[index]

            if neighbor in closed:
                continue

            weight = float(
                graph[current][neighbor].get("weight", 1.0)
            )
            new_cost = best_cost[current] + weight

            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                parent[neighbor] = current

                # Small deterministic tie-breaker; GNN score is the
                # main exploration-ordering signal.
                priority = -float(scores[index].item())

                heapq.heappush(
                    frontier,
                    (priority, next(serial), neighbor),
                )

    return None


def evaluate_route(graph, algorithm, search_function, source, destination):
    start_time = time.perf_counter()
    path = search_function(source, destination)
    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    success = path is not None
    cost = path_cost(graph, path) if success else float("inf")

    return {
        "algorithm": algorithm,
        "src": source,
        "dst": destination,
        "success": success,
        "runtime_ms": runtime_ms,
        "path_cost": cost,
        "path_length_edges": len(path) - 1 if success else np.nan,
    }


def main():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test split not found: {TEST_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run scripts/train_gnn_roadnet.py first."
        )

    graph, model, embeddings = load_model_and_graph()
    test_df = pd.read_csv(TEST_PATH)

    print(f"Dataset: demo_road")
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Held-out test routes: {len(test_df)}")
    print("Evaluating Dijkstra, A*, and GNN-guided search...")

    rows = []

    for index, row in test_df.iterrows():
        source = int(row["src"])
        destination = int(row["dst"])

        rows.append(
            evaluate_route(
                graph,
                "Dijkstra",
                lambda s, d: dijkstra_search(graph, s, d),
                source,
                destination,
            )
        )

        rows.append(
            evaluate_route(
                graph,
                "A* (zero heuristic)",
                lambda s, d: astar_zero_search(graph, s, d),
                source,
                destination,
            )
        )

        rows.append(
            evaluate_route(
                graph,
                "GNN-guided best-first",
                lambda s, d: gnn_guided_search(
                    graph, model, embeddings, s, d
                ),
                source,
                destination,
            )
        )

        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{len(test_df)} routes")

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    summary = results.groupby("algorithm").agg(
        routes=("success", "size"),
        success_rate=("success", lambda values: values.mean() * 100),
        average_runtime_ms=("runtime_ms", "mean"),
        successful_routes=("success", "sum"),
        average_successful_path_cost=(
            "path_cost",
            lambda values: values[np.isfinite(values)].mean()
            if np.isfinite(values).any()
            else np.nan,
        ),
        average_successful_path_length=(
            "path_length_edges",
            "mean",
        ),
    )

    print("\n===== GNN-GUIDED SEARCH RESULTS =====")
    print(summary.round(3).to_string())
    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

