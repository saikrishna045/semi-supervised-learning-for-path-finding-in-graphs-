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
OUTPUT_PATH = ROOT / "results" / "cost_aware_gnn_comparison.csv"

DEVICE = torch.device("cpu")
MAX_EXPANSIONS = 10000


def load_everything():
    graph_data = GraphData.load(GRAPH_PATH)
    graph = graph_data.to_networkx()
    pyg_data = graph_data.to_pyg().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = GraphSAGEPathModel(input_dim=pyg_data.x.shape[1])
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        embeddings = model(pyg_data)

    return graph, model, embeddings.detach()


def route_cost(graph, path):
    if path is None:
        return float("inf")
    return sum(
        float(graph[u][v].get("weight", 1.0))
        for u, v in zip(path[:-1], path[1:])
    )


def shortest_path(graph, source, destination):
    try:
        return nx.dijkstra_path(
            graph, source, destination, weight="weight"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def current_gnn_search(graph, model, embeddings, source, destination):
    """Reproduce the existing GNN-prioritized search strategy."""
    if source == destination:
        return [source]

    serial = count()
    frontier = [(0.0, next(serial), source)]
    best_cost = {source: 0.0}
    parent = {source: None}
    closed = set()
    expansions = 0

    while frontier and expansions < MAX_EXPANSIONS:
        _, _, current = heapq.heappop(frontier)

        if current in closed:
            continue

        if current == destination:
            return reconstruct(parent, destination)

        closed.add(current)
        expansions += 1
        neighbors = list(graph.neighbors(current))

        if not neighbors:
            continue

        candidates = torch.tensor(neighbors, dtype=torch.long)
        with torch.no_grad():
            scores = model.score_candidates(
                embeddings, source, destination, candidates
            )

        for index in torch.argsort(scores, descending=True).tolist():
            neighbor = neighbors[index]
            if neighbor in closed:
                continue

            weight = float(graph[current][neighbor].get("weight", 1.0))
            new_cost = best_cost[current] + weight

            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(
                    frontier,
                    (-float(scores[index]), next(serial), neighbor),
                )

    return None


def reconstruct(parent, destination):
    if destination not in parent:
        return None

    path = []
    current = destination

    while current is not None:
        path.append(current)
        current = parent[current]

    return path[::-1]


def cost_aware_search(
    graph, model, embeddings, source, destination, alpha=0.25
):
    """
    Experimental weighted search.

    Priority = accumulated edge cost + alpha * learned score penalty.

    The GNN score is transformed into a bounded penalty so the accumulated
    route cost remains the dominant term. This is an experimental heuristic,
    not a guarantee of shortest-path optimality.
    """
    if source == destination:
        return [source]

    serial = count()
    frontier = [(0.0, next(serial), source)]
    best_cost = {source: 0.0}
    parent = {source: None}
    closed = set()
    expansions = 0

    while frontier and expansions < MAX_EXPANSIONS:
        _, _, current = heapq.heappop(frontier)

        if current in closed:
            continue

        if current == destination:
            return reconstruct(parent, destination)

        closed.add(current)
        expansions += 1
        neighbors = list(graph.neighbors(current))

        if not neighbors:
            continue

        candidates = torch.tensor(neighbors, dtype=torch.long)
        with torch.no_grad():
            scores = model.score_candidates(
                embeddings, source, destination, candidates
            )

        # Convert scores into a bounded penalty. The highest-scoring
        # candidate receives the lowest penalty.
        score_min = scores.min()
        score_max = scores.max()
        score_range = score_max - score_min

        if float(score_range) > 1e-8:
            normalized = (scores - score_min) / score_range
        else:
            normalized = torch.zeros_like(scores)

        penalties = 1.0 - normalized

        for index, neighbor in enumerate(neighbors):
            if neighbor in closed:
                continue

            weight = float(graph[current][neighbor].get("weight", 1.0))
            new_cost = best_cost[current] + weight

            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                parent[neighbor] = current

                priority = new_cost + alpha * float(penalties[index])
                heapq.heappush(
                    frontier,
                    (priority, next(serial), neighbor),
                )

    return None


def main():
    graph, model, embeddings = load_everything()
    tests = pd.read_csv(TEST_PATH)
    rows = []

    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Test routes: {len(tests)}")
    print("Running cost-aware GNN experiment...")

    for i, row in tests.iterrows():
        source, destination = int(row["src"]), int(row["dst"])

        methods = [
            ("Dijkstra", lambda: shortest_path(
                graph, source, destination
            )),
            ("Current GNN-guided", lambda: current_gnn_search(
                graph, model, embeddings, source, destination
            )),
            ("Cost-aware GNN alpha=0.25", lambda: cost_aware_search(
                graph, model, embeddings, source, destination, alpha=0.25
            )),
        ]

        optimal = shortest_path(graph, source, destination)
        optimal_cost = route_cost(graph, optimal)

        for name, function in methods:
            start = time.perf_counter()
            path = function()
            runtime = (time.perf_counter() - start) * 1000

            cost = route_cost(graph, path)
            success = path is not None

            rows.append({
                "algorithm": name,
                "src": source,
                "dst": destination,
                "success": success,
                "runtime_ms": runtime,
                "path_cost": cost,
                "optimal_cost": optimal_cost,
                "cost_ratio": (
                    cost / optimal_cost
                    if success and optimal_cost > 0
                    else np.nan
                ),
                "path_length_edges": (
                    len(path) - 1 if success else np.nan
                ),
            })

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(tests)} routes")

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    summary = results.groupby("algorithm").agg(
        routes=("success", "size"),
        success_rate=("success", "mean"),
        avg_runtime_ms=("runtime_ms", "mean"),
        avg_path_cost=("path_cost", "mean"),
        avg_cost_ratio=("cost_ratio", "mean"),
        avg_path_length=("path_length_edges", "mean"),
    )

    summary["success_rate"] *= 100

    print("\n===== COST-AWARE GNN RESULTS =====")
    print(summary.round(3).to_string())
    print(f"\nSaved detailed results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
