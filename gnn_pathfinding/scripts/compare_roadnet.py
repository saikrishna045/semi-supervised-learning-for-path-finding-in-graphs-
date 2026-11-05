import sys
import time
from pathlib import Path

import pandas as pd
import networkx as nx
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import GraphData
from src.gnn_model import GraphSAGEPathModel

GRAPH_PATH = ROOT / "data" / "processed" / "demo_road" / "graph.npz"
LABEL_PATH = ROOT / "data" / "processed" / "demo_road" / "splits" / "labeled_10pct.csv"
MODEL_PATH = ROOT / "models" / "graphsage_roadnet.pt"
OUTPUT_PATH = ROOT / "results" / "roadnet_comparison.csv"

MAX_ROUTES = 100


def timed_call(function):
    start = time.perf_counter()
    result = function()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def path_cost(graph, path):
    if not path or len(path) < 2:
        return 0.0
    return sum(
        float(graph[u][v].get("weight", 1.0))
        for u, v in zip(path, path[1:])
    )


def run_gnn(graph, model, data, embeddings, source, destination):
    current = source
    route = [current]
    visited = {current}

    # Limit steps to avoid infinite loops.
    max_steps = min(graph.number_of_nodes(), 500)

    for _ in range(max_steps):
        if current == destination:
            return route

        neighbors = sorted(graph.neighbors(current))
        neighbors = [n for n in neighbors if n not in visited]

        if not neighbors:
            return None

        candidate_tensor = torch.tensor(neighbors, dtype=torch.long)

        with torch.no_grad():
            scores = model.score_candidates(
                embeddings,
                current,
                destination,
                candidate_tensor,
            )

        next_node = neighbors[int(torch.argmax(scores).item())]

        route.append(next_node)
        visited.add(next_node)
        current = next_node

    return route if current == destination else None


def main():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")
    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Labels not found: {LABEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train_gnn_roadnet.py first."
        )

    graph_data = GraphData.load(GRAPH_PATH)
    G = graph_data.to_networkx()
    data = graph_data.to_pyg()
    labels = pd.read_csv(LABEL_PATH)

    checkpoint = torch.load(
        MODEL_PATH, map_location="cpu", weights_only=False
    )
    model = GraphSAGEPathModel(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        embedding_dim=checkpoint["embedding_dim"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Use the validation/test pair columns if available in the split files.
    # The labeled training paths are used here only to choose route pairs.
    # This is a preliminary comparison, not an independent benchmark.
    routes = []
    for row in labels.itertuples(index=False):
        source = int(row.src)
        destination = int(row.dst)
        if source != destination and source in G and destination in G:
            routes.append((source, destination))

    # Keep the experiment manageable.
    routes = routes[:MAX_ROUTES]

    if not routes:
        raise ValueError("No valid source-destination pairs found.")

    print("Dataset:", graph_data.name)
    print("Nodes:", graph_data.num_nodes)
    print("Edges:", graph_data.num_edges)
    print("Routes:", len(routes))
    print("\nRunning Dijkstra, A*, and GNN...")

    # Calculate embeddings once, outside the per-route timer.
    with torch.no_grad():
        embeddings = model(data)

    results = []

    for index, (source, destination) in enumerate(routes, start=1):
        try:
            dijkstra_path, dijkstra_ms = timed_call(
                lambda: nx.dijkstra_path(
                    G, source, destination, weight="weight"
                )
            )
            dijkstra_success = True
            dijkstra_cost = path_cost(G, dijkstra_path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dijkstra_path = None
            dijkstra_success = False
            dijkstra_cost = float("nan")
            dijkstra_ms = float("nan")

        try:
            astar_path, astar_ms = timed_call(
                lambda: nx.astar_path(
                    G,
                    source,
                    destination,
                    heuristic=lambda u, v: 0,
                    weight="weight",
                )
            )
            astar_success = True
            astar_cost = path_cost(G, astar_path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            astar_path = None
            astar_success = False
            astar_cost = float("nan")
            astar_ms = float("nan")

        gnn_path, gnn_ms = timed_call(
            lambda: run_gnn(
                G, model, data, embeddings, source, destination
            )
        )
        gnn_success = (
            gnn_path is not None and gnn_path[-1] == destination
        )
        gnn_cost = path_cost(G, gnn_path) if gnn_success else float("nan")

        for name, success, runtime, cost in [
            ("Dijkstra", dijkstra_success, dijkstra_ms, dijkstra_cost),
            ("A* (zero heuristic)", astar_success, astar_ms, astar_cost),
            ("GNN", gnn_success, gnn_ms, gnn_cost),
        ]:
            results.append({
                "source": source,
                "destination": destination,
                "algorithm": name,
                "success": success,
                "runtime_ms": runtime,
                "path_cost": cost,
            })

        if index % 10 == 0 or index == len(routes):
            print(f"Processed {index}/{len(routes)} routes")

    df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n===== ROAD-NETWORK COMPARISON =====")
    summary = df.groupby("algorithm").agg(
        success_rate=("success", "mean"),
        average_runtime_ms=("runtime_ms", "mean"),
        average_path_cost=("path_cost", "mean"),
    )
    summary["success_rate"] *= 100
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nResults saved to:", OUTPUT_PATH)
    print("\nNotes:")
    print("- A* uses a zero heuristic, so it behaves like Dijkstra.")
    print("- GNN runtime excludes the one-time embedding calculation.")
    print("- Failed GNN routes have no path cost and are not included in the cost mean.")
    print("- These route pairs come from labeled training data; this is preliminary.")
    print("- The graph uses simulated weights in the current demo dataset.")


if __name__ == "__main__":
    main()
