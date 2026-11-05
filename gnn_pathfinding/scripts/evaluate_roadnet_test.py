import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import GraphData
from src.gnn_model import GraphSAGEPathModel

GRAPH_PATH = ROOT / "data" / "processed" / "demo_road" / "graph.npz"
TEST_PATH = ROOT / "data" / "processed" / "demo_road" / "splits" / "test.csv"
MODEL_PATH = ROOT / "models" / "graphsage_roadnet.pt"
OUTPUT_PATH = ROOT / "results" / "roadnet_test_comparison.csv"


def route_cost(graph, path):
    if path is None or len(path) < 2:
        return 0.0 if path is not None else float("nan")

    return sum(
        float(graph[u][v].get("weight", 1.0))
        for u, v in zip(path, path[1:])
    )


def gnn_route(graph, model, data, embeddings, source, destination):
    current = source
    path = [current]
    visited = {current}

    for _ in range(min(graph.number_of_nodes(), 500)):
        if current == destination:
            return path

        candidates = sorted(
            node for node in graph.neighbors(current)
            if node not in visited
        )

        if not candidates:
            return None

        candidate_tensor = torch.tensor(candidates, dtype=torch.long)

        with torch.no_grad():
            scores = model.score_candidates(
                embeddings,
                current,
                destination,
                candidate_tensor,
            )

        current = candidates[int(torch.argmax(scores).item())]
        path.append(current)
        visited.add(current)

    return path if current == destination else None


def timed_route(function):
    start = time.perf_counter()
    result = function()
    runtime_ms = (time.perf_counter() - start) * 1000
    return result, runtime_ms


def main():
    for file_path in (GRAPH_PATH, TEST_PATH, MODEL_PATH):
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    graph_data = GraphData.load(GRAPH_PATH)
    graph = graph_data.to_networkx()
    data = graph_data.to_pyg()
    test_df = pd.read_csv(TEST_PATH)

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

    # Embedding computation is shared and excluded from per-route GNN runtime.
    with torch.no_grad():
        embeddings = model(data)

    results = []
    total_routes = len(test_df)

    print("Dataset:", graph_data.name)
    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())
    print("Held-out test routes:", total_routes)
    print("\nEvaluating Dijkstra, A*, and GNN...")

    for index, row in enumerate(test_df.itertuples(index=False), start=1):
        source = int(row.src)
        destination = int(row.dst)

        algorithms = []

        try:
            path, runtime = timed_route(
                lambda: nx.dijkstra_path(
                    graph, source, destination, weight="weight"
                )
            )
            algorithms.append(("Dijkstra", path, runtime))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            algorithms.append(("Dijkstra", None, float("nan")))

        try:
            path, runtime = timed_route(
                lambda: nx.astar_path(
                    graph,
                    source,
                    destination,
                    heuristic=lambda u, v: 0,
                    weight="weight",
                )
            )
            algorithms.append(("A* (zero heuristic)", path, runtime))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            algorithms.append(("A* (zero heuristic)", None, float("nan")))

        path, runtime = timed_route(
            lambda: gnn_route(
                graph, model, data, embeddings, source, destination
            )
        )
        algorithms.append(("GNN", path, runtime))

        for name, path, runtime in algorithms:
            success = path is not None and path[-1] == destination

            results.append({
                "source": source,
                "destination": destination,
                "algorithm": name,
                "success": success,
                "runtime_ms": runtime,
                "path_cost": route_cost(graph, path) if success else float("nan"),
                "path_length_edges": len(path) - 1 if success else float("nan"),
            })

        if index % 25 == 0 or index == total_routes:
            print(f"Processed {index}/{total_routes} routes")

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    summary = results_df.groupby("algorithm").agg(
        routes=("success", "size"),
        success_rate=("success", "mean"),
        average_runtime_ms=("runtime_ms", "mean"),
        successful_routes=("path_cost", "count"),
        average_successful_path_cost=("path_cost", "mean"),
        average_successful_path_length=("path_length_edges", "mean"),
    )

    summary["success_rate"] *= 100

    print("\n===== HELD-OUT TEST RESULTS =====")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))

    print("\nDetailed results saved to:", OUTPUT_PATH)
    print("\nInterpretation:")
    print("- The test source-destination pairs were not used in supervised training.")
    print("- A* uses a zero heuristic, so it behaves like Dijkstra.")
    print("- GNN runtime excludes the one-time embedding computation.")
    print("- Path costs are compared only among successful routes.")
    print("- The graph uses simulated weights in this demo dataset.")


if __name__ == "__main__":
    main()
