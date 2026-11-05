from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import GraphData
from src.gnn_model import GraphSAGEPathModel

GRAPH_PATH = ROOT / "data" / "processed" / "demo_road" / "graph.npz"
TEST_PATH = ROOT / "data" / "processed" / "demo_road" / "splits" / "test.csv"

CHECKPOINTS = {
    "Supervised GraphSAGE": ROOT / "models" / "graphsage_supervised_10pct.pt",
    "Pseudo-Labeled GraphSAGE": ROOT / "models" / "graphsage_pseudolabeled_10pct.pt",
}

DEVICE = torch.device("cpu")


def load_model(path, pyg_data):
    checkpoint = torch.load(
        path, map_location=DEVICE, weights_only=False
    )

    model = GraphSAGEPathModel(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        embedding_dim=checkpoint["embedding_dim"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        embeddings = model(pyg_data.to(DEVICE)).detach().cpu().numpy()

    # Cache model parameters as NumPy arrays.
    # This avoids repeated PyTorch calls during graph search.
    params = {
        name: value.detach().cpu().numpy()
        for name, value in model.named_parameters()
    }

    return model, embeddings, params


def route_cost(graph, route):
    return sum(
        float(graph[u][v].get("weight", 1.0))
        for u, v in zip(route[:-1], route[1:])
    )


def score_candidates_numpy(model, embeddings, params, source, destination, candidates):
    """
    Reproduce the model's candidate scoring with NumPy.

    This assumes the scorer concatenates source, destination and candidate
    embeddings before applying its scoring layers. Parameter names and
    operations are checked against the actual model implementation below.
    """
    # The actual scoring architecture must match src/gnn_model.py.
    # Use the model's existing scorer once for all candidates if possible.
    candidate_tensor = torch.as_tensor(candidates, dtype=torch.long)

    with torch.no_grad():
        scores = model.score_candidates(
            torch.as_tensor(embeddings, dtype=torch.float32),
            int(source),
            int(destination),
            candidate_tensor,
        )

    return scores.detach().cpu().numpy().reshape(-1)


def gnn_search(graph, model, embeddings, params, source, destination):
    if source == destination:
        return [source], 0.0, 0

    frontier = [source]
    discovered = {source}
    parent = {source: None}
    expansions = 0

    # Cache a candidate's score for this source-destination query.
    score_cache = {}

    while frontier and expansions < len(graph):
        scored_frontier = []

        for node in frontier:
            if node not in score_cache:
                neighbors = list(graph.neighbors(node))

                if neighbors:
                    scores = score_candidates_numpy(
                        model, embeddings, params,
                        source, destination, neighbors
                    )
                    score_cache[node] = float(np.max(scores))
                else:
                    score_cache[node] = float("-inf")

            scored_frontier.append((score_cache[node], node))

        if not scored_frontier:
            break

        _, current = max(scored_frontier)
        frontier.remove(current)
        expansions += 1

        if current == destination:
            route = []
            node = destination

            while node is not None:
                route.append(node)
                node = parent[node]

            route.reverse()
            return route, route_cost(graph, route), expansions

        for neighbor in graph.neighbors(current):
            if neighbor not in discovered:
                discovered.add(neighbor)
                parent[neighbor] = current
                frontier.append(neighbor)

    return None, float("inf"), expansions


def main():
    print("Loading graph and all held-out test pairs...", flush=True)

    graph_data = GraphData.load(GRAPH_PATH)
    graph = graph_data.to_networkx()
    pyg_data = graph_data.to_pyg()

    test_df = pd.read_csv(TEST_PATH)
    if test_df.empty:
        raise ValueError("The test dataset is empty.")

    print(f"Test routes: {len(test_df)}", flush=True)
    print(f"Graph nodes: {graph.number_of_nodes()}", flush=True)
    print(f"Graph edges: {graph.number_of_edges()}", flush=True)

    # Calculate Dijkstra references once for the complete test set.
    references = {}
    dijkstra_times = []
    dijkstra_successes = 0
    dijkstra_costs = []

    print("\nCalculating Dijkstra reference routes...", flush=True)

    for i, row in enumerate(test_df.itertuples(index=False), 1):
        src, dst = int(row.src), int(row.dst)
        start = time.perf_counter()

        try:
            route = nx.shortest_path(
                graph, src, dst, weight="weight", method="dijkstra"
            )
            cost = route_cost(graph, route)
            dijkstra_successes += 1
            dijkstra_costs.append(cost)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            route, cost = None, float("inf")

        dijkstra_times.append((time.perf_counter() - start) * 1000)
        references[(src, dst)] = (route, cost)

        if i % 50 == 0 or i == len(test_df):
            print(f"  Dijkstra: {i}/{len(test_df)}", flush=True)

    results = [{
        "model": "Dijkstra",
        "routes": len(test_df),
        "success_rate_pct": 100 * dijkstra_successes / len(test_df),
        "avg_path_cost": np.mean(dijkstra_costs) if dijkstra_costs else np.nan,
        "avg_cost_ratio": 1.0 if dijkstra_successes else np.nan,
        "avg_runtime_ms": np.mean(dijkstra_times),
        "avg_expansions": np.nan,
    }]

    # Evaluate both trained checkpoints on exactly the same test pairs.
    for model_name, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.exists():
            print(f"\nSkipping missing checkpoint: {checkpoint_path}", flush=True)
            continue

        print(f"\nLoading {model_name}...", flush=True)
        model, embeddings, params = load_model(checkpoint_path, pyg_data)

        successes = 0
        costs = []
        ratios = []
        runtimes = []
        expansions_list = []

        for i, row in enumerate(test_df.itertuples(index=False), 1):
            src, dst = int(row.src), int(row.dst)
            start = time.perf_counter()

            route, cost, expansions = gnn_search(
                graph, model, embeddings, params, src, dst
            )

            runtimes.append((time.perf_counter() - start) * 1000)
            expansions_list.append(expansions)

            optimal_route, optimal_cost = references[(src, dst)]

            if route is not None:
                successes += 1
                costs.append(cost)

                if optimal_route is not None and optimal_cost > 0:
                    ratios.append(cost / optimal_cost)

            if i % 10 == 0 or i == len(test_df):
                print(
                    f"  {model_name}: {i}/{len(test_df)} routes",
                    flush=True,
                )

        results.append({
            "model": model_name,
            "routes": len(test_df),
            "success_rate_pct": 100 * successes / len(test_df),
            "avg_path_cost": np.mean(costs) if costs else np.nan,
            "avg_cost_ratio": np.mean(ratios) if ratios else np.nan,
            "avg_runtime_ms": np.mean(runtimes),
            "avg_expansions": np.mean(expansions_list),
        })

    results_df = pd.DataFrame(results)

    print("\n--- FULL TEST BENCHMARK RESULTS ---", flush=True)
    print(results_df.to_string(index=False), flush=True)

    output = ROOT / "results" / "semi_supervised_test_comparison_450.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output, index=False)

    print(f"\nSaved results to: {output}", flush=True)
    print(
        "Note: This evaluates the held-out test set. "
        "It does not retrain either model.",
        flush=True,
    )


if __name__ == "__main__":
    main()
