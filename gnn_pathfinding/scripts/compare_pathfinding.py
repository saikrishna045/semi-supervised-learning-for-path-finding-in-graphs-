import sys
import time
from pathlib import Path

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gnn_model import GraphSAGEPathModel

MODEL_PATH = ROOT / "models" / "graphsage_path_model.pt"
RESULTS_DIR = ROOT / "results"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run python scripts/train_gnn.py first."
    )

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
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
model.eval()
embeddings = model(data)

def gnn_path(source, destination, max_steps=20):
    path = [source]
    current = source
    visited = {source}

    with torch.no_grad():
        for _ in range(max_steps):
            if current == destination:
                return path

            neighbors = list(graph.neighbors(current))
            if not neighbors:
                return None

            candidates = torch.tensor(neighbors, dtype=torch.long)
            scores = model.score_candidates(
                embeddings, source, destination, candidates
            )
            next_node = int(candidates[torch.argmax(scores)].item())

            if next_node in visited:
                return None

            path.append(next_node)
            visited.add(next_node)
            current = next_node

    return path if current == destination else None

def calculate_cost(path):
    if path is None:
        return None
    return sum(
        graph[path[i]][path[i + 1]]["weight"]
        for i in range(len(path) - 1)
    )

test_pairs = [
    (0, 5),
    (5, 0),
    (0, 4),
    (1, 5),
    (4, 0),
    (2, 5),
    (3, 0),
    (1, 3),
]

results = []

for source, destination in test_pairs:
    optimal_start = time.perf_counter()
    dijkstra_path = nx.dijkstra_path(
        graph, source, destination, weight="weight"
    )
    dijkstra_time = time.perf_counter() - optimal_start
    optimal_cost = calculate_cost(dijkstra_path)

    astar_start = time.perf_counter()
    astar_path = nx.astar_path(
        graph,
        source,
        destination,
        heuristic=lambda u, v: 0,
        weight="weight",
    )
    astar_time = time.perf_counter() - astar_start
    astar_cost = calculate_cost(astar_path)

    gnn_start = time.perf_counter()
    predicted_path = gnn_path(source, destination)
    gnn_time = time.perf_counter() - gnn_start
    gnn_cost = calculate_cost(predicted_path)

    results.append({
        "source": source,
        "destination": destination,
        "algorithm": "GNN",
        "success": predicted_path is not None,
        "path": str(predicted_path),
        "path_cost": gnn_cost,
        "optimal_cost": optimal_cost,
        "optimality_gap": (
            (gnn_cost - optimal_cost) / optimal_cost * 100
            if gnn_cost is not None and optimal_cost > 0
            else None
        ),
        "runtime_ms": gnn_time * 1000,
    })

    results.append({
        "source": source,
        "destination": destination,
        "algorithm": "Dijkstra",
        "success": True,
        "path": str(dijkstra_path),
        "path_cost": dijkstra_cost if (dijkstra_cost := optimal_cost) else 0,
        "optimal_cost": optimal_cost,
        "optimality_gap": 0.0,
        "runtime_ms": dijkstra_time * 1000,
    })

    results.append({
        "source": source,
        "destination": destination,
        "algorithm": "A* (zero heuristic)",
        "success": True,
        "path": str(astar_path),
        "path_cost": astar_cost,
        "optimal_cost": optimal_cost,
        "optimality_gap": (
            (astar_cost - optimal_cost) / optimal_cost * 100
            if optimal_cost > 0 else 0.0
        ),
        "runtime_ms": astar_time * 1000,
    })

    print(f"\nRoute: {source} -> {destination}")
    print(f"  Dijkstra: {dijkstra_path}, cost={optimal_cost:.2f}")
    print(f"  A*:       {astar_path}, cost={astar_cost:.2f}")
    print(f"  GNN:      {predicted_path}, cost={gnn_cost}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
output_file = RESULTS_DIR / "pathfinding_comparison.csv"
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)

print("\n--- Summary ---")
print(df.groupby("algorithm").agg(
    success_rate=("success", "mean"),
    average_runtime_ms=("runtime_ms", "mean"),
    average_path_cost=("path_cost", "mean"),
).to_string())

print("\nResults saved to:", output_file)
print("Note: These are preliminary results on a six-node synthetic graph.")
