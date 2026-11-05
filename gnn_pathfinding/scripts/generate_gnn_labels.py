import csv
import random
from pathlib import Path
import networkx as nx

graph = nx.Graph()

edges = [
    (0, 1, 1.0),
    (1, 2, 1.0),
    (2, 3, 1.0),
    (3, 4, 1.0),
    (4, 5, 1.0),
    (0, 2, 3.0),
    (1, 4, 4.0),
    (2, 5, 5.0),
]

graph.add_weighted_edges_from(edges, weight="weight")
random.seed(42)
nodes = list(graph.nodes())
examples = []

for _ in range(20):
    source, destination = random.sample(nodes, 2)
    path = nx.shortest_path(
        graph,
        source=source,
        target=destination,
        weight="weight"
    )
    correct_next_hop = path[1]
    path_cost = nx.shortest_path_length(
        graph,
        source=source,
        target=destination,
        weight="weight"
    )

    for candidate in graph.neighbors(source):
        examples.append({
            "source": source,
            "destination": destination,
            "candidate": candidate,
            "label": int(candidate == correct_next_hop),
            "path_cost": path_cost,
        })

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "gnn_training_labels.csv"

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "source",
            "destination",
            "candidate",
            "label",
            "path_cost",
        ],
    )
    writer.writeheader()
    writer.writerows(examples)

print("Graph nodes:", graph.number_of_nodes())
print("Graph edges:", graph.number_of_edges())
print("Training examples:", len(examples))
print("Labels saved to:", output_file)
