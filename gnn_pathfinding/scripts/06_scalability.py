"""
STEP 6 - Scalability: graph size vs. query time, nodes expanded and memory (+ plot).

    python scripts/06_scalability.py --sizes 1000 2000 5000 10000 20000

Samples connected subgraphs of increasing size from the FULL RoadNet file (so it needs the raw file),
runs each solver on random queries and saves results/<dataset>/scalability.csv and scalability.png.

To add the GNN to the same plot, register it (see the docstring in src/scalability.py) and add its
name to --solvers.
"""
import argparse
from pathlib import Path

import _common
from _common import config, die
from src.data_loader import find_roadnet_file, load_edge_list
from src.scalability import plot_scalability, run_scalability

# --- Person B: register the GNN here, then run with  --solvers dijkstra astar_alt gnn ---------------
# from src.scalability import register_solver
# register_solver("gnn", lambda ctx: my_gnn_solver(ctx))
# ------------------------------------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="roadnet", help="results folder name")
ap.add_argument("--input", help="RoadNet edge-list file (default: first file in data/raw/roadnet)")
ap.add_argument("--sizes", type=int, nargs="*", default=[1000, 2000, 5000, 10000, 20000])
ap.add_argument("--queries", type=int, default=50)
ap.add_argument("--solvers", nargs="*", default=["dijkstra", "astar_alt"])
ap.add_argument("--seed", type=int, default=config.SEED)
a = ap.parse_args()

path = Path(a.input) if a.input else find_roadnet_file(config.RAW_ROADNET)
if path is None or not path.exists():
    die(f"No RoadNet file found in {config.RAW_ROADNET}. Put roadNet-PA.txt(.gz) there or pass --input.")

print(f"Loading {path} ...")
G = load_edge_list(path)
print(f"  {G.number_of_nodes():,} nodes. Running solvers {a.solvers} on sizes {a.sizes}")
df = run_scalability(G, a.sizes, n_queries=a.queries, seed=a.seed, weight_range=config.EDGE_WEIGHT_RANGE,
                     solvers=tuple(a.solvers), n_landmarks=config.N_LANDMARKS)
if df.empty:
    die("No sizes could be run (all larger than the graph).")

out = config.results_dir(a.dataset)
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "scalability.csv", index=False)
plot_scalability(df, out / "scalability.png", title=f"Scalability on {a.dataset} (random queries, {a.queries} per size)")
print(f"\nSaved {out / 'scalability.csv'} and {out / 'scalability.png'}")
