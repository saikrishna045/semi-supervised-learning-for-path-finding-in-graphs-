"""
STEP 3 - Classical baselines (Dijkstra and A*) on the test (or val) pairs.

    python scripts/03_run_baselines.py --dataset roadnet

A* variants: astar_alt always (landmark heuristic, no coordinates needed);
astar_euclidean / astar_octile too when the graph has coordinates (MovingAI).

Output: results/<dataset>/baselines_summary.csv and baselines_per_query.csv
"""
import argparse

import pandas as pd

import _common
from _common import config, die
from src.metrics import evaluate_predictions, peak_memory_mb
from src.shortest_paths import GraphContext, run_queries

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="roadnet")
ap.add_argument("--split", default="test", choices=["test", "val"])
ap.add_argument("--algos", nargs="*", help="default: all available (dijkstra astar_alt [astar_euclidean astar_octile])")
ap.add_argument("--mem_queries", type=int, default=20, help="queries used to measure peak memory")
a = ap.parse_args()

g, G, info = _common.load_processed(a.dataset)
split_file = config.processed_dir(a.dataset) / "splits" / f"{a.split}.csv"
if not split_file.exists():
    die(f"{split_file} not found. Run scripts/02_make_labels_splits.py first.")
truth = pd.read_csv(split_file)

ctx = GraphContext(G, coords=g.coords, n_landmarks=config.N_LANDMARKS, seed=config.SEED)
algos = a.algos or ctx.available()
rows, per_query = [], []
for name in algos:
    solver = ctx.solver(name)
    preds = run_queries(solver, truth, name)
    summary, _ = evaluate_predictions(G, truth, preds)
    pairs = list(zip(truth["src"].tolist(), truth["dst"].tolist()))[: a.mem_queries]
    summary["peak_memory_mb"] = peak_memory_mb(lambda: [solver(s, t) for s, t in pairs])
    summary["mean_nodes_expanded"] = float(preds["expanded"].mean())
    summary["preprocess_s"] = ctx.alt_preprocess_s if name == "astar_alt" else 0.0
    rows.append({"dataset": a.dataset, "split": a.split, "algo": name, **summary})
    per_query.append(preds)
    print(f"{name:<16} success={summary['success_rate']:.3f}  accuracy={summary['path_accuracy']:.3f}  "
          f"ratio={summary['mean_cost_ratio']:.4f}  {summary['mean_runtime_ms']:.3f} ms/query  "
          f"expanded={summary['mean_nodes_expanded']:.0f}  mem={summary['peak_memory_mb']:.2f} MB")

out = config.results_dir(a.dataset)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "baselines_summary.csv", index=False)
pd.concat(per_query).to_csv(out / "baselines_per_query.csv", index=False)
print(f"\nSaved to {out}")
print(f"Next:  python scripts/04_make_perturbations.py --dataset {a.dataset}")
