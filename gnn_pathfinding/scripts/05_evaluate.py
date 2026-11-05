"""
STEP 5 - Evaluate a prediction file (e.g. from the GNN) with the shared metrics.

    python scripts/05_evaluate.py --dataset roadnet --pred results/roadnet/gnn_20pct_preds.csv --name gnn_20pct
    python scripts/05_evaluate.py --dataset roadnet --pred preds.csv --name gnn_20pct --perturbation edge_removal_10

The prediction CSV needs the columns  src, dst, path  (node ids separated by spaces; empty = no path found)
and optionally  runtime_ms  (inference time per query). Rows are matched to the test pairs by (src, dst).

Metrics are explained in src/metrics.py. Each call appends one row to results/<dataset>/evaluations.csv.
"""
import argparse
from pathlib import Path

import pandas as pd

import _common
from _common import config, die
from src.data_loader import GraphData
from src.metrics import evaluate_predictions

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="roadnet")
ap.add_argument("--pred", required=True, help="CSV with columns src,dst,path[,runtime_ms]")
ap.add_argument("--name", default="model", help="label for this run, e.g. gnn_20pct")
ap.add_argument("--split", default="test", choices=["test", "val"])
ap.add_argument("--perturbation", help="evaluate on a perturbed graph, e.g. edge_removal_10")
ap.add_argument("--peak_memory_mb", type=float, help="optionally report your model's measured peak memory")
a = ap.parse_args()

base = config.processed_dir(a.dataset)
folder = base / "perturbed" / a.perturbation if a.perturbation else base
gpath = folder / "graph.npz"
tpath = (folder / f"{a.split}.csv") if a.perturbation else (base / "splits" / f"{a.split}.csv")
for p in (gpath, tpath, Path(a.pred)):
    if not p.exists():
        die(f"{p} not found.")

G = GraphData.load(gpath).to_networkx()
truth = pd.read_csv(tpath)
preds = pd.read_csv(a.pred)
for c in ("src", "dst", "path"):
    if c not in preds.columns:
        die(f"Prediction file is missing the column '{c}'.")

summary, detail = evaluate_predictions(G, truth, preds)
if a.peak_memory_mb is not None:
    summary["peak_memory_mb"] = a.peak_memory_mb
row = {"dataset": a.dataset, "run": a.name, "perturbation": a.perturbation or "none", "split": a.split, **summary}

out = config.results_dir(a.dataset)
out.mkdir(parents=True, exist_ok=True)
log = out / "evaluations.csv"
(pd.concat([pd.read_csv(log), pd.DataFrame([row])]) if log.exists() else pd.DataFrame([row])).to_csv(log, index=False)
detail.to_csv(out / f"eval_{a.name}_{a.perturbation or 'none'}_{a.split}_per_query.csv", index=False)

print(f"{a.name} on {a.dataset} ({a.perturbation or 'original graph'}, {a.split})")
for k, v in summary.items():
    print(f"  {k:<20} {v:.4f}" if isinstance(v, float) else f"  {k:<20} {v}")
print(f"Appended to {log}")
