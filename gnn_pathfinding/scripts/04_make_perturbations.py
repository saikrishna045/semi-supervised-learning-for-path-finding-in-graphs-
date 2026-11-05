"""
STEP 4 - Incomplete and evolving graph versions for robustness tests.

    python scripts/04_make_perturbations.py --dataset roadnet

Versions created (each in data/processed/<dataset>/perturbed/<version>/):
    edge_removal_10, edge_removal_20     10% / 20% of the edges removed (graph kept connected)
    node_removal_5                       5% of the nodes lose all their edges
    weight_drift_t1, _t2, _t3            edge weights drift and some roads close, accumulating over time

Each folder holds:  graph.npz (features recomputed on the perturbed graph), pairs_truth.csv (ALL pairs with
NEW optimal paths; unreachable pairs dropped), test.csv, val.csv, info.json.

It also reports the "stale path" baseline: reuse the OLD optimal path on the NEW graph. Its success rate
shows how much the perturbation actually breaks precomputed answers, which is what a robust model must beat.
Output summary: results/<dataset>/perturbation_summary.csv
"""
import argparse
import json

import pandas as pd

import _common
from _common import config, die
from src.data_loader import GraphData
from src.metrics import evaluate_predictions
from src.perturbations import evolve_weights, recompute_truth, remove_random_edges, remove_random_nodes

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="roadnet")
ap.add_argument("--seed", type=int, default=config.SEED)
ap.add_argument("--edge_fracs", type=float, nargs="*", default=[0.10, 0.20])
ap.add_argument("--node_frac", type=float, default=0.05)
ap.add_argument("--drift_steps", type=int, default=3)
ap.add_argument("--allow_disconnect", action="store_true",
                help="edge removal may disconnect the graph (pairs that become unreachable are dropped)")
a = ap.parse_args()

g, G, info = _common.load_processed(a.dataset)
d = config.processed_dir(a.dataset)
truth_file = d / "splits" / "pairs_all_truth.csv"
if not truth_file.exists():
    die(f"{truth_file} not found. Run scripts/02_make_labels_splits.py first.")
orig = pd.read_csv(truth_file)
lap_dim = sum(n.startswith("lap_pe_") for n in g.feature_names)

versions = []
for f in a.edge_fracs:
    H, i = remove_random_edges(G, f, a.seed, keep_connected=not a.allow_disconnect)
    versions.append((f"edge_removal_{int(round(f * 100))}", H, i))
H, i = remove_random_nodes(G, a.node_frac, a.seed)
versions.append((f"node_removal_{int(round(a.node_frac * 100))}", H, i))
for H, i in evolve_weights(G, a.drift_steps, a.seed):
    versions.append((f"weight_drift_t{i['step']}", H, i))

summary_rows = []
for name, H, pinfo in versions:
    new_truth = recompute_truth(H, orig)
    keep = orig.merge(new_truth[["src", "dst"]], on=["src", "dst"])
    test_new = new_truth[new_truth["split"] == "test"]

    # stale-path baseline: old optimal path evaluated on the new graph
    stale = orig[orig["split"] == "test"][["src", "dst", "path"]]
    stale_sum, _ = evaluate_predictions(H, test_new, stale)

    old_cost = orig.set_index(["src", "dst"])["cost"]
    new_cost = new_truth.set_index(["src", "dst"])["cost"]
    ratio = (new_cost / old_cost.loc[new_cost.index]).mean()

    out = d / "perturbed" / name
    out.mkdir(parents=True, exist_ok=True)
    GraphData.from_networkx(H, name, coords=g.coords, lap_pe_dim=lap_dim,
                            meta={"base": a.dataset, **{k: v for k, v in pinfo.items() if k != "dropped_nodes"}}).save(out / "graph.npz")
    new_truth.to_csv(out / "pairs_truth.csv", index=False)
    cols = ["src", "dst", "cost", "hops", "path"]
    test_new[cols].to_csv(out / "test.csv", index=False)
    new_truth[new_truth["split"] == "val"][cols].to_csv(out / "val.csv", index=False)

    row = {"version": name, "edges": H.number_of_edges(), "edges_original": G.number_of_edges(),
           "pairs_kept": f"{len(keep)}/{len(orig)}", "test_pairs_kept": len(test_new),
           "optimal_cost_change": float(ratio),
           "stale_path_success": stale_sum["success_rate"], "stale_path_accuracy": stale_sum["path_accuracy"],
           "stale_path_mean_cost_ratio": stale_sum["mean_cost_ratio"]}
    summary_rows.append(row)
    (out / "info.json").write_text(json.dumps({**pinfo, **row}, indent=2, default=str))
    print(f"{name:<16} edges {row['edges']}/{row['edges_original']}  pairs kept {row['pairs_kept']}  "
          f"opt-cost x{ratio:.3f}  stale-path success {stale_sum['success_rate']:.3f}  "
          f"stale-path optimal {stale_sum['path_accuracy']:.3f}")

res = config.results_dir(a.dataset)
res.mkdir(parents=True, exist_ok=True)
pd.DataFrame(summary_rows).to_csv(res / "perturbation_summary.csv", index=False)
print(f"\nSaved perturbed graphs to {d / 'perturbed'}  and summary to {res / 'perturbation_summary.csv'}")
