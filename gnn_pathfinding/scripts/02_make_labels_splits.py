"""
STEP 2 - Generate source/destination pairs, label them with exact shortest paths (Dijkstra),
         and create the reproducible labeled / unlabeled / validation / test splits.

    python scripts/02_make_labels_splits.py --dataset roadnet
    python scripts/02_make_labels_splits.py --dataset movingai_arena --pairs 2000

For MovingAI maps, pairs come from the .scen file when one was found in step 1
(use --no_scen to generate random pairs instead).

Output: data/processed/<dataset>/splits/   (see src/splits.py for the file list)
"""
import argparse
import json

import numpy as np
import pandas as pd

import _common
from _common import config
from src.data_loader import read_scen
from src.perturbations import recompute_truth
from src.shortest_paths import build_csr, generate_pairs
from src.splits import make_splits, save_splits

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="roadnet")
ap.add_argument("--pairs", type=int, default=config.N_PAIRS)
ap.add_argument("--min_hops", type=int, default=config.MIN_HOPS)
ap.add_argument("--seed", type=int, default=config.SEED)
ap.add_argument("--no_scen", action="store_true")
a = ap.parse_args()

g, G, info = _common.load_processed(a.dataset)
d = config.processed_dir(a.dataset)
fracs = config.LABEL_FRACTIONS

pairs = None
scen_file = info.get("scen_file")
if info.get("kind") == "movingai" and scen_file and not a.no_scen and (d / "cell_to_node.npy").exists():
    c2n = np.load(d / "cell_to_node.npy")
    rows = []
    for sx, sy, gx, gy, opt in read_scen(scen_file):
        if 0 <= sy < c2n.shape[0] and 0 <= sx < c2n.shape[1] and 0 <= gy < c2n.shape[0] and 0 <= gx < c2n.shape[1]:
            s, t = int(c2n[sy, sx]), int(c2n[gy, gx])
            if s >= 0 and t >= 0 and s != t:
                rows.append(dict(src=s, dst=t, cost=np.nan, hops=0, path="", scen_optimal=opt))
    if rows:
        df = pd.DataFrame(rows).drop_duplicates(["src", "dst"])
        pairs = recompute_truth(G, df)
        diff = (pairs["cost"] - pairs["scen_optimal"]).abs()
        print(f"Scenario pairs: {len(pairs)}; cost differs from the .scen optimal on "
              f"{int((diff > 1e-3 * np.maximum(1, pairs['scen_optimal'])).sum())} pairs (max |diff| = {diff.max():.4f})")
        pairs = pairs.drop(columns=["scen_optimal"])
        pairs = pairs[pairs["hops"] >= a.min_hops]
        if len(pairs) > a.pairs:
            pairs = pairs.sample(n=a.pairs, random_state=a.seed)
        pairs = pairs.reset_index(drop=True)

if pairs is None or len(pairs) == 0:
    print("Generating random pairs with Dijkstra labels ...")
    pairs = generate_pairs(build_csr(G), a.pairs, a.min_hops, a.seed)

df = make_splits(pairs, fracs, config.TEST_FRAC, config.VAL_FRAC, a.seed)
meta = save_splits(df, d / "splits", fracs, a.seed)

print(f"\n{len(df)} pairs  | mean optimal cost {df['cost'].mean():.2f} | mean hops {df['hops'].mean():.1f}")
print(json.dumps(meta, indent=2))
print(f"Saved to {d / 'splits'}")
print(f"Next:  python scripts/03_run_baselines.py --dataset {a.dataset}")
