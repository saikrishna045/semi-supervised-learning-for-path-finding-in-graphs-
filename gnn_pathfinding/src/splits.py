"""
splits.py  --  reproducible labeled / unlabeled / validation / test splits.

Pairs are shuffled with a fixed seed, then:   test | validation | train
The labeled sets are NESTED:  5% labeled  is a subset of  10%  is a subset of  20%
(fractions are shares of the TRAIN pairs). All remaining train pairs are "unlabeled":
we keep only (src, dst) and drop the path/cost.

Files written to  data/processed/<dataset>/splits/
    pairs_all_truth.csv         everything with ground truth (for evaluation scripts only!)
    test.csv, val.csv           src,dst,cost,hops,path
    labeled_5pct.csv ...        src,dst,cost,hops,path   <- supervised loss
    unlabeled_5pct.csv ...      src,dst                  <- unlabeled loss (the rest of train)
    split_meta.json             seed, counts
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_splits(pairs: pd.DataFrame, label_fracs=(0.05, 0.10, 0.20), test_frac=0.15,
                val_frac=0.10, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pairs.iloc[rng.permutation(len(pairs))].reset_index(drop=True)
    n = len(df)
    n_test, n_val = int(round(test_frac * n)), int(round(val_frac * n))
    df["split"] = "train"
    df.loc[: n_test - 1, "split"] = "test"
    df.loc[n_test: n_test + n_val - 1, "split"] = "val"
    train_idx = df.index[df["split"] == "train"]
    for f in label_fracs:
        k = int(round(f * len(train_idx)))
        col = f"labeled_{int(round(f * 100))}pct"
        df[col] = False
        df.loc[train_idx[:k], col] = True
    return df


def save_splits(df: pd.DataFrame, out_dir, label_fracs=(0.05, 0.10, 0.20), seed=42) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth_cols = ["src", "dst", "cost", "hops", "path"]
    df.to_csv(out / "pairs_all_truth.csv", index=False)
    df[df.split == "test"][truth_cols].to_csv(out / "test.csv", index=False)
    df[df.split == "val"][truth_cols].to_csv(out / "val.csv", index=False)
    meta = {"seed": seed, "n_pairs": int(len(df)),
            "n_test": int((df.split == "test").sum()), "n_val": int((df.split == "val").sum()),
            "n_train": int((df.split == "train").sum()), "label_fractions": {}}
    train = df[df.split == "train"]
    for f in label_fracs:
        pct = int(round(f * 100))
        col = f"labeled_{pct}pct"
        train[train[col]][truth_cols].to_csv(out / f"labeled_{pct}pct.csv", index=False)
        train[~train[col]][["src", "dst"]].to_csv(out / f"unlabeled_{pct}pct.csv", index=False)
        meta["label_fractions"][f"{pct}pct"] = {"labeled": int(train[col].sum()),
                                                "unlabeled": int((~train[col]).sum())}
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
