"""
metrics.py  --  evaluation of predicted paths (used for baselines AND for the GNN).

Predictions are a DataFrame / CSV with columns:
    src, dst, path            path = node ids separated by spaces, e.g. "4 17 9 12" ("" = no path found)
    runtime_ms (optional)     per-query inference time

Metrics
    success_rate          predicted path is VALID: starts at src, ends at dst, every hop is a real edge
    path_accuracy         valid AND its cost equals the optimal cost (ties count as correct)
    exact_match_rate      identical node sequence to the stored optimal path (stricter; ties may differ)
    near_optimal_rate     valid AND cost <= 1.05 x optimal
    mean/median/p95_cost_ratio   predicted cost / optimal cost over valid paths (1.0 = optimal)
    mean_runtime_ms       average query time (if runtime_ms column is given)
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd

INF = float("inf")


def parse_path(s) -> list:
    if isinstance(s, str) and s.strip():
        return [int(tok) for tok in s.split()]
    return []


def validate_and_cost(G, path, s: int, t: int):
    """(is_valid, cost). Cost is inf for invalid paths."""
    if not path or path[0] != s or path[-1] != t:
        return False, INF
    cost = 0.0
    for a, b in zip(path[:-1], path[1:]):
        if not G.has_edge(a, b):
            return False, INF
        cost += G[a][b].get("weight", 1.0)
    return True, cost


def evaluate_predictions(G, truth: pd.DataFrame, preds: pd.DataFrame, tol: float = 1e-6):
    """Compare predictions with ground truth on graph G.

    truth must have: src, dst, cost, path.   Missing predictions count as failures.
    Returns (summary dict, per-query DataFrame).
    """
    cols = ["src", "dst", "path"] + (["runtime_ms"] if "runtime_ms" in preds.columns else [])
    p = preds[cols].rename(columns={"path": "pred_path"}).drop_duplicates(["src", "dst"])
    m = truth[["src", "dst", "cost", "path"]].merge(p, on=["src", "dst"], how="left")

    valid, pcost, ratio, exact = [], [], [], []
    for r in m.itertuples(index=False):
        path = parse_path(r.pred_path)
        ok, c = validate_and_cost(G, path, int(r.src), int(r.dst))
        valid.append(ok)
        pcost.append(c)
        ratio.append(c / r.cost if ok and r.cost > 0 else (1.0 if ok else np.nan))
        exact.append(isinstance(r.pred_path, str) and r.pred_path.strip() == str(r.path).strip())
    m["valid"], m["pred_cost"], m["cost_ratio"], m["exact"] = valid, pcost, ratio, exact
    m["optimal"] = m["valid"] & (m["pred_cost"] <= m["cost"] * (1 + tol) + tol)
    m["near_optimal"] = m["valid"] & (m["pred_cost"] <= m["cost"] * 1.05 + tol)

    r_ok = m.loc[m["valid"], "cost_ratio"]
    summary = {
        "n_queries": int(len(m)),
        "success_rate": float(m["valid"].mean()),
        "path_accuracy": float(m["optimal"].mean()),
        "exact_match_rate": float(m["exact"].mean()),
        "near_optimal_rate": float(m["near_optimal"].mean()),
        "mean_cost_ratio": float(r_ok.mean()) if len(r_ok) else float("nan"),
        "median_cost_ratio": float(r_ok.median()) if len(r_ok) else float("nan"),
        "p95_cost_ratio": float(r_ok.quantile(0.95)) if len(r_ok) else float("nan"),
    }
    if "runtime_ms" in m.columns:
        summary["mean_runtime_ms"] = float(m["runtime_ms"].mean())
    return summary, m


# ---------------------------------------------------------------------------
# Runtime / memory helpers
# ---------------------------------------------------------------------------
def mean_runtime_ms(fn, repeat: int = 1) -> float:
    """Average wall-clock time of fn() in milliseconds."""
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) * 1e3 / repeat


def peak_memory_mb(fn) -> float:
    """Peak Python memory allocated while fn() runs (search state, heaps, dicts).
    Measured separately from runtime because tracemalloc slows code down."""
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024 * 1024)
