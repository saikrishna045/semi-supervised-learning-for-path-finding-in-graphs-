"""
scalability.py  --  graph size vs. query time / expanded nodes / memory, plus the plot.

Person B: to put the GNN in the same plot, register it before running:

    from src.scalability import register_solver
    def gnn_factory(ctx):                  # ctx has .G (networkx), .adj, .csr, .n, .coords
        model = ...                        # build / load / run inference for this graph
        return lambda s, t: (predicted_path_list_or_None, predicted_cost, 0)
    register_solver("gnn", gnn_factory)

Then:  python scripts/06_scalability.py --solvers dijkstra astar_alt gnn
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data_loader import add_edge_weights, clean_graph, sample_connected_subgraph
from .metrics import peak_memory_mb
from .shortest_paths import GraphContext

EXTRA_SOLVERS = {}

# validated categorical palette slots 1-3 (blue, orange, aqua) and chart chrome (light surface)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def register_solver(name: str, factory) -> None:
    """factory(ctx: GraphContext) -> callable(s, t) -> (path | None, cost, nodes_expanded)"""
    EXTRA_SOLVERS[name] = factory


def run_scalability(base_G, sizes, n_queries: int = 50, seed: int = 42, weight_range=(1, 5),
                    solvers=("dijkstra", "astar_alt"), mem_queries: int = 10, n_landmarks: int = 4):
    rows = []
    for size in sizes:
        if size > base_G.number_of_nodes():
            print(f"[skip] size {size} > graph size {base_G.number_of_nodes()}")
            continue
        sub, _ = clean_graph(sample_connected_subgraph(base_G, size, seed))
        add_edge_weights(sub, seed, *weight_range)
        ctx = GraphContext(sub, n_landmarks=n_landmarks, seed=seed, build_alt="astar_alt" in solvers)
        rng = np.random.default_rng(seed)
        n = sub.number_of_nodes()
        queries = []
        while len(queries) < n_queries:
            s, t = rng.integers(n, size=2).tolist()
            if s != t:
                queries.append((s, t))
        for name in solvers:
            fn = EXTRA_SOLVERS[name](ctx) if name in EXTRA_SOLVERS else ctx.solver(name)
            fn(*queries[0])                                         # warm-up
            times, expanded = [], []
            for s, t in queries:
                t0 = time.perf_counter()
                _, _, e = fn(s, t)
                times.append((time.perf_counter() - t0) * 1e3)
                expanded.append(e)
            mem = peak_memory_mb(lambda: [fn(s, t) for s, t in queries[:mem_queries]])
            rows.append(dict(
                nodes=n, edges=sub.number_of_edges(), solver=name,
                mean_query_ms=float(np.mean(times)), median_query_ms=float(np.median(times)),
                mean_expanded=float(np.mean(expanded)), peak_memory_mb=float(mem),
                preprocess_s=ctx.alt_preprocess_s if name == "astar_alt" else 0.0,
            ))
            print(f"  n={n:>7}  {name:<16} {np.mean(times):9.3f} ms/query  "
                  f"expanded={np.mean(expanded):9.0f}  peakmem={mem:6.2f} MB")
    return pd.DataFrame(rows)


def plot_scalability(df: pd.DataFrame, out_png, title: str = "Scalability: graph size vs. cost") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solvers = list(dict.fromkeys(df["solver"]))
    panels = [("mean_query_ms", "Mean query time (ms)"),
              ("mean_expanded", "Nodes expanded per query"),
              ("peak_memory_mb", "Peak search memory (MB)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax, (col, label) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for i, name in enumerate(solvers):
            d = df[df["solver"] == name].sort_values("nodes")
            color = SERIES[i % len(SERIES)]
            ax.plot(d["nodes"], d[col], color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=name)
            if len(d):   # direct label at the line end, in ink (not the series color)
                ax.annotate(name, (d["nodes"].iloc[-1], d[col].iloc[-1]), xytext=(6, 0),
                            textcoords="offset points", fontsize=9, color=INK2, va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Graph size (nodes)", color=INK2)
        ax.set_title(label, loc="left", fontsize=11, color=INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)
        ax.margins(x=0.18)
    if len(solvers) >= 2:
        axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
