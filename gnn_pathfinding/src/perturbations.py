"""
perturbations.py  --  incomplete and evolving graph versions for robustness tests.

  remove_random_edges   missing roads / links          (incomplete graph)
  remove_random_nodes   missing intersections          (all edges of the node are cut; ids are kept)
  evolve_weights        traffic changes over time      (list of snapshots t=1..T: weights drift + closures)

Node ids never change, so a trained model's node features / predictions stay comparable.
After a perturbation the optimal paths change, so use recompute_truth() to get new labels.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from .shortest_paths import build_csr, reconstruct_path


def _remove_edges(H: nx.Graph, order, k: int, keep_connected: bool) -> int:
    """Remove up to k edges from H following `order`. Returns how many were removed."""
    removed = 0
    for (u, v) in order:
        if removed >= k:
            break
        if not H.has_edge(u, v):
            continue
        if not keep_connected:
            H.remove_edge(u, v)
            removed += 1
            continue
        if H.degree(u) <= 1 or H.degree(v) <= 1:
            continue
        data = dict(H[u][v])
        H.remove_edge(u, v)
        if nx.has_path(H, u, v):
            removed += 1
        else:
            H.add_edge(u, v, **data)      # that edge was a bridge, put it back
    return removed


def remove_random_edges(G: nx.Graph, frac: float, seed: int = 0, keep_connected: bool = True):
    """Remove `frac` of the edges. keep_connected=True skips edges whose removal would disconnect
    the graph (road graphs are almost trees, so very large fractions may not be reachable)."""
    rng = np.random.default_rng(seed)
    edges = list(G.edges)
    k = int(round(frac * len(edges)))
    order = [edges[i] for i in rng.permutation(len(edges))]
    H = G.copy()
    removed = _remove_edges(H, order, k, keep_connected)
    return H, {"requested_edges": k, "removed_edges": removed}


def remove_random_nodes(G: nx.Graph, frac: float, seed: int = 0):
    """Cut every edge of a random `frac` of the nodes (nodes stay as isolated ids)."""
    rng = np.random.default_rng(seed)
    nodes = np.array(sorted(G.nodes))
    k = int(round(frac * len(nodes)))
    dropped = rng.choice(nodes, size=k, replace=False).tolist()
    H = G.copy()
    for u in dropped:
        H.remove_edges_from(list(H.edges(u)))
    return H, {"dropped_nodes": sorted(int(d) for d in dropped)}


def evolve_weights(G: nx.Graph, n_steps: int = 3, seed: int = 0, change_frac: float = 0.15,
                   drift: float = 0.5, closure_frac: float = 0.01, min_weight: float = 0.5):
    """Simulate traffic over time. At each step a random `change_frac` of edges gets its weight
    multiplied by U(1-drift, 1+drift), and `closure_frac` of edges are closed (graph stays
    connected). Returns a list of (graph_t, info) for t = 1..n_steps (changes accumulate)."""
    rng = np.random.default_rng(seed)
    snapshots, H = [], G.copy()
    for t in range(1, n_steps + 1):
        H = H.copy()
        edges = list(H.edges)
        n_change = int(round(change_frac * len(edges)))
        for i in rng.choice(len(edges), size=n_change, replace=False):
            u, v = edges[i]
            H[u][v]["weight"] = float(max(min_weight, H[u][v]["weight"] * rng.uniform(1 - drift, 1 + drift)))
        k = int(round(closure_frac * len(edges)))
        order = [edges[i] for i in rng.permutation(len(edges))]
        closed = _remove_edges(H, order, k, keep_connected=True)
        snapshots.append((H, {"step": t, "reweighted_edges": n_change, "closed_edges": closed}))
    return snapshots


def recompute_truth(G: nx.Graph, pairs: pd.DataFrame) -> pd.DataFrame:
    """Exact shortest paths on (perturbed) G for the given pairs. Pairs that became unreachable
    are dropped. Extra columns (e.g. 'split') are kept."""
    csr = build_csr(G)
    rows = []
    for s, grp in pairs.groupby("src"):
        dist, pred = sp_dijkstra(csr, directed=False, indices=int(s), return_predecessors=True)
        for rec in grp.to_dict("records"):
            t = int(rec["dst"])
            if not np.isfinite(dist[t]):
                continue
            path = reconstruct_path(pred, int(s), t)
            if path is None:
                continue
            rec.update(cost=float(dist[t]), hops=len(path) - 1, path=" ".join(map(str, path)))
            rows.append(rec)
    cols = list(pairs.columns) + [c for c in ("cost", "hops", "path") if c not in pairs.columns]
    return pd.DataFrame(rows, columns=cols)
