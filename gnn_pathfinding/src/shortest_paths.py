"""
shortest_paths.py  --  ground-truth path generation + classical baselines (Dijkstra, A*).

A* heuristics:
  * astar_alt        landmark (ALT) heuristic - works on ANY graph, no coordinates needed
                     (use this for RoadNet, which has no coordinates)
  * astar_euclidean  needs node coordinates (MovingAI grids)
  * astar_octile     needs node coordinates, tight for 8-connected grids (MovingAI)
All heuristics here are admissible, so A* returns the same optimal cost as Dijkstra.
"""
from __future__ import annotations

import heapq
import math
import time
from typing import Callable, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

INF = float("inf")


# ---------------------------------------------------------------------------
# Graph structures
# ---------------------------------------------------------------------------
def build_adj(G: nx.Graph):
    """Adjacency list: adj[u] = [(v, weight), ...]. Nodes must be 0..N-1."""
    n = G.number_of_nodes()
    return [[(v, float(d.get("weight", 1.0))) for v, d in G[u].items()] for u in range(n)]


def build_csr(G: nx.Graph):
    n = G.number_of_nodes()
    return nx.to_scipy_sparse_array(G, nodelist=range(n), weight="weight", format="csr")


def reconstruct_path(pred, s: int, t: int):
    path = [t]
    while path[-1] != s:
        p = int(pred[path[-1]])
        if p < 0 or len(path) > len(pred):
            return None
        path.append(p)
    return path[::-1]


def _trace(parent, t):
    path = [t]
    while parent[path[-1]] != -1:
        path.append(parent[path[-1]])
    return path[::-1]


# ---------------------------------------------------------------------------
# Search algorithms.  Each returns (path or None, cost, nodes_expanded)
# ---------------------------------------------------------------------------
def dijkstra(adj, s: int, t: int):
    dist = {s: 0.0}
    parent = {s: -1}
    heap = [(0.0, s)]
    expanded = 0
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, INF):
            continue
        expanded += 1
        if u == t:
            return _trace(parent, t), d, expanded
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return None, INF, expanded


def astar(adj, s: int, t: int, h: Callable[[int, int], float]):
    g = {s: 0.0}
    parent = {s: -1}
    heap = [(h(s, t), 0.0, s)]
    expanded = 0
    while heap:
        f, gu, u = heapq.heappop(heap)
        if gu > g.get(u, INF):
            continue
        expanded += 1
        if u == t:
            return _trace(parent, t), gu, expanded
        for v, w in adj[u]:
            ng = gu + w
            if ng < g.get(v, INF):
                g[v] = ng
                parent[v] = u
                heapq.heappush(heap, (ng + h(v, t), ng, v))
    return None, INF, expanded


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def select_landmarks(csr, k: int = 4, seed: int = 0):
    """Farthest-point landmarks. Returns (landmark ids, D of shape (k, N))."""
    n = csr.shape[0]
    rng = np.random.default_rng(seed)
    d0 = sp_dijkstra(csr, directed=False, indices=int(rng.integers(n)))
    lms = [int(np.argmax(np.where(np.isfinite(d0), d0, -1)))]
    rows = [sp_dijkstra(csr, directed=False, indices=lms[0])]
    mind = rows[0].copy()
    for _ in range(k - 1):
        nxt = int(np.argmax(np.where(np.isfinite(mind), mind, -1)))
        lms.append(nxt)
        rows.append(sp_dijkstra(csr, directed=False, indices=nxt))
        mind = np.minimum(mind, rows[-1])
    D = np.vstack(rows)
    return lms, np.where(np.isfinite(D), D, 0.0)


def alt_heuristic(D: np.ndarray):
    per_node = D.T.tolist()

    def h(n, t):
        a, b = per_node[n], per_node[t]
        return max(abs(x - y) for x, y in zip(a, b))
    return h


def coord_heuristic(coords: np.ndarray, kind: str):
    cx, cy = coords[:, 0].astype(float).tolist(), coords[:, 1].astype(float).tolist()
    r2 = math.sqrt(2.0)
    if kind == "euclidean":
        return lambda n, t: math.hypot(cx[n] - cx[t], cy[n] - cy[t])
    if kind == "octile":
        def h(n, t):
            dx, dy = abs(cx[n] - cx[t]), abs(cy[n] - cy[t])
            return (max(dx, dy) - min(dx, dy)) + r2 * min(dx, dy)
        return h
    raise ValueError(kind)


class GraphContext:
    """Bundles a graph with everything the solvers need and builds solver callables.

    solver(name) -> callable(s, t) -> (path | None, cost, nodes_expanded)
    Add your own (e.g. the GNN) via src.scalability.register_solver.
    """

    def __init__(self, G: nx.Graph, coords: Optional[np.ndarray] = None, n_landmarks: int = 4,
                 seed: int = 0, build_alt: bool = True):
        self.G = G
        self.n = G.number_of_nodes()
        self.adj = build_adj(G)
        self.csr = build_csr(G)
        self.coords = coords
        self.alt_preprocess_s = 0.0
        self._alt_D = None
        if build_alt:
            t0 = time.perf_counter()
            self.landmarks, self._alt_D = select_landmarks(self.csr, n_landmarks, seed)
            self.alt_preprocess_s = time.perf_counter() - t0

    def available(self):
        names = ["dijkstra"]
        if self._alt_D is not None:
            names.append("astar_alt")
        if self.coords is not None:
            names += ["astar_euclidean", "astar_octile"]
        return names

    def solver(self, name: str):
        adj = self.adj
        if name == "dijkstra":
            return lambda s, t: dijkstra(adj, s, t)
        if name == "astar_alt":
            if self._alt_D is None:
                raise ValueError("ALT landmarks were not built (build_alt=False).")
            h = alt_heuristic(self._alt_D)
            return lambda s, t: astar(adj, s, t, h)
        if name in ("astar_euclidean", "astar_octile"):
            if self.coords is None:
                raise ValueError(f"{name} needs node coordinates (grid maps only).")
            h = coord_heuristic(self.coords, name.split("_")[1])
            return lambda s, t: astar(adj, s, t, h)
        raise ValueError(f"unknown solver {name}")


# ---------------------------------------------------------------------------
# Ground-truth pair generation + timing
# ---------------------------------------------------------------------------
def generate_pairs(csr, n_pairs: int, min_hops: int = 5, seed: int = 42, targets_per_source: int = 5):
    """Random (source, destination) pairs with the exact Dijkstra shortest path as label.
    One Dijkstra run per source gives paths to many destinations."""
    rng = np.random.default_rng(seed)
    n = csr.shape[0]
    rows, seen, attempts = [], set(), 0
    while len(rows) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        s = int(rng.integers(n))
        dist, pred = sp_dijkstra(csr, directed=False, indices=s, return_predecessors=True)
        reach = np.flatnonzero(np.isfinite(dist))
        reach = reach[reach != s]
        if len(reach) == 0:
            continue
        got = 0
        for t in rng.permutation(reach)[:40]:
            t = int(t)
            if (s, t) in seen:
                continue
            path = reconstruct_path(pred, s, t)
            if path is None or len(path) - 1 < min_hops:
                continue
            rows.append(dict(src=s, dst=t, cost=float(dist[t]), hops=len(path) - 1,
                             path=" ".join(map(str, path))))
            seen.add((s, t))
            got += 1
            if got >= targets_per_source or len(rows) >= n_pairs:
                break
    if not rows:
        raise ValueError("Could not generate any pairs - lower --min_hops or use a bigger graph.")
    if len(rows) < n_pairs:
        print(f"[warn] only {len(rows)} of {n_pairs} requested pairs could be generated.")
    return pd.DataFrame(rows)


def run_queries(solver, pairs: pd.DataFrame, name: str) -> pd.DataFrame:
    """Run a solver on every (src, dst) in `pairs`, timing each query."""
    rows = []
    for s, t in zip(pairs["src"].tolist(), pairs["dst"].tolist()):
        t0 = time.perf_counter()
        path, cost, expanded = solver(int(s), int(t))
        dt = (time.perf_counter() - t0) * 1e3
        rows.append(dict(src=s, dst=t, algo=name, cost=cost, expanded=expanded, runtime_ms=dt,
                         path=" ".join(map(str, path)) if path else ""))
    return pd.DataFrame(rows)
