"""
data_loader.py  --  load + clean raw datasets and build graph objects.

Both team members use this module. The main object is `GraphData`:
    edge_index  (2, 2E)  both directions, int64       -> adjacency
    edge_weight (2E,)    float32                      -> edge weights
    x           (N, F)   float32                      -> node features
    coords      (N, 2)   float32 or None              -> grid x,y (MovingAI only)

Typical use:
    from src.data_loader import GraphData
    g = GraphData.load("data/processed/roadnet/graph.npz")
    G = g.to_networkx()          # networkx graph with 'weight' on edges
    A = g.adjacency()            # scipy CSR weighted adjacency matrix
    data = g.to_pyg()            # PyTorch Geometric Data (needs torch_geometric)
"""
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

SQRT2 = 2 ** 0.5
PASSABLE = set(".GS")   # MovingAI: '.' and 'G' ground, 'S' swamp are walkable; '@','O','T','W' are not


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------
@dataclass
class GraphData:
    name: str
    num_nodes: int
    edge_index: np.ndarray
    edge_weight: np.ndarray
    x: np.ndarray
    feature_names: list
    coords: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def num_edges(self) -> int:
        """Number of undirected edges."""
        return int(self.edge_index.shape[1] // 2)

    # ---- conversions ----
    def adjacency(self) -> sp.csr_matrix:
        n = self.num_nodes
        return sp.csr_matrix(
            (self.edge_weight.astype(np.float64), (self.edge_index[0], self.edge_index[1])),
            shape=(n, n),
        )

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        src, dst = self.edge_index
        keep = src < dst
        G.add_weighted_edges_from(
            zip(src[keep].tolist(), dst[keep].tolist(), self.edge_weight[keep].astype(float).tolist())
        )
        return G

    def to_pyg(self):
        """PyTorch Geometric `Data` object (install torch + torch_geometric first)."""
        import torch
        from torch_geometric.data import Data

        w = torch.tensor(self.edge_weight, dtype=torch.float32)
        d = Data(
            x=torch.tensor(self.x, dtype=torch.float32),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            edge_attr=w.view(-1, 1),
            edge_weight=w,
        )
        if self.coords is not None:
            d.pos = torch.tensor(self.coords, dtype=torch.float32)
        return d

    # ---- build / save / load ----
    @staticmethod
    def from_networkx(G: nx.Graph, name: str, coords=None, lap_pe_dim: int = 4, meta=None) -> "GraphData":
        """G must have nodes 0..N-1 and a 'weight' attribute on edges."""
        n = G.number_of_nodes()
        if set(G.nodes) != set(range(n)):
            raise ValueError("Graph nodes must be exactly 0..N-1 (use clean_graph first).")
        edges = [(u, v, float(d.get("weight", 1.0))) for u, v, d in G.edges(data=True)]
        if edges:
            u, v, w = (np.array(c) for c in zip(*edges))
        else:
            u = v = np.zeros(0, dtype=np.int64)
            w = np.zeros(0)
        edge_index = np.concatenate([np.stack([u, v]), np.stack([v, u])], axis=1).astype(np.int64)
        edge_weight = np.concatenate([w, w]).astype(np.float32)
        x, names = build_node_features(G, coords=coords, lap_pe_dim=lap_pe_dim)
        return GraphData(
            name=name, num_nodes=n, edge_index=edge_index, edge_weight=edge_weight, x=x,
            feature_names=names,
            coords=None if coords is None else np.asarray(coords, dtype=np.float32),
            meta=dict(meta or {}),
        )

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            edge_index=self.edge_index, edge_weight=self.edge_weight, x=self.x,
            feature_names=np.array(self.feature_names), num_nodes=np.array(self.num_nodes),
            name=np.array(self.name), meta=np.array(json.dumps(self.meta, default=str)),
        )
        if self.coords is not None:
            arrays["coords"] = self.coords
        np.savez_compressed(path, **arrays)

    @staticmethod
    def load(path) -> "GraphData":
        z = np.load(Path(path), allow_pickle=False)
        return GraphData(
            name=str(z["name"]), num_nodes=int(z["num_nodes"]), edge_index=z["edge_index"],
            edge_weight=z["edge_weight"], x=z["x"], feature_names=z["feature_names"].tolist(),
            coords=z["coords"] if "coords" in z.files else None,
            meta=json.loads(str(z["meta"])),
        )


# ---------------------------------------------------------------------------
# Cleaning + sampling
# ---------------------------------------------------------------------------
def clean_graph(G: nx.Graph):
    """Simple undirected graph, no self-loops / isolates, largest component, nodes -> 0..N-1.

    Returns (clean_graph, old_ids) where old_ids[i] is the original id of new node i
    (use it to re-index coordinates).
    """
    G = nx.Graph(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty after cleaning.")
    lcc = max(nx.connected_components(G), key=len)
    old_ids = np.array(sorted(lcc))
    mapping = {int(old): i for i, old in enumerate(old_ids)}
    H = nx.relabel_nodes(G.subgraph(lcc), mapping, copy=True)
    return H, old_ids


def sample_connected_subgraph(G: nx.Graph, size: int, seed: int = 0, max_tries: int = 20) -> nx.Graph:
    """BFS ball of `size` nodes around a random start (stays connected). Keeps original ids."""
    if G.number_of_nodes() <= size:
        return G.copy()
    rng = random.Random(seed)
    nodes = list(G.nodes)
    best = None
    for _ in range(max_tries):
        start = rng.choice(nodes)
        seen, queue = {start}, deque([start])
        while queue and len(seen) < size:
            u = queue.popleft()
            for v in G.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
                    if len(seen) >= size:
                        break
        if best is None or len(seen) > len(best):
            best = seen
        if len(seen) >= size:
            break
    return G.subgraph(best).copy()


def add_edge_weights(G: nx.Graph, seed: int = 0, low: int = 1, high: int = 5) -> None:
    """RoadNet has no weights, so simulate integer travel times in [low, high]."""
    rng = np.random.default_rng(seed)
    edges = list(G.edges)
    w = rng.integers(low, high + 1, size=len(edges))
    for (u, v), wi in zip(edges, w):
        G[u][v]["weight"] = float(wi)


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------
def laplacian_positional_encoding(G: nx.Graph, k: int) -> np.ndarray:
    n = G.number_of_nodes()
    if k <= 0:
        return np.zeros((n, 0), dtype=np.float32)
    pe = np.zeros((n, k))
    try:
        L = nx.normalized_laplacian_matrix(G, nodelist=range(n), weight=None).astype(float)
        if n <= 800:
            vals, vecs = np.linalg.eigh(L.toarray())
        else:
            vals, vecs = eigsh(L, k=min(k + 1, n - 1), sigma=-0.01, which="LM")
            vecs = vecs[:, np.argsort(vals)]
        got = vecs[:, 1:k + 1]
        pe[:, :got.shape[1]] = got
        # fix the sign ambiguity of eigenvectors so results are reproducible
        idx = np.argmax(np.abs(pe), axis=0)
        signs = np.sign(pe[idx, np.arange(k)])
        signs[signs == 0] = 1
        pe = pe * signs
        std = pe.std(axis=0)
        pe = pe / np.where(std > 1e-12, std, 1.0)      # unit variance, so the scale matches other features
    except Exception as e:  # keep the pipeline alive; features simply become zeros
        print(f"[warn] Laplacian PE failed ({e}); using zeros.")
    return pe.astype(np.float32)


def build_node_features(G: nx.Graph, coords=None, lap_pe_dim: int = 4):
    """Structural features (RoadNet has none): degree, clustering, mean edge weight,
    normalised coordinates (if any) and Laplacian positional encodings."""
    n = G.number_of_nodes()
    deg = np.array([G.degree(i) for i in range(n)], dtype=np.float64)
    wdeg = np.array([G.degree(i, weight="weight") for i in range(n)], dtype=np.float64)
    mean_w = np.divide(wdeg, deg, out=np.zeros(n), where=deg > 0)
    clus_d = nx.clustering(G)
    clus = np.array([clus_d[i] for i in range(n)], dtype=np.float64)

    cols = [deg / max(deg.max(), 1.0), clus, mean_w / max(mean_w.max(), 1e-9)]
    names = ["degree_norm", "clustering", "mean_edge_weight_norm"]
    if coords is not None:
        c = np.asarray(coords, dtype=np.float64)
        lo = c.min(axis=0)
        span = np.maximum(c.max(axis=0) - lo, 1e-9)
        cn = (c - lo) / span
        cols += [cn[:, 0], cn[:, 1]]
        names += ["x_norm", "y_norm"]
    pe = laplacian_positional_encoding(G, lap_pe_dim)
    for j in range(pe.shape[1]):
        cols.append(pe[:, j])
        names.append(f"lap_pe_{j}")
    return np.stack(cols, axis=1).astype(np.float32), names


# ---------------------------------------------------------------------------
# RoadNet (SNAP edge list)
# ---------------------------------------------------------------------------
def load_edge_list(path) -> nx.Graph:
    """SNAP format: '#' comment lines, then 'FromNodeId<TAB>ToNodeId'. .gz is fine too."""
    return nx.read_edgelist(str(path), comments="#", nodetype=int)


def find_roadnet_file(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    ok = {".txt", ".gz", ".edges", ".csv"}
    files = [p for p in sorted(folder.iterdir())
             if p.is_file() and p.suffix.lower() in ok and not p.name.upper().startswith("README")]
    return files[0] if files else None


# ---------------------------------------------------------------------------
# MovingAI grid maps
# ---------------------------------------------------------------------------
def read_movingai_map(path):
    lines = Path(path).read_text().splitlines()
    header, i = {}, 0
    while i < len(lines):
        if lines[i].strip() == "map":
            i += 1
            break
        parts = lines[i].split()
        if len(parts) == 2:
            header[parts[0].lower()] = parts[1]
        i += 1
    h, w = int(header["height"]), int(header["width"])
    grid = [row.ljust(w, "@")[:w] for row in lines[i:i + h]]
    while len(grid) < h:
        grid.append("@" * w)
    return grid, w, h


def grid_to_graph(grid, w: int, h: int, diagonal: bool = True):
    """Grid -> graph. Cardinal moves cost 1, diagonal moves cost sqrt(2) and are only allowed
    when all four cells of the 2x2 block are free (MovingAI 'no corner cutting' rule).

    Returns (G, coords (N,2) as x,y, ids (h,w) with node id or -1).
    """
    free = np.array([[c in PASSABLE for c in row] for row in grid], dtype=bool)
    ids = -np.ones((h, w), dtype=np.int64)
    ids[free] = np.arange(int(free.sum()))
    ys, xs = np.nonzero(free)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)

    G = nx.Graph()
    G.add_nodes_from(range(int(free.sum())))

    def add(src, dst, wt):
        G.add_weighted_edges_from(zip(src.tolist(), dst.tolist(), [wt] * len(src)))

    right = free[:, :-1] & free[:, 1:]
    add(ids[:, :-1][right], ids[:, 1:][right], 1.0)
    down = free[:-1, :] & free[1:, :]
    add(ids[:-1, :][down], ids[1:, :][down], 1.0)
    if diagonal:
        block = free[:-1, :-1] & free[:-1, 1:] & free[1:, :-1] & free[1:, 1:]
        add(ids[:-1, :-1][block], ids[1:, 1:][block], SQRT2)    # top-left  -> bottom-right
        add(ids[:-1, 1:][block], ids[1:, :-1][block], SQRT2)    # top-right -> bottom-left
    return G, coords, ids


def read_scen(path):
    """Rows of (start_x, start_y, goal_x, goal_y, optimal_length)."""
    out = []
    for line in Path(path).read_text().splitlines()[1:]:
        p = line.split()
        if len(p) >= 9:
            out.append((int(p[4]), int(p[5]), int(p[6]), int(p[7]), float(p[8])))
    return out
