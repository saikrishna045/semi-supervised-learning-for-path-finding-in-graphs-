"""
OPTIONAL - create small SYNTHETIC datasets in the same file formats as the real ones, so you can
test the whole pipeline in a minute before (or while) downloading RoadNet / MovingAI.

    python scripts/make_demo_data.py          # writes into data/demo/ (never into data/raw/)

Then, for example:
    python scripts/01_prepare_graph.py --kind roadnet  --input data/demo/roadNet-DEMO.txt --size 2000 --name demo_road
    python scripts/01_prepare_graph.py --kind movingai --map data/demo/movingai/maps/demo_rooms.map --scen data/demo/movingai/scen/demo_rooms.map.scen
These are NOT real datasets - do not report results from them.
"""
import argparse

import numpy as np

import _common
from _common import ROOT
from src.data_loader import clean_graph, grid_to_graph, read_movingai_map
from src.shortest_paths import build_csr
from scipy.sparse.csgraph import dijkstra

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "data" / "demo"))
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
from pathlib import Path  # noqa: E402

out = Path(a.out)
rng = np.random.default_rng(a.seed)

# ---- synthetic road network in SNAP format (grid with random missing roads) ----
L = 100
ids = rng.permutation(100000)[: L * L].reshape(L, L)        # scrambled node ids, like real data
edges = []
for y in range(L):
    for x in range(L):
        if x + 1 < L and rng.random() < 0.75:
            edges.append((ids[y, x], ids[y, x + 1]))
        if y + 1 < L and rng.random() < 0.75:
            edges.append((ids[y, x], ids[y + 1, x]))
road = out / "roadNet-DEMO.txt"
road.parent.mkdir(parents=True, exist_ok=True)
with open(road, "w") as f:
    f.write("# Synthetic demo road network (NOT real data)\n# FromNodeId\tToNodeId\n")
    for u, v in edges:
        f.write(f"{u}\t{v}\n")
print(f"wrote {road}  ({len(edges)} edges)")

# ---- synthetic MovingAI map + scenario file ----
H, W = 64, 64
m = rng.random((H, W)) < 0.18
m[::16, :] = True
m[:, ::16] = True                                              # walls forming rooms ...
for k in range(0, H, 16):                                      # ... with doors
    for j in range(2, W, 8):
        m[k, j:j + 2] = False
        m[j:j + 2, k] = False
grid = ["".join("@" if c else "." for c in row) for row in m]
mp = out / "movingai" / "maps" / "demo_rooms.map"
mp.parent.mkdir(parents=True, exist_ok=True)
mp.write_text(f"type octile\nheight {H}\nwidth {W}\nmap\n" + "\n".join(grid) + "\n")

g_, w_, h_ = read_movingai_map(mp)
G, coords, cell_ids = grid_to_graph(g_, w_, h_, diagonal=True)
Hc, old = clean_graph(G)
lookup = -np.ones(G.number_of_nodes(), dtype=int)
lookup[old] = np.arange(len(old))
xy = coords[old].astype(int)
csr = build_csr(Hc)
lines = ["version 1"]
for _ in range(300):
    s = int(rng.integers(Hc.number_of_nodes()))
    dist = dijkstra(csr, directed=False, indices=s)
    t = int(rng.choice(np.flatnonzero((dist > 8) & np.isfinite(dist))))
    lines.append(f"0\tdemo_rooms.map\t{W}\t{H}\t{xy[s][0]}\t{xy[s][1]}\t{xy[t][0]}\t{xy[t][1]}\t{dist[t]:.8f}")
sc = out / "movingai" / "scen" / "demo_rooms.map.scen"
sc.parent.mkdir(parents=True, exist_ok=True)
sc.write_text("\n".join(lines) + "\n")
print(f"wrote {mp} and {sc}")
