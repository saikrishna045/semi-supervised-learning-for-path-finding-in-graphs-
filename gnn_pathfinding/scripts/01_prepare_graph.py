"""
STEP 1 - Load the raw dataset, clean it and build the graph object.

RoadNet (SNAP edge list):
    python scripts/01_prepare_graph.py --kind roadnet
    python scripts/01_prepare_graph.py --kind roadnet --input data/raw/roadnet/roadNet-PA.txt --size 3000

MovingAI grid map:
    python scripts/01_prepare_graph.py --kind movingai --map arena.map

Output:  data/processed/<name>/graph.npz   (+ info.json, node_id_map.npy / cell_to_node.npy)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import _common
from _common import config, die
from src.data_loader import (GraphData, add_edge_weights, clean_graph, find_roadnet_file, grid_to_graph,
                             load_edge_list, read_movingai_map, sample_connected_subgraph)

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["roadnet", "movingai"], default="roadnet")
ap.add_argument("--input", help="RoadNet edge-list file (default: first file in data/raw/roadnet)")
ap.add_argument("--map", help="MovingAI .map file name or path (default: first one in data/raw/movingai/maps)")
ap.add_argument("--scen", help="MovingAI .scen file (default: <map>.scen in data/raw/movingai/scen)")
ap.add_argument("--name", help="Name of the processed dataset folder")
ap.add_argument("--size", type=int, default=config.SUBGRAPH_SIZE, help="RoadNet: nodes in the sampled subgraph")
ap.add_argument("--seed", type=int, default=config.SEED)
ap.add_argument("--wmin", type=int, default=config.EDGE_WEIGHT_RANGE[0])
ap.add_argument("--wmax", type=int, default=config.EDGE_WEIGHT_RANGE[1])
ap.add_argument("--lap_pe", type=int, default=config.LAP_PE_DIM)
ap.add_argument("--no_diagonal", action="store_true", help="MovingAI: 4-connected grid instead of 8-connected")
a = ap.parse_args()

t0 = time.time()
if a.kind == "roadnet":
    path = Path(a.input) if a.input else find_roadnet_file(config.RAW_ROADNET)
    if path is None or not path.exists():
        die(f"No RoadNet file found.\nDownload roadNet-PA.txt.gz from https://snap.stanford.edu/data/roadNet-PA.html\n"
            f"and put it in: {config.RAW_ROADNET}\n(or pass --input <file>; to try the pipeline first, run scripts/make_demo_data.py)")
    name = a.name or _common.default_name("roadnet")
    print(f"Loading {path} ...")
    G = load_edge_list(path)
    print(f"  raw graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    sub = sample_connected_subgraph(G, a.size, a.seed)
    H, old_ids = clean_graph(sub)
    add_edge_weights(H, a.seed, a.wmin, a.wmax)
    coords = None
    extra = {"source_file": str(path), "weights": f"simulated integers {a.wmin}..{a.wmax}",
             "raw_nodes": G.number_of_nodes(), "raw_edges": G.number_of_edges()}
    arrays = {"node_id_map.npy": old_ids}          # new id -> original RoadNet id
    scen_path = None
else:
    map_path = _common.resolve_map(a.map)
    name = a.name or _common.default_name("movingai", map_path)
    print(f"Loading {map_path} ...")
    grid, w, h = read_movingai_map(map_path)
    G, coords_raw, ids = grid_to_graph(grid, w, h, diagonal=not a.no_diagonal)
    H, old_ids = clean_graph(G)
    coords = coords_raw[old_ids]
    lookup = -np.ones(G.number_of_nodes(), dtype=np.int64)
    lookup[old_ids] = np.arange(len(old_ids))
    cell_to_node = np.where(ids >= 0, lookup[np.clip(ids, 0, None)], -1)
    scen_path = Path(a.scen) if a.scen else None
    if scen_path is None:
        for cand in (config.RAW_MOVINGAI_SCEN / f"{map_path.name}.scen", config.RAW_MOVINGAI_SCEN / f"{map_path.stem}.scen"):
            if cand.exists():
                scen_path = cand
                break
    extra = {"source_file": str(map_path), "map_width": w, "map_height": h,
             "diagonal_moves": not a.no_diagonal, "scen_file": str(scen_path) if scen_path else None}
    arrays = {"cell_to_node.npy": cell_to_node}
    print(f"  grid {w}x{h}: {int((ids >= 0).sum()):,} free cells, kept {H.number_of_nodes():,} in the largest component")
    print(f"  scenario file: {scen_path if scen_path else 'none found (random pairs will be generated)'}")

gd = GraphData.from_networkx(H, name, coords=coords, lap_pe_dim=a.lap_pe,
                             meta={"kind": a.kind, "seed": a.seed, **extra})
out = config.processed_dir(name)
out.mkdir(parents=True, exist_ok=True)
gd.save(out / "graph.npz")
for fname, arr in arrays.items():
    np.save(out / fname, arr)
info = {"dataset": name, "kind": a.kind, "nodes": gd.num_nodes, "edges": gd.num_edges,
        "features": gd.feature_names, "has_coords": coords is not None, "seed": a.seed, **extra}
(out / "info.json").write_text(json.dumps(info, indent=2, default=str))

print(f"\nDataset '{name}': {gd.num_nodes:,} nodes, {gd.num_edges:,} edges, {gd.x.shape[1]} features {gd.feature_names}")
print(f"Saved to {out}   ({time.time() - t0:.1f}s)")
print(f"Next:  python scripts/02_make_labels_splits.py --dataset {name}")
