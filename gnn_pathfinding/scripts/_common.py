"""Shared helpers for the scripts (path setup, friendly errors, dataset lookup)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config                      # noqa: E402
from src.data_loader import GraphData       # noqa: E402


def die(msg: str):
    print(f"\nERROR: {msg}\n")
    sys.exit(1)


def load_processed(dataset: str):
    """Returns (GraphData, networkx graph, info dict) for a prepared dataset."""
    d = config.processed_dir(dataset)
    p = d / "graph.npz"
    if not p.exists():
        die(f"{p} not found. Run  python scripts/01_prepare_graph.py  first "
            f"(dataset name was '{dataset}').")
    g = GraphData.load(p)
    info_path = d / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return g, g.to_networkx(), info


def resolve_map(arg):
    """MovingAI map file: explicit path, a name inside data/raw/movingai/maps, or the first .map there."""
    folder = config.RAW_MOVINGAI_MAPS
    if arg:
        p = Path(arg)
        if p.exists():
            return p
        if (folder / arg).exists():
            return folder / arg
        die(f"Map '{arg}' not found. Put .map files in {folder}")
    maps = sorted(folder.glob("*.map")) if folder.exists() else []
    if not maps:
        die(f"No .map files found in {folder}\n"
            f"Download maps from https://movingai.com/benchmarks/grids.html and put them there "
            f"(or run scripts/make_demo_data.py to test with a synthetic map).")
    return maps[0]


def default_name(kind: str, map_path=None) -> str:
    return "roadnet" if kind == "roadnet" else f"movingai_{Path(map_path).stem}"
