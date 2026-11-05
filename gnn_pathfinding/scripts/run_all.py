"""
Run steps 1-4 (+ scalability for RoadNet) in one go.

    python scripts/run_all.py                                   # RoadNet from data/raw/roadnet
    python scripts/run_all.py --kind movingai --map arena.map   # a MovingAI map
    python scripts/run_all.py --skip_scalability --size 2000
"""
import argparse
import subprocess
import sys
from pathlib import Path

import _common

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["roadnet", "movingai"], default="roadnet")
ap.add_argument("--input", help="RoadNet file")
ap.add_argument("--map", help="MovingAI map")
ap.add_argument("--scen", help="MovingAI scenario file")
ap.add_argument("--name", help="processed dataset name")
ap.add_argument("--size", type=int, default=3000)
ap.add_argument("--pairs", type=int, default=3000)
ap.add_argument("--sizes", type=int, nargs="*", default=[1000, 2000, 5000, 10000, 20000])
ap.add_argument("--skip_scalability", action="store_true")
a = ap.parse_args()

if a.kind == "movingai":
    name = a.name or _common.default_name("movingai", _common.resolve_map(a.map))
else:
    name = a.name or "roadnet"


def run(script, *args):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print("\n" + "=" * 78 + f"\n>> {' '.join(cmd[1:])}\n" + "=" * 78, flush=True)
    if subprocess.call(cmd) != 0:
        sys.exit(f"{script} failed - fix the error above and re-run.")


s1 = ["--kind", a.kind, "--name", name]
if a.kind == "roadnet":
    s1 += ["--size", a.size] + (["--input", a.input] if a.input else [])
else:
    s1 += (["--map", a.map] if a.map else []) + (["--scen", a.scen] if a.scen else [])
run("01_prepare_graph.py", *s1)
run("02_make_labels_splits.py", "--dataset", name, "--pairs", a.pairs)
run("03_run_baselines.py", "--dataset", name)
run("04_make_perturbations.py", "--dataset", name)
if a.kind == "roadnet" and not a.skip_scalability:
    run("06_scalability.py", "--dataset", name, "--sizes", *a.sizes, *(["--input", a.input] if a.input else []))
print(f"\nAll done. Processed data: data/processed/{name}   Results: results/{name}")
