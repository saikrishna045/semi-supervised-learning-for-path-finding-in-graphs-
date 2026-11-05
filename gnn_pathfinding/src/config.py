"""Central paths and default settings. Edit values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- WHERE YOU PUT THE DOWNLOADED DATASETS --------------------------------
RAW_DIR = ROOT / "data" / "raw"
RAW_ROADNET = RAW_DIR / "roadnet"                    # put roadNet-PA.txt (or .txt.gz) here
RAW_MOVINGAI_MAPS = RAW_DIR / "movingai" / "maps"    # put *.map files here
RAW_MOVINGAI_SCEN = RAW_DIR / "movingai" / "scen"    # put *.scen files here

# ---- GENERATED FILES -------------------------------------------------------
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# ---- DEFAULT EXPERIMENT SETTINGS -------------------------------------------
SEED = 42
SUBGRAPH_SIZE = 3000            # nodes sampled from RoadNet (1.1M nodes is too big for one day)
EDGE_WEIGHT_RANGE = (1, 5)      # RoadNet has no weights: simulated travel times (integers)
N_PAIRS = 3000                  # source/destination queries to generate
MIN_HOPS = 5                    # skip trivial queries shorter than this many edges
LABEL_FRACTIONS = (0.05, 0.10, 0.20)   # share of TRAIN pairs that keep their path label
TEST_FRAC = 0.15
VAL_FRAC = 0.10
LAP_PE_DIM = 4                  # Laplacian positional-encoding features added to each node
N_LANDMARKS = 4                 # landmarks for the ALT heuristic (A* on graphs without coordinates)


def processed_dir(dataset: str) -> Path:
    return PROCESSED_DIR / dataset


def results_dir(dataset: str) -> Path:
    return RESULTS_DIR / dataset
