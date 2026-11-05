# Semi-supervised GNN Path Finding: data, baselines and evaluation

This folder is the **common foundation** for the project: loading and cleaning the datasets, building graph
objects, labeled/unlabeled splits, ground-truth shortest paths (Dijkstra and A\*), perturbed graphs,
evaluation metrics and scalability experiments. Person B's GNN plugs into the files it produces.

## 1. Where to put the datasets

```
data/raw/roadnet/            <- put roadNet-PA.txt.gz (or .txt) here        (SNAP: snap.stanford.edu/data/roadNet-PA.html)
data/raw/movingai/maps/      <- put the *.map files here                    (movingai.com/benchmarks/grids.html)
data/raw/movingai/scen/      <- put the *.scen files here (optional)
```
You do not need to unzip the RoadNet `.gz`. Each folder has a README file with details. Scripts find the
files automatically, or you can pass `--input` / `--map` explicitly.

## 2. Setup and quick start

```bash
pip install -r requirements.txt

python scripts/run_all.py                                    # RoadNet: steps 1-4 + scalability
python scripts/run_all.py --kind movingai --map arena.map    # a MovingAI map: steps 1-4
```
Want to test before downloading anything? `python scripts/make_demo_data.py` creates small **synthetic** files
in `data/demo/` (same formats as the real data; never report results from them), e.g.
`python scripts/run_all.py --input data/demo/roadNet-DEMO.txt --size 2000 --name demo_road --sizes 500 1000 2000 4000`.

## 3. The pipeline, step by step

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `python scripts/01_prepare_graph.py --kind roadnet` | Load + clean (no self-loops/isolates, largest component, ids 0..N-1), sample a 3000-node connected subgraph, add edge weights and node features | `data/processed/<name>/graph.npz` |
| 2 | `python scripts/02_make_labels_splits.py --dataset roadnet` | Random source/destination pairs, Dijkstra ground-truth paths, 5/10/20% labeled splits | `data/processed/<name>/splits/` |
| 3 | `python scripts/03_run_baselines.py --dataset roadnet` | Dijkstra and A\* on the test pairs (accuracy, time, nodes expanded, memory) | `results/<name>/baselines_summary.csv` |
| 4 | `python scripts/04_make_perturbations.py --dataset roadnet` | Incomplete and evolving graphs + new ground truth + "stale path" baseline | `data/processed/<name>/perturbed/`, `results/<name>/perturbation_summary.csv` |
| 5 | `python scripts/05_evaluate.py --dataset roadnet --pred preds.csv --name gnn_20pct` | Score any prediction file (the GNN's) with the shared metrics | `results/<name>/evaluations.csv` |
| 6 | `python scripts/06_scalability.py --sizes 1000 2000 5000 10000 20000` | Graph size vs. query time, nodes expanded, memory, plus plot | `results/<name>/scalability.csv/.png` |

For MovingAI the dataset name is `movingai_<mapname>` (printed by step 1). Common options: `--size`, `--pairs`, `--seed`
(all defaults live in `src/config.py`).

## 4. Where each task lives

| Task | Code | Output |
|---|---|---|
| Load and clean datasets, build graph objects (adjacency, node features, edge weights) | `src/data_loader.py` (`GraphData`) | `graph.npz` |
| Labeled / unlabeled splits (5, 10, 20%), fixed seed | `src/splits.py` | `splits/*.csv`, `split_meta.json` |
| Ground-truth shortest paths with Dijkstra and A\* | `src/shortest_paths.py` | `pairs_all_truth.csv`, baseline results |
| Incomplete and evolving graphs (edge/node removal, weight drift over time) | `src/perturbations.py` | `perturbed/<version>/` |
| Metrics: path accuracy, cost ratio, success rate, runtime, memory | `src/metrics.py` | `evaluations.csv` |
| Scalability experiments + plot | `src/scalability.py` | `scalability.csv/.png` |

## 5. Data formats (the hand-off to Person B)

**Graph** (`GraphData.load("data/processed/roadnet/graph.npz")`): `.edge_index` (2, 2E, both directions),
`.edge_weight`, `.x` node features (N, F), `.feature_names`, `.coords` (grid maps only).
`.to_networkx()`, `.adjacency()` (scipy CSR) and `.to_pyg()` (PyTorch Geometric `Data`, needs torch_geometric) are provided.

Node features: RoadNet has none, so they are built from structure: normalised degree, clustering coefficient,
mean incident edge weight and 4 Laplacian positional encodings. Grid maps also get normalised x, y.

**Splits** (`data/processed/<name>/splits/`):

| File | Columns | Use |
|---|---|---|
| `labeled_{5,10,20}pct.csv` | src, dst, cost, hops, path | supervised loss |
| `unlabeled_{5,10,20}pct.csv` | src, dst | unlabeled loss (the other train pairs of that regime) |
| `val.csv`, `test.csv` | src, dst, cost, hops, path | model selection / final evaluation |
| `pairs_all_truth.csv` | everything + split flags | evaluation scripts only; do not train on it |

A path is node ids separated by spaces (`"4 17 9 12"`). Labeled sets are nested (5% ⊂ 10% ⊂ 20%). Pairs
are shuffled with the fixed seed, so rerunning gives identical files.

**Predictions** (input of step 5): CSV with `src, dst, path` and optionally `runtime_ms`.

**Perturbed graphs**: each `perturbed/<version>/` has `graph.npz` (features recomputed), `test.csv`, `val.csv`,
`pairs_truth.csv` (all pairs, new optimal paths; unreachable pairs dropped) and `info.json`. Versions:
`edge_removal_10`, `edge_removal_20`, `node_removal_5`, `weight_drift_t1..t3`. Node ids never change.

## 6. Metrics (`src/metrics.py`)

| Metric | Meaning |
|---|---|
| `success_rate` | predicted path is valid: starts at src, ends at dst, every hop is a real edge |
| `path_accuracy` | valid and its cost equals the optimal cost (ties count as correct) |
| `exact_match_rate` | identical node sequence to the stored path (stricter, since equal-cost paths can differ) |
| `near_optimal_rate` | valid and cost ≤ 1.05 × optimal |
| `mean / median / p95_cost_ratio` | predicted cost ÷ optimal cost over valid paths (1.0 = optimal) |
| `mean_runtime_ms` | average query time; `peak_memory_mb` is Python memory allocated while answering queries |

## 7. Things to know

- **RoadNet has no weights, coordinates or features.** Weights are simulated (random integers 1 to 5, seeded;
  change in `config.py`), otherwise Dijkstra would just be BFS. State this in your report.
- **A\* on RoadNet** uses the landmark (ALT) heuristic, which needs no coordinates. On MovingAI maps A\* also
  runs with Euclidean and octile heuristics. All heuristics are admissible, so A\* matches Dijkstra's cost.
- **MovingAI graphs** are 8-connected: straight moves cost 1, diagonals √2, and diagonals are only allowed when
  all four cells of the 2×2 block are free (the benchmark's rule). If a `.scen` file exists, step 2 uses its
  start/goal pairs and prints how many of its optimal lengths differ from our Dijkstra costs.
- **Perturbation baseline.** Step 4 also reports the "stale path": the old optimal path reused on the new
  graph. Long paths break easily (a 34-hop path rarely survives 10% edge removal), and that is the gap a robust
  model has to close.
- Pairs are split at random, so a source node can appear in both train and test with different destinations.
- Everything is seeded (`--seed`, default 42). Use the same seed and the same `data/processed` folder
  for both team members.

## 8. Adding the GNN to the scalability plot (Person B)

```python
from src.scalability import register_solver
register_solver("gnn", lambda ctx: my_solver(ctx))   # my_solver(ctx) returns f(s, t) -> (path or None, cost, 0)
# then: python scripts/06_scalability.py --solvers dijkstra astar_alt gnn
```
(The commented hook is already in `scripts/06_scalability.py`.)

## 9. Folder layout

```
gnn_pathfinding/
├── README.md, requirements.txt
├── data/
│   ├── raw/{roadnet, movingai/maps, movingai/scen}   <- INSERT DATASETS HERE
│   └── processed/                                    <- generated
├── src/        config, data_loader, shortest_paths, splits, perturbations, metrics, scalability
├── scripts/    01_prepare_graph ... 06_scalability, run_all, make_demo_data
└── results/                                          <- generated tables and plots
```
