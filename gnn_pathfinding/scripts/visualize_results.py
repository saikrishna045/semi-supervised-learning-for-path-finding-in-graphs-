from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    ROOT / "results" / "semi_supervised_test_comparison_450.csv"
)

OUTPUT_DIR = ROOT / "results" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not RESULTS_FILE.exists():
    raise FileNotFoundError(
        f"Results file not found: {RESULTS_FILE}"
    )

df = pd.read_csv(RESULTS_FILE)

required_columns = {
    "model",
    "avg_path_cost",
    "avg_runtime_ms",
    "avg_cost_ratio",
    "success_rate_pct",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns in results CSV: {missing}")

print("\nLoaded benchmark results:")
print(df.to_string(index=False))

# --------------------------------------------------
# Graph 1: Average path cost comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    df["model"],
    df["avg_path_cost"],
)

plt.title("Average Path Cost Comparison")
plt.xlabel("Routing Method")
plt.ylabel("Average Path Cost")
plt.xticks(rotation=12, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

path_file = OUTPUT_DIR / "average_path_cost.png"
plt.savefig(path_file, dpi=300)
plt.close()

print(f"\nSaved: {path_file}")

# --------------------------------------------------
# Graph 2: Average runtime comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    df["model"],
    df["avg_runtime_ms"],
)

plt.title("Average Runtime Comparison")
plt.xlabel("Routing Method")
plt.ylabel("Average Runtime (milliseconds)")
plt.xticks(rotation=12, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

runtime_file = OUTPUT_DIR / "average_runtime.png"
plt.savefig(runtime_file, dpi=300)
plt.close()

print(f"Saved: {runtime_file}")

# --------------------------------------------------
# Graph 3: Supervised vs pseudo-labeled GraphSAGE
# --------------------------------------------------

gnn_df = df[
    df["model"].isin([
        "Supervised GraphSAGE",
        "Pseudo-Labeled GraphSAGE",
    ])
].copy()

if len(gnn_df) != 2:
    raise ValueError(
        "Expected both Supervised and Pseudo-Labeled GraphSAGE results."
    )

metrics = [
    ("Average Path Cost", "avg_path_cost"),
    ("Average Runtime (ms)", "avg_runtime_ms"),
    ("Average Cost Ratio", "avg_cost_ratio"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (title, column) in zip(axes, metrics):
    ax.bar(
        gnn_df["model"],
        gnn_df[column],
    )

    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Effect of Dijkstra-Based Pseudo-Labeling",
    fontsize=15,
)

fig.tight_layout()

improvement_file = OUTPUT_DIR / "pseudo_labeling_comparison.png"
fig.savefig(improvement_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {improvement_file}")

# --------------------------------------------------
# Save a concise numerical summary
# --------------------------------------------------

supervised = gnn_df[
    gnn_df["model"] == "Supervised GraphSAGE"
].iloc[0]

pseudo = gnn_df[
    gnn_df["model"] == "Pseudo-Labeled GraphSAGE"
].iloc[0]

cost_reduction = (
    (supervised["avg_path_cost"] - pseudo["avg_path_cost"])
    / supervised["avg_path_cost"]
) * 100

runtime_reduction = (
    (supervised["avg_runtime_ms"] - pseudo["avg_runtime_ms"])
    / supervised["avg_runtime_ms"]
) * 100

summary = [
    "GNN PATHFINDING BENCHMARK SUMMARY",
    "=" * 40,
    f"Test routes: {int(df['routes'].iloc[0])}",
    "",
    f"Supervised average path cost: {supervised['avg_path_cost']:.3f}",
    f"Pseudo-labeled average path cost: {pseudo['avg_path_cost']:.3f}",
    f"Path cost reduction: {cost_reduction:.2f}%",
    "",
    f"Supervised runtime: {supervised['avg_runtime_ms']:.3f} ms",
    f"Pseudo-labeled runtime: {pseudo['avg_runtime_ms']:.3f} ms",
    f"Runtime reduction: {runtime_reduction:.2f}%",
    "",
    "Interpretation:",
    "Pseudo-labeling improved the average path cost and runtime",
    "relative to supervised GraphSAGE in this benchmark.",
    "Dijkstra remained the strongest baseline for path cost and speed.",
    "",
    "Caution: Pseudo-labels were generated using Dijkstra reference paths.",
]

summary_file = OUTPUT_DIR / "benchmark_summary.txt"
summary_file.write_text("\n".join(summary), encoding="utf-8")

print(f"Saved: {summary_file}")

print("\nAll visualizations generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
