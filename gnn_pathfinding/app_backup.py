
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import networkx as nx

ROOT = Path(__file__).resolve().parent

RESULTS_PATH = (
    ROOT / "results" / "semi_supervised_test_comparison_450.csv"
)

GRAPH_PATH = (
    ROOT / "data" / "processed" / "demo_road" / "graph.npz"
)

st.set_page_config(
    page_title="GNN Pathfinding Dashboard",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Semi-Supervised GNN Pathfinding")
st.caption(
    "Comparing Dijkstra, Supervised GraphSAGE, "
    "and Dijkstra-Pseudo-Labeled GraphSAGE"
)

if not RESULTS_PATH.exists():
    st.error(f"Benchmark results not found: {RESULTS_PATH}")
    st.stop()

df = pd.read_csv(RESULTS_PATH)

# -----------------------------
# Overview
# -----------------------------

st.header("Project Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Test Routes", "450")
c2.metric("Dijkstra Success", "100%")
c3.metric("Supervised GNN Success", "100%")
c4.metric("Pseudo-Labeled GNN Success", "100%")

st.info(
    "The graph uses simulated edge weights. "
    "The displayed results come from the saved 450-route benchmark."
)

# -----------------------------
# Benchmark results
# -----------------------------

st.header("Benchmark Results")

st.dataframe(
    df.round(3),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# Path cost
# -----------------------------

st.subheader("Average Path Cost")

fig_cost = px.bar(
    df,
    x="model",
    y="avg_path_cost",
    text="avg_path_cost",
    title="Average Path Cost Across 450 Test Routes",
    labels={
        "model": "Routing Method",
        "avg_path_cost": "Average Path Cost",
    },
)

fig_cost.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig_cost.update_layout(xaxis_tickangle=-15)

st.plotly_chart(fig_cost, use_container_width=True)

# -----------------------------
# Runtime
# -----------------------------

st.subheader("Average Runtime")

fig_runtime = px.bar(
    df,
    x="model",
    y="avg_runtime_ms",
    text="avg_runtime_ms",
    title="Average Runtime Across 450 Test Routes",
    labels={
        "model": "Routing Method",
        "avg_runtime_ms": "Runtime (milliseconds)",
    },
)

fig_runtime.update_traces(
    texttemplate="%{text:.2f} ms",
    textposition="outside",
)
fig_runtime.update_layout(xaxis_tickangle=-15)

st.plotly_chart(fig_runtime, use_container_width=True)

# -----------------------------
# GNN comparison
# -----------------------------

st.subheader("Effect of Pseudo-Labeling")

gnn_df = df[
    df["model"].isin([
        "Supervised GraphSAGE",
        "Pseudo-Labeled GraphSAGE",
    ])
]

if len(gnn_df) == 2:
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

    a, b = st.columns(2)

    a.metric(
        "Average Path Cost Reduction",
        f"{cost_reduction:.2f}%",
    )

    b.metric(
        "Average Runtime Reduction",
        f"{runtime_reduction:.2f}%",
    )

    st.write(
        "In this benchmark, Dijkstra-based pseudo-labeling "
        "reduced average path cost and runtime relative to "
        "Supervised GraphSAGE."
    )

# -----------------------------
# Graph information
# -----------------------------

st.header("Graph Dataset")

if GRAPH_PATH.exists():
    try:
        from src.data_loader import GraphData

        graph_data = GraphData.load(GRAPH_PATH)
        graph = graph_data.to_networkx()

        g1, g2, g3 = st.columns(3)

        g1.metric("Graph Nodes", f"{graph.number_of_nodes():,}")
        g2.metric("Graph Edges", f"{graph.number_of_edges():,}")
        g3.metric(
            "Connected Components",
            f"{nx.number_connected_components(graph):,}",
        )

        st.caption(
            "Network statistics are calculated from the processed graph."
        )

    except Exception as exc:
        st.warning(f"Could not load graph statistics: {exc}")
else:
    st.warning("The processed graph file was not found.")

# -----------------------------
# Interpretation
# -----------------------------

st.header("Key Findings")

st.markdown(
    """
    - All three methods achieved a 100% route success rate in the benchmark.
    - Dijkstra achieved the lowest average path cost and runtime.
    - Pseudo-Labeled GraphSAGE improved average path cost relative to
      Supervised GraphSAGE.
    - The GNN-guided search methods did not outperform Dijkstra in this
      experiment.
    """
)

st.warning(
    "The pseudo-labels were generated from Dijkstra reference paths. "
    "The results therefore measure the effect of additional "
    "Dijkstra-derived training labels, not independent discovery "
    "of optimal paths by the GNN."
)

