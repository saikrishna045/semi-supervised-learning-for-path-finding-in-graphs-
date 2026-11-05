
from pathlib import Path
import time

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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

# --------------------------------------------------
# Load benchmark results
# --------------------------------------------------

if not RESULTS_PATH.exists():
    st.error(
        f"Benchmark results not found:\n{RESULTS_PATH}"
    )
    st.stop()

df = pd.read_csv(RESULTS_PATH)

required_columns = {
    "model",
    "routes",
    "success_rate_pct",
    "avg_path_cost",
    "avg_cost_ratio",
    "avg_runtime_ms",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        f"The benchmark CSV is missing columns: "
        f"{', '.join(sorted(missing_columns))}"
    )
    st.stop()


# --------------------------------------------------
# Project overview
# --------------------------------------------------

st.header("📊 Project Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Test Routes", "450")
c2.metric("Dijkstra Success", "100%")
c3.metric("Supervised GNN Success", "100%")
c4.metric("Pseudo-Labeled GNN Success", "100%")

st.info(
    "The graph uses simulated edge weights. "
    "Results below come from the saved 450-route benchmark."
)


# --------------------------------------------------
# Benchmark table
# --------------------------------------------------

st.header("📋 Benchmark Results")

st.dataframe(
    df.round(3),
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Average path cost
# --------------------------------------------------

st.subheader("Average Path Cost")

fig_cost = px.bar(
    df,
    x="model",
    y="avg_path_cost",
    color="model",
    text="avg_path_cost",
    title="Average Path Cost Across 450 Test Routes",
    labels={
        "model": "Routing Method",
        "avg_path_cost": "Average Path Cost",
    },
)

fig_cost.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
)

fig_cost.update_layout(
    xaxis_tickangle=-15,
    showlegend=False,
)

st.plotly_chart(
    fig_cost,
    use_container_width=True,
)


# --------------------------------------------------
# Average runtime
# --------------------------------------------------

st.subheader("Average Runtime")

fig_runtime = px.bar(
    df,
    x="model",
    y="avg_runtime_ms",
    color="model",
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

fig_runtime.update_layout(
    xaxis_tickangle=-15,
    showlegend=False,
)

st.plotly_chart(
    fig_runtime,
    use_container_width=True,
)


# --------------------------------------------------
# Semi-supervised learning comparison
# --------------------------------------------------

st.header("🧠 Effect of Pseudo-Labeling")

gnn_df = df[
    df["model"].isin(
        [
            "Supervised GraphSAGE",
            "Pseudo-Labeled GraphSAGE",
        ]
    )
]

if len(gnn_df) == 2:
    supervised = gnn_df[
        gnn_df["model"] == "Supervised GraphSAGE"
    ].iloc[0]

    pseudo = gnn_df[
        gnn_df["model"] == "Pseudo-Labeled GraphSAGE"
    ].iloc[0]

    cost_reduction = (
        (
            supervised["avg_path_cost"]
            - pseudo["avg_path_cost"]
        )
        / supervised["avg_path_cost"]
        * 100
    )

    runtime_reduction = (
        (
            supervised["avg_runtime_ms"]
            - pseudo["avg_runtime_ms"]
        )
        / supervised["avg_runtime_ms"]
        * 100
    )

    a, b = st.columns(2)

    a.metric(
        "Path Cost Reduction",
        f"{cost_reduction:.2f}%",
    )

    b.metric(
        "Runtime Reduction",
        f"{runtime_reduction:.2f}%",
    )

    st.write(
        "In this benchmark, the pseudo-labeled model "
        "achieved a lower average path cost and runtime "
        "than the supervised GraphSAGE model."
    )


# --------------------------------------------------
# Load graph
# --------------------------------------------------

@st.cache_resource
def load_graph(graph_path):
    from src.data_loader import GraphData

    graph_data = GraphData.load(graph_path)
    return graph_data.to_networkx()


graph = None

if GRAPH_PATH.exists():
    try:
        graph = load_graph(str(GRAPH_PATH))

        st.header("🌐 Graph Dataset")

        g1, g2, g3 = st.columns(3)

        g1.metric(
            "Graph Nodes",
            f"{graph.number_of_nodes():,}",
        )

        g2.metric(
            "Graph Edges",
            f"{graph.number_of_edges():,}",
        )

        g3.metric(
            "Connected Components",
            f"{nx.number_connected_components(graph):,}",
        )

        st.caption(
            "Network statistics are calculated from the "
            "processed graph."
        )

    except Exception as exc:
        st.error(f"Could not load the processed graph: {exc}")

else:
    st.error(
        f"Processed graph file not found:\n{GRAPH_PATH}"
    )


# --------------------------------------------------
# Interactive shortest-path finder
# --------------------------------------------------

st.header("🗺️ Interactive Shortest Path Finder")

st.write(
    "Choose any two nodes in the graph to calculate "
    "their shortest route using Dijkstra's algorithm."
)

if graph is not None and graph.number_of_nodes() > 0:

    node_ids = sorted(graph.nodes())

    col1, col2 = st.columns(2)

    with col1:
        source = st.selectbox(
            "Select Source Node",
            options=node_ids,
            index=0,
            key="source_node",
        )

    with col2:
        default_destination = (
            100 if 100 in graph else node_ids[
                min(1, len(node_ids) - 1)
            ]
        )

        destination = st.selectbox(
            "Select Destination Node",
            options=node_ids,
            index=node_ids.index(default_destination),
            key="destination_node",
        )

    if st.button(
        "🔍 Find Shortest Path",
        type="primary",
        use_container_width=True,
    ):

        if source == destination:

            st.warning(
                "Please select two different nodes."
            )

        else:

            try:
                start_time = time.perf_counter()

                path = nx.shortest_path(
                    graph,
                    source=source,
                    target=destination,
                    weight="weight",
                    method="dijkstra",
                )

                path_cost = nx.path_weight(
                    graph,
                    path,
                    weight="weight",
                )

                runtime_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                st.session_state["selected_path"] = {
                    "source": source,
                    "destination": destination,
                    "path": path,
                    "cost": path_cost,
                    "runtime_ms": runtime_ms,
                }

            except nx.NetworkXNoPath:

                st.session_state.pop(
                    "selected_path",
                    None,
                )

                st.error(
                    "No path exists between the selected nodes."
                )

            except nx.NodeNotFound:

                st.session_state.pop(
                    "selected_path",
                    None,
                )

                st.error(
                    "The selected source or destination "
                    "does not exist in the graph."
                )

    # ----------------------------------------------
    # Display the selected path
    # ----------------------------------------------

    result = st.session_state.get("selected_path")

    if (
        result is not None
        and result["source"] == source
        and result["destination"] == destination
    ):

        path = result["path"]

        st.success(
            f"Shortest path found: {source} → {destination}"
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Total Path Cost",
            f"{result['cost']:.2f}",
        )

        m2.metric(
            "Number of Hops",
            str(len(path) - 1),
        )

        m3.metric(
            "Execution Time",
            f"{result['runtime_ms']:.3f} ms",
        )

        st.subheader("Calculated Route")

        st.code(
            " → ".join(map(str, path)),
            language=None,
        )

        # ------------------------------------------
        # Route visualization
        # ------------------------------------------

        st.subheader("📍 Route Visualization")

        st.caption(
            "This visualization shows the selected route "
            "and nearby graph connections. Node positions "
            "are generated for display and do not represent "
            "real geographical coordinates."
        )

        # Build a neighborhood around the path so that
        # the complete 2,000-node graph does not clutter
        # the visualization.

        visible_nodes = set(path)

        for node in path:
            visible_nodes.update(
                list(graph.neighbors(node))[:8]
            )

        visible_graph = graph.subgraph(
            visible_nodes
        ).copy()

        # Deterministic layout for a repeatable display.
        pos = nx.spring_layout(
            visible_graph,
            seed=42,
            weight="weight",
            iterations=30,
        )

        path_edges = list(
            zip(path[:-1], path[1:])
        )

        path_edge_set = {
            frozenset(edge)
            for edge in path_edges
        }

        # Ordinary edges.
        edge_x = []
        edge_y = []

        # Shortest-path edges.
        route_x = []
        route_y = []

        for u, v in visible_graph.edges():

            x0, y0 = pos[u]
            x1, y1 = pos[v]

            if frozenset((u, v)) in path_edge_set:

                route_x.extend(
                    [x0, x1, None]
                )

                route_y.extend(
                    [y0, y1, None]
                )

            else:

                edge_x.extend(
                    [x0, x1, None]
                )

                edge_y.extend(
                    [y0, y1, None]
                )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(
                    width=1,
                    color="rgba(150,150,150,0.35)",
                ),
                hoverinfo="skip",
                name="Other edges",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=route_x,
                y=route_y,
                mode="lines",
                line=dict(
                    width=4,
                    color="#00CC96",
                ),
                hoverinfo="skip",
                name="Shortest route",
            )
        )

        node_x = []
        node_y = []
        node_text = []
        node_colors = []

        for node in visible_graph.nodes():

            x, y = pos[node]

            node_x.append(x)
            node_y.append(y)

            if node == source:
                node_colors.append("#00CC96")
                node_text.append(
                    f"Source: {node}"
                )

            elif node == destination:
                node_colors.append("#EF553B")
                node_text.append(
                    f"Destination: {node}"
                )

            elif node in path:
                node_colors.append("#636EFA")
                node_text.append(
                    f"Route node: {node}"
                )

            else:
                node_colors.append("#B8C2CC")
                node_text.append(
                    f"Node: {node}"
                )

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(
                    size=9,
                    color=node_colors,
                    line=dict(
                        width=1,
                        color="white",
                    ),
                ),
                text=node_text,
                hovertemplate="%{text}<extra></extra>",
                name="Nodes",
            )
        )

        fig.update_layout(
            title=(
                f"Shortest Route: {source} → {destination}"
            ),
            showlegend=True,
            height=600,
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10,
            ),
            xaxis=dict(
                visible=False,
            ),
            yaxis=dict(
                visible=False,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "Green: source node and shortest-route edges. "
            "Red: destination node. Blue: intermediate "
            "route nodes. Gray: nearby graph connections."
        )


# --------------------------------------------------
# Interpretation
# --------------------------------------------------

st.header("📌 Key Findings")

st.markdown(
    """
    - All three methods achieved a 100% route success rate
      in the saved benchmark.
    - Dijkstra achieved the lowest average path cost and
      runtime in the current experiment.
    - Pseudo-Labeled GraphSAGE improved average path cost
      relative to Supervised GraphSAGE.
    - The GNN-guided methods did not outperform Dijkstra
      in the current benchmark.
    """
)

st.warning(
    "The pseudo-labels were generated from Dijkstra "
    "reference paths. The results measure the effect of "
    "additional Dijkstra-derived training labels; they do "
    "not demonstrate independent discovery of optimal paths "
    "by the GNN."
)


