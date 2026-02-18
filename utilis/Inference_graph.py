import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from graphviz import Graph
import torch


def visualize_articulated_graph(
    joint_conne_pred, adj,
    img=None, threshold=0.5,
    node_labels=None, figsize=(12, 8),
    out_prefix="articulated_graph"
):

    # ---- to numpy ----
    joint_conne_pred = joint_conne_pred.detach().cpu().numpy()
    adj_np = adj.detach().cpu().numpy() if hasattr(adj, "detach") else np.asarray(adj)
    N = joint_conne_pred.shape[0]

    if node_labels is None:
        node_labels = [f"P{i}" for i in range(N)]


    # Graphviz paper-style palette
    gv_palette = [
        "#d62728",  # red
        "#17becf",  # cyan
        "#1f77b4",  # blue
        "#9467bd",  # purple
        "#2ca02c",   # green
        "#e377c2",  # pink
        "#8c564b",  # brown
        "#ff7f0e",  # orange
        "#7f7f7f",  # gray
    ]

    node_colors = [gv_palette[i % len(gv_palette)] for i in range(N)]

    # ---- initial edges ----
    init_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj_np[i, j] > threshold:
                init_edges.append((i, j))

    # ---- predicted edges ----
    pred_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if joint_conne_pred[i, j] >= threshold:
                pred_edges.append((i, j))

    # ---- helper to render graph ----
    def render_graphviz_graph(name, engine, edges):

        g = Graph(name=name, engine=engine, format="png",)

        g.attr(
            "graph",
            bgcolor="white",
            overlap="false",
            splines="true",
            pad="0.4",
            sep="+20",        # increase node separation
            nodesep="0.8",    # horizontal spacing
            dpi="600"
        )
        
        g.attr(
            "node",
            shape="circle",
            style="filled",
            color="black",
            fontname="Helvetica",
            fontsize="12",
            width="0.35",
            height="0.35",            
            fixedsize="true"
        )

        g.attr(
            "edge",
            color="black",
            penwidth="1.2"
        )

        # nodes
        for i in range(N):
            g.node(str(i), label=node_labels[i], fillcolor=node_colors[i])

        # edges
        for (i, j) in edges:
            g.edge(str(i), str(j))

        png_path = g.render(filename=f"{out_prefix}_{name}", cleanup=True)
        # svg_path = g.render(filename=f"{out_prefix}_{name}", format="svg", cleanup=True)

        return png_path

    # layouts similar to before
    init_png = render_graphviz_graph("initial_graph", "neato", init_edges)
    pred_png = render_graphviz_graph("predicted_graph", "circo", pred_edges)

    # ---- load images ----
    init_img = mpimg.imread(init_png)
    pred_img = mpimg.imread(pred_png)

    # legend
    labels = [f"P{i}" for i in range(min(N, len(node_colors)))]
    legend_elements = [
        Patch(facecolor=node_colors[i], edgecolor="k", label=labels[i])
        for i in range(min(N, len(node_colors)))
    ]

    if img is not None:
        fig, axes = plt.subplots(
            1, 3, figsize=figsize,
            gridspec_kw={"width_ratios": [1.4, 1, 1]}
        )

        axes[0].imshow(img)
        axes[0].set_title("Input Point Cloud with Masks", fontsize=16, fontweight="bold")
        axes[0].legend(handles=legend_elements, loc="center left", bbox_to_anchor=(0.9, 0.5))
        axes[0].axis("off")

        axes[1].imshow(init_img)
        axes[1].set_title("Initial Kinematic Graph", fontsize=16, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(pred_img)
        axes[2].set_title("Kinematic Graph Prediction", fontsize=16, fontweight="bold")
        axes[2].axis("off")

    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={"width_ratios": [1, 1]}
                                 )

        axes[0].imshow(init_img)
        axes[0].set_title("Initial Kinematic Graph", fontsize=16, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(pred_img)
        axes[1].set_title("Kinematic Graph Prediction", fontsize=16, fontweight="bold")
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()
