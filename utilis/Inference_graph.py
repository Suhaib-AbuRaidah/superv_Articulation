import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def visualize_articulated_graph(joint_conne_pred, adj, joint_types_pred, screw_axes_pred,
                                img,threshold=0.5, node_labels=None, figsize=(16, 10)):
    joint_conne_pred = joint_conne_pred.detach().cpu().numpy()
    N = joint_conne_pred.shape[0]

    edges = []
    edge_labels = {}
    edge_count = 0

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G1 = nx.Graph()
    G1.add_nodes_from(range(N))

    if node_labels is None:
        node_labels = [f"P{i}" for i in range(N)]

    for i in range(N):
        for j in range(i+1, N):
            if adj[i, j] > threshold:
                G1.add_edge(i, j)


    for i in range(N):
        for j in range(i+1, N):
            if joint_conne_pred[i, j] > threshold:
                G.add_edge(i, j)
                joint_type = joint_types_pred[edge_count].item()
                screw_axis = screw_axes_pred[edge_count].detach().cpu().numpy()
                jt_label = "P" if joint_type >= 0.5 else "R"
                axis_str = np.array2string(screw_axis, precision=2, suppress_small=True, separator=", ")
                edge_labels[(i, j)] = f"{jt_label}\n{axis_str}"
                edge_count += 1

    colors = [
        (1, 0, 0),      # red
        (0, 1, 0),      # green
        (0, 0, 1),      # blue
        (1, 1, 0),      # yellow
        (1, 0, 1),      # magenta
        (0, 1, 1),      # cyan
        (0.5, 0.5, 0.5), # gray
        (0.8, 0.5, 0.2),  # orange
        (0.6, 0.2, 0.8),  # purple
        (0.2, 0.8, 0.5)  # teal
    ]
    labels = [f"P{i}" for i in range(len(colors))]

    # Create legend handles
    legend_elements = [Patch(facecolor=colors[i], edgecolor='k', label=labels[i]) for i in range(N)]
    pos = nx.circular_layout(G)
    pos1 = nx.spring_layout(G1, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1, 2]})

    # --- Left: Point Cloud ---
    axes[0].imshow(img)
    axes[0].set_title("Input Point Cloud with Masks", fontsize=16, fontweight="bold")
    axes[0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize=12)
    axes[0].axis("off")

    # --- Middle: Ground Truth Graph ---
    nx.draw(G1, pos1, with_labels=True, labels={i: node_labels[i] for i in range(N)},
            node_size=800, node_color="lightblue", font_size=12, font_weight="bold", ax=axes[1])

    axes[1].set_title("Initial Kinematic Graph", fontsize=16, fontweight="bold")
    axes[1].axis("off")

    # --- Right: Predicted Graph ---
    nx.draw(G, pos, with_labels=True, labels={i: node_labels[i] for i in range(N)},
            node_size=800, node_color="lightblue", font_size=14, font_weight="bold", ax=axes[2])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                font_color="darkred", font_size=12, font_weight="bold", ax=axes[2])
    axes[2].set_title("Kinematic Graph with Articulation Prediction", fontsize=16, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("articulated_graph.svg")
    plt.show()
