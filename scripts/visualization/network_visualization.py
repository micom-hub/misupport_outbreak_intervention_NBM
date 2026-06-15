import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import igraph as ig
from scripts.graph.lhd_network_extraction import (
    get_known_network_snapshot,
    build_lhd_igraph,
    get_knowledge_growth_df,
)


def plot_lhd_network_snapshot(
    model,
    run_number: int,
    t: int,
    ax=None,
    save_path: str = None,
    fixed_layout: Optional[ig.Layout] = None,
):
    """
    Plots a spatial representation of the LHD's known network at time t.
    Nodes are colored by their status:
    - Red: LHD Tested Positive Case
    - Purple: Passively Reported Case
    - Black: Case status unknown (traced contact)
    - Grey Edges: Traced but not a case
    - Red Edges: Traced and found to be a case
    """

    g = build_lhd_igraph(model, run_number, t)

    if g.vcount() == 0:
        if ax:
            ax.set_title(f"Day {t}: No LHD Knowledge")
            ax.axis("off")
        return

    # Color mapping
    colors = []
    for v in g.vs:  # Prioritize coloring based on reporting status
        if v["is_tested_positive"]:
            colors.append("#FF0000")  # Red: LHD Tested Positive
        elif v["is_passively_reported"]:
            colors.append("#800080")  # Purple: Passively Reported
        elif v["is_traced_contact"]:
            colors.append("#000000")  # Black: Unknown Status
        else:
            colors.append("#ADD8E6")  # Light Blue

    # Determine layout
    if fixed_layout:
        current_layout_coords = [fixed_layout[int(v["name"])] for v in g.vs]
        layout = ig.Layout(current_layout_coords)
    else:
        layout = g.layout("fr")

    edge_colors = []
    for e in g.es:
        u_idx, v_idx = e.tuple
        if g.vs[u_idx]["is_reported"] and g.vs[v_idx]["is_reported"]:
            edge_colors.append("#FF0000")  # Red: Successful Trace
        else:
            edge_colors.append("#D3D3D3")  # Grey: Baseline/Unsuccessful

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    ig.plot(
        g,
        target=ax,
        layout=layout,
        vertex_color=colors,
        vertex_size=12,
        edge_color=edge_colors,
        edge_arrow_size=0.5,
        edge_curved = 5,
        edge_width=0.6,
        vertex_label=g.vs["name"] if g.vcount() < 40 else None,
        vertex_label_size=7,
    )
    ax.set_title(
        f"LHD Knowledge Network - Day {t}\nRed=Tested+, Purple=Passive, Black=Unknown"
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()


def plot_comparison_snapshot(
    model,
    run_number: int,
    t: int,
    save_path: str = None,
    fixed_layout: Optional[ig.Layout] = None,
):
    """
    Creates a side-by-side comparison:
    Left: Ground Truth (S/E/I/R status)
    Right: LHD Knowledge (Reported/Isolated/Traced status)
    """
    # Build graph including truth nodes
    snap = get_known_network_snapshot(model, run_number, t, include_truth=True)

    # Convert snap to igraph
    sorted_nodes = sorted(snap["nodes"], key=lambda x: x["node_id"])
    node_ids = [n["node_id"] for n in sorted_nodes]
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    g = ig.Graph(directed=True)
    g.add_vertices(len(node_ids))
    g.vs["name"] = [str(nid) for nid in node_ids]
    g.vs["is_reported"] = [bool(n.get("is_reported", False)) for n in sorted_nodes]
    edges = [(id_map[e["u"]], id_map[e["v"]]) for e in snap["edges"]]
    g.add_edges(edges)

    # Determine layout
    if fixed_layout:
        current_layout_coords = [fixed_layout[int(v["name"])] for v in g.vs]
        layout = ig.Layout(current_layout_coords)
    else:
        layout = g.layout("fr")

    fig, (ax_true, ax_lhd) = plt.subplots(1, 2, figsize=(20, 10))

    # Left Plot: Truth
    # 0: Gray (S), 1: Yellow (E), 2: Red (I), 3: Green (R)
    true_colors = []
    for n in sorted_nodes:
        s = n.get("true_stage", 0)
        if s == 1:
            true_colors.append("#F1C40F")  # Exposed
        elif s == 2:
            true_colors.append("#E74C3C")  # Infectious
        elif s == 3:
            true_colors.append("#2ECC71")  # Recovered
        else:
            true_colors.append("#D3D3D3")  # Susceptible

    ig.plot(
        g,
        target=ax_true,
        layout=layout,
        vertex_color=true_colors,
        vertex_size=20,
        edge_color="#EEE",
        edge_arrow_size=0,
        vertex_frame_width=0,
    )
    ax_true.set_title(f"Ground Truth (Day {t})\nRed=Inf, Yellow=Exp, Green=Rec")
    ax_true.axis("off")

    # Right Plot: LHD View
    lhd_colors = []
    for n in sorted_nodes:  # Prioritize coloring based on reporting status
        if n.get("is_tested_positive", False):
            lhd_colors.append("#FF0000")  # Red: Tested Positive
        elif n.get("is_passively_reported", False):
            lhd_colors.append("#800080")  # Purple: Passively Reported
        elif n.get("is_traced_contact", False):
            lhd_colors.append("#000000")  # Black: Unknown Status
        else:
            lhd_colors.append("#ADD8E6")

    # Edge coloring for LHD view
    edge_colors_lhd = []
    for e in g.es:
        u_idx, v_idx = e.tuple
        if g.vs[u_idx]["is_reported"] and g.vs[v_idx]["is_reported"]:
            edge_colors_lhd.append("#FF0000")
        else:
            edge_colors_lhd.append("#D3D3D3")

    # Add special markers for reported nodes in LHD view
    # This logic can remain as is, as it highlights any reported case
    vertex_width = [2 if n.get("is_reported", False) else 0 for n in sorted_nodes]

    ig.plot(
        g,
        target=ax_lhd,
        layout=layout,
        vertex_color=lhd_colors,
        vertex_size=20,
        edge_color=edge_colors_lhd,
        edge_arrow_size=0.5,
        vertex_frame_width=vertex_width,
    )
    ax_lhd.set_title(
        f"LHD Operational View (Day {t})\nRed=Tested+, Purple=Passive, Black=Unknown"
    )
    ax_lhd.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_lhd_evolution(model, run_number: int = 0, output_dir: str = None):
    """
    Generates a suite of visualizations showing how the LHD builds its knowledge
    base and applies interventions over the course of the outbreak.
    """
    if output_dir is None:
        output_dir = os.path.join(
            model.results_folder, f"lhd_evolution_run_{run_number}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # 1. Knowledge Growth Curves
    # Compute a fixed layout for the entire network once
    full_g = ig.Graph(n=model.N, directed=False)
    full_g.add_edges(model.edge_list[["source", "target"]].values.tolist())
    fixed_layout = full_g.layout("fr")

    df_growth = get_knowledge_growth_df(model, run_number)
    if not df_growth.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(
            df_growth["t"],
            df_growth["cumulative_reported_cases"],
            label="Known Cases",
            color="red",
            lw=2,
        )
        plt.plot(
            df_growth["t"],
            df_growth["cumulative_discovered_edges"],
            label="Traced Edges",
            color="blue",
            ls="--",
        )
        plt.title(f"LHD Network Knowledge Accumulation (Run {run_number})")
        plt.xlabel("Day")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(output_dir, "knowledge_growth.png"))
        plt.close()

    # 2. Time-series Snapshots
    end_t = model.all_end_days[run_number]
    timepoints = np.linspace(0, end_t, 10, dtype=int)
    for t in timepoints:
        # Single view
        plot_lhd_network_snapshot(
            model,
            run_number,
            t,
            fixed_layout=fixed_layout,
            save_path=os.path.join(output_dir, f"lhd_view_t{t:03d}.png"),
        )

        # Comparison view
        plot_comparison_snapshot(
            model,
            run_number,
            t,
            fixed_layout=fixed_layout,
            save_path=os.path.join(output_dir, f"comparison_t{t:03d}.png"),
        )
