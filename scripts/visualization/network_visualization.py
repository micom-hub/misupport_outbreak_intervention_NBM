import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import igraph as ig
import iplotx as ipx
import imageio
import tempfile
import shutil
from scripts.graph.lhd_network_extraction import (
    get_known_network_snapshot,
    build_lhd_igraph,
    get_knowledge_growth_df,
)
from typing import Dict, Any, List, Callable
import datetime
import numbers


def _layout_coords_from_fixed(fixed_layout, g):
    # fixed_layout: igraph Layout object where index==node_id
    # g: igraph Graph whose v["name"] holds the node_id (string)
    return [tuple(fixed_layout[int(v["name"])]) for v in g.vs]


def _layout_coords_auto(g):
    # igraph Layout converts to an iterable of coordinates
    L = g.layout("fr")
    return [tuple(L[i]) for i in range(g.vcount())]


def plot_lhd_network_snapshot(
    model,
    run_number: int,
    t: int,
    ax=None,
    save_path: str = None,
    fixed_layout: Optional[ig.Layout] = None,
    inactive_days=7,
    remove_recovered=True,
    remove_susceptible=False,
    prune_without_time=False,
    node_filter=None,
):
    g = build_lhd_igraph(model, run_number, t)

    # Optionally prune recovered/susceptible vertices from the igraph graph
    if g.vcount() > 0:
        _prune_igraph_vertices(
            g,
            t,
            model,
            remove_recovered=remove_recovered,
            remove_susceptible=remove_susceptible,
            inactive_days=inactive_days,
            prune_without_time=prune_without_time,
            node_filter=node_filter,
        )

    if g.vcount() == 0:
        if ax:
            ax.set_title(f"Day {t}: No LHD Knowledge")
            ax.axis("off")
        return

    # Per-node facecolors
    facecolors = []
    for v in g.vs:
        if v["is_tested_positive"]:
            facecolors.append("#FF0000")  # red
        elif v["is_passively_reported"]:
            facecolors.append("#800080")  # purple
        elif v["is_traced_contact"]:
            facecolors.append("#000000")  # black
        else:
            facecolors.append("#ADD8E6")  # light blue

    # Per-edge colors
    edge_colors = []
    for e in g.es:
        u_idx, v_idx = e.tuple
        if g.vs[u_idx]["is_reported"] and g.vs[v_idx]["is_reported"]:
            edge_colors.append("#FF0000")  # red
        else:
            edge_colors.append("#D3D3D3")  # grey baseline

    # Labels (iplotx will pick up igraph vertex attribute "label" if set)
    if g.vcount() < 40:
        labels = [v["name"] for v in g.vs]
        g.vs["label"] = labels
    else:
        g.vs["label"] = [""] * g.vcount()

    # Layout coords for ipx
    if fixed_layout is not None:
        layout_coords = _layout_coords_from_fixed(fixed_layout, g)
    else:
        layout_coords = _layout_coords_auto(g)

    # Vertex border widths: highlight reported nodes
    vertex_edgewidth = [2.0 if v["is_reported"] else 0.0 for v in g.vs]
    vertex_edgecolor = ["#000000" if w > 0 else "#00000000" for w in vertex_edgewidth]

    # Build iplotx style dict
    style = {
        "vertex": {
            "size": 12,  # points
            "facecolor": facecolors,  # per-vertex list OK
            "edgecolor": vertex_edgecolor,  # per-vertex border color
            "linewidth": vertex_edgewidth,  # per-vertex border width
            "label": {
                "color": "#000000",
                "fontsize": 7,
            },
        },
        "edge": {
            "color": edge_colors,  # per-edge list
            "linewidth": 0.6,
            "curved": True,
            "tension": 5.0,  # maps to igraph's "edge_curved"
        },
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    ipx.plot(g, ax=ax, layout=layout_coords, style=style)
    ax.set_title(
        f"LHD Knowledge Network - Day {t}\nRed=Tested+, Purple=Passive, Black=Unknown"
    )
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()


def plot_overlay_snapshot(
    model,
    run_number: int,
    t: int,
    ax=None,
    save_path: str = None,
    fixed_layout: Optional[ig.Layout] = None,
    inactive_days: int = 7,
    remove_recovered: bool = True,
    remove_susceptible: bool = True,
    prune_without_time: bool = False,
    node_filter: Optional[Callable[[Dict[str, Any], int, Any], bool]] = None,
):
    """
    Plots an overlay of the true epidemiological state and the LHD knowledge.
    - Vertex facecolors: True SEIR status.
    - Vertex markers: Diamonds for LHD-tested positives, circles otherwise.
    - Vertex edge: Thick black for actively isolated individuals.
    - Edge color/style: Red/Solid for perceived transmission, Grey/Dotted for traced contacts.
    """

    snap = get_known_network_snapshot(model, run_number, t, include_truth=True)

    snap = _prune_snapshot_nodes(
        snap,
        t,
        model,
        remove_recovered=remove_recovered,
        remove_susceptible=remove_susceptible,
        inactive_days=inactive_days,
        prune_without_time=prune_without_time,
        node_filter=node_filter,
    )

    lhd = model.lhd

    sorted_nodes = sorted(snap["nodes"], key=lambda x: x["node_id"])
    node_ids = [n["node_id"] for n in sorted_nodes]
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    g = ig.Graph(directed=False)
    g.add_vertices(len(node_ids))
    g.vs["name"] = [str(nid) for nid in node_ids]

    # Vertex attributes for styling
    facecolors = []
    markers = []
    edgecolors = []
    linewidths = []

    for n in sorted_nodes:
        nid = n["node_id"]
        # 1. Facecolor based on Truth (SEIR)
        s = n.get("true_stage", 0)
        if s == 1:
            facecolors.append("#F1C40F")  # Exposed (yellow)
        elif s == 2:
            facecolors.append("#E74C3C")  # Infectious (red)
        elif s == 3:
            facecolors.append("#2ECC71")  # Recovered (green)
        else:
            facecolors.append("#D3D3D3")  # Susceptible (grey)

        # 2. Marker based on LHD tested positive
        if n.get("is_tested_positive", False):
            markers.append("d")  # Diamond
        else:
            markers.append("o")  # Circle

        # 3. Border based on active isolation
        if lhd.state.isolated_until[nid] > t:
            edgecolors.append("#000000")
            linewidths.append(2.5)
        else:
            edgecolors.append("#00000000")
            linewidths.append(0.0)

    # Edges
    edges = []
    edge_colors = []
    edge_styles = []
    reported_ids = {n["node_id"] for n in sorted_nodes if n.get("is_reported", False)}

    for e in snap["edges"]:
        u, v = e["u"], e["v"]
        edges.append((id_map[u], id_map[v]))

        # Red/Solid if perceived transmission (both endpoints reported)
        if u in reported_ids and v in reported_ids:
            edge_colors.append("#FF0000")
            edge_styles.append("-")
        else:
            edge_colors.append("#A9A9A9")  # Grey
            edge_styles.append("--")  # Dotted/Dashed

    g.add_edges(edges)

    # Layout
    if fixed_layout is not None:
        layout_coords = _layout_coords_from_fixed(fixed_layout, g)
    else:
        layout_coords = _layout_coords_auto(g)

    style = {
        "vertex": {
            "size": 10,
            "facecolor": facecolors,
            "edgecolor": edgecolors,
            "linewidth": linewidths,
            "marker": markers,
        },
        "edge": {
            "color": edge_colors,
            "linewidth": 1.0,
            "linestyle": edge_styles,
        },
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    ipx.plot(g, ax=ax, layout=layout_coords, style=style)
    ax.set_title(
        f"LHD/Truth Overlay - Day {t}\nDiamonds=Tested+, Thick Border=Isolated"
    )
    ax.axis("off")

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
    snap = get_known_network_snapshot(model, run_number, t, include_truth=True)

    sorted_nodes = sorted(snap["nodes"], key=lambda x: x["node_id"])
    node_ids = [n["node_id"] for n in sorted_nodes]
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    g = ig.Graph(directed=False)
    g.add_vertices(len(node_ids))
    g.vs["name"] = [str(nid) for nid in node_ids]
    g.vs["is_reported"] = [bool(n.get("is_reported", False)) for n in sorted_nodes]
    edges = [(id_map[e["u"]], id_map[e["v"]]) for e in snap["edges"]]
    g.add_edges(edges)

    # Labels
    if g.vcount() < 40:
        g.vs["label"] = [v["name"] for v in g.vs]
    else:
        g.vs["label"] = [""] * g.vcount()

    # Layout coords
    if fixed_layout is not None:
        layout_coords = _layout_coords_from_fixed(fixed_layout, g)
    else:
        layout_coords = _layout_coords_auto(g)

    fig, (ax_true, ax_lhd) = plt.subplots(1, 2, figsize=(20, 10))

    # Left plot: ground truth colors
    true_colors = []
    for n in sorted_nodes:
        s = n.get("true_stage", 0)
        if s == 1:
            true_colors.append("#F1C40F")  # Exposed (yellow)
        elif s == 2:
            true_colors.append("#E74C3C")  # Infectious (red)
        elif s == 3:
            true_colors.append("#2ECC71")  # Recovered (green)
        else:
            true_colors.append("#D3D3D3")  # Susceptible (grey)

    g_true = g.as_undirected() if g.is_directed() else g
    style_true = {
        "vertex": {
            "size": 20,
            "facecolor": true_colors,
            "edgecolor": "#00000000",
            "label": {"color": "#000000", "fontsize": 8},
        },
        "edge": {
            "color": "#EEEEEE",
            "linewidth": 0.6,
        },
    }

    ipx.plot(g_true, ax=ax_true, layout=layout_coords, style=style_true)

    ipx.plot(g, ax=ax_true, layout=layout_coords, style=style_true)
    ax_true.set_title(f"Ground Truth (Day {t})\nRed=Inf, Yellow=Exp, Green=Rec")
    ax_true.axis("off")

    # Right plot: LHD view
    lhd_colors = []
    for n in sorted_nodes:
        if n.get("is_tested_positive", False):
            lhd_colors.append("#FF0000")
        elif n.get("is_passively_reported", False):
            lhd_colors.append("#800080")
        elif n.get("is_traced_contact", False):
            lhd_colors.append("#000000")
        else:
            lhd_colors.append("#ADD8E6")

    edge_colors_lhd = []
    for e in g.es:
        u_idx, v_idx = e.tuple
        if g.vs[u_idx]["is_reported"] and g.vs[v_idx]["is_reported"]:
            edge_colors_lhd.append("#FF0000")
        else:
            edge_colors_lhd.append("#D3D3D3")

    vertex_edgewidth = [
        2.0 if n.get("is_reported", False) else 0.0 for n in sorted_nodes
    ]
    vertex_edgecolor = ["#000000" if w > 0 else "#00000000" for w in vertex_edgewidth]

    style_lhd = {
        "vertex": {
            "size": 20,
            "facecolor": lhd_colors,
            "edgecolor": vertex_edgecolor,
            "linewidth": vertex_edgewidth,
            "label": {"color": "#000000", "fontsize": 8},
        },
        "edge": {
            "color": edge_colors_lhd,
            "linewidth": 0.8,
            "curved": False,
        },
    }

    ipx.plot(g, ax=ax_lhd, layout=layout_coords, style=style_lhd)
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


def visualize_lhd_evolution(
    model,
    run_number: int = 0,
    plots=10,
    output_dir: str = None,
    inactive_days: int = 7,
    remove_recovered: bool = True,
    remove_susceptible: bool = True,
    prune_without_time: bool = False,
    node_filter: Optional[Callable[[Dict[str, Any], int, Any], bool]] = None,
):
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
    timepoints = np.linspace(0, end_t, plots, dtype=int)
    for t in timepoints:
        # Single view
        plot_lhd_network_snapshot(
            model,
            run_number,
            t,
            fixed_layout=fixed_layout,
            save_path=os.path.join(output_dir, f"lhd_view_t{t:03d}.png"),
            inactive_days=inactive_days,
            remove_recovered=remove_recovered,
            remove_susceptible=remove_susceptible,
            prune_without_time=prune_without_time,
            node_filter=node_filter,
        )

        # Comparison view
        plot_comparison_snapshot(
            model,
            run_number,
            t,
            fixed_layout=fixed_layout,
            save_path=os.path.join(output_dir, f"comparison_t{t:03d}.png"),
            inactive_days=inactive_days,
            remove_recovered=remove_recovered,
            remove_susceptible=remove_susceptible,
            prune_without_time=prune_without_time,
            node_filter=node_filter,
        )

        # Overlay view
        plot_overlay_snapshot(
            model,
            run_number,
            t,
            ax=None,
            save_path=os.path.join(output_dir, f"overlay_t{t:03d}.png"),
            fixed_layout=fixed_layout,
            inactive_days=inactive_days,
            remove_recovered=remove_recovered,
            remove_susceptible=remove_susceptible,
            prune_without_time=prune_without_time,
            node_filter=node_filter,
        )


def lhd_evolution_movie(
    model,
    run_number: int = 0,
    frames: int = 100,
    output_dir: str = None,
    inactive_days: int = 7,
    remove_recovered: bool = True,
    remove_susceptible: bool = True,
    prune_without_time: bool = False,
    node_filter: Optional[Callable[[Dict[str, Any], int, Any], bool]] = None,
):
    """
    Generates a video showing the evolution of LHD knowledge and true epidemiological
    state over the course of the outbreak. The video includes comparison and overlay
    snapshots.

    Args:
        model: The simulation model object.
        run_number: The specific run number to visualize.
        frames: The number of frames to include in the video.
        output_dir: Optional directory to save the video files. If None, a default
                    path within model.results_folder will be used.
    """
    if output_dir is None:
        output_dir = os.path.join(
            model.results_folder, f"network_evolution_run_{run_number}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Create a temporary directory for storing individual frames
    temp_frame_dir = tempfile.mkdtemp()
    print(f"Saving temporary frames to: {temp_frame_dir}")

    # Compute a fixed layout for the entire network once for consistent node positioning
    full_g = ig.Graph(n=model.N, directed=False)
    full_g.add_edges(model.edge_list[["source", "target"]].values.tolist())
    fixed_layout = full_g.layout("fr")

    end_t = model.all_end_days[run_number]
    timepoints = np.linspace(0, end_t, frames, dtype=int)

    comparison_frame_paths = []
    overlay_frame_paths = []

    tenth = -(len(timepoints) // -10)
    for i, t in enumerate(timepoints):
        # print progress every 10%
        if t % tenth == 0:
            print(f"Generating frame {i+1}/{frames} for day {t}...")
        comparison_frame_path = os.path.join(
            temp_frame_dir, f"comparison_frame_{i:04d}.png"
        )
        plot_comparison_snapshot(
            model,
            run_number,
            t,
            save_path=comparison_frame_path,
            fixed_layout=fixed_layout,
        )
        comparison_frame_paths.append(comparison_frame_path)

        overlay_frame_path = os.path.join(temp_frame_dir, f"overlay_frame_{i:04d}.png")
        plot_overlay_snapshot(
            model,
            run_number,
            t,
            save_path=overlay_frame_path,
            fixed_layout=fixed_layout,
        )
        overlay_frame_paths.append(overlay_frame_path)

    # Create videos from the generated frames
    comparison_video_path = os.path.join(output_dir, "comparison_evolution.mp4")
    overlay_video_path = os.path.join(output_dir, "overlay_evolution.mp4")

    print(f"Creating comparison video: {comparison_video_path}")
    imageio.mimsave(
        comparison_video_path,
        [imageio.imread(f) for f in comparison_frame_paths],
        fps=2,
    )  # fps can be adjusted

    print(f"Creating overlay video: {overlay_video_path}")
    imageio.mimsave(
        overlay_video_path, [imageio.imread(f) for f in overlay_frame_paths], fps=1
    )  # fps can be adjusted

    # Clean up temporary frames directory
    shutil.rmtree(temp_frame_dir)
    print(f"Cleaned up temporary frame directory: {temp_frame_dir}")

    print("Video generation complete.")


# Helper Funcs
def _extract_stage_time_from_node(node: Dict[str, Any], stage: int) -> Optional[float]:
    """
    Try to find a numeric timestamp (in days) when `node` entered `stage`.
    Returns a float day or None if not found / not numeric.

    Looks for a variety of common key names used in snapshots and model exports.
    """
    # candidate keys for recovered (3) and susceptible (0)
    rec_keys = [
        "recovered_at",
        "t_recovered",
        "time_recovered",
        "recovery_time",
        "t_recovery",
        "time_of_recovery",
    ]
    sus_keys = [
        "became_susceptible_at",
        "susceptible_since",
        "time_susceptible",
        "t_became_susceptible",
        "susceptible_time",
    ]
    generic_keys = [
        "stage_since",
        "stage_time",
        "true_stage_time",
        "time_stage_entered",
        "stage_entered_at",
        "time_of_stage_change",
        "last_state_change_time",
        "last_change_time",
        "time_of_transition",
    ]

    keys: List[str] = []
    if stage == 3:
        keys += rec_keys
    elif stage == 0:
        keys += sus_keys
    keys += generic_keys

    # Try explicit keys first
    for k in keys:
        if k in node and node[k] is not None:
            val = node[k]
            # numeric types
            if isinstance(val, numbers.Number):
                return float(val)
            # numpy numeric
            try:
                import numpy as _np

                if hasattr(val, "dtype") and _np.issubdtype(val.dtype, _np.number):
                    return float(val)
            except Exception:
                pass
            # strings that can be converted to float
            if isinstance(val, str):
                try:
                    return float(val)
                except Exception:
                    pass
            # datetime-like -> can't convert without knowing model epoch, skip
            if isinstance(val, (datetime.date, datetime.datetime)):
                # we don't attempt to convert datetimes without context; return None
                return None
            # list/tuple where first element is numeric
            if (
                isinstance(val, (list, tuple))
                and len(val) > 0
                and isinstance(val[0], numbers.Number)
            ):
                return float(val[0])

    # Try structured dicts mapping stage->time
    for candidate in [
        "stage_times",
        "stage_time",
        "times",
        "time_by_stage",
        "t_by_stage",
        "time_in_stages",
    ]:
        st = node.get(candidate)
        if isinstance(st, dict):
            if stage in st and isinstance(st[stage], numbers.Number):
                return float(st[stage])
            if str(stage) in st and isinstance(st[str(stage)], numbers.Number):
                return float(st[str(stage)])

    # Try a history list of events (reverse search for last matching stage)
    for candidate in ["history", "true_history", "events", "state_history"]:
        hist = node.get(candidate)
        if isinstance(hist, list) and len(hist) > 0:
            for evt in reversed(hist):
                if not isinstance(evt, dict):
                    continue
                evt_stage = (
                    evt.get("stage") or evt.get("state") or evt.get("true_stage")
                )
                evt_time = evt.get("t") or evt.get("time") or evt.get("day")
                if evt_stage is None or evt_time is None:
                    continue
                if evt_stage == stage or (
                    isinstance(evt_stage, str) and str(evt_stage) == str(stage)
                ):
                    if isinstance(evt_time, numbers.Number):
                        return float(evt_time)
                    if isinstance(evt_time, str):
                        try:
                            return float(evt_time)
                        except Exception:
                            pass

    return None


def _prune_snapshot_nodes(
    snap: Dict[str, Any],
    t: int,
    model: Any,
    remove_recovered: bool = True,
    remove_susceptible: bool = True,
    inactive_days: int = 7,
    prune_without_time: bool = False,
    node_filter: Optional[Callable[[Dict[str, Any], int, Any], bool]] = None,
) -> Dict[str, Any]:
    """
    Given a 'snap' dict from get_known_network_snapshot, return a shallow-copied
    snap with nodes/edges removed according to stage/time rules.

    node_filter, if provided, is node_filter(node_dict, t, model) -> bool (True=keep).
    """
    nodes = list(snap.get("nodes", []))
    edges = list(snap.get("edges", []))

    keep_ids = set()
    for node in nodes:
        nid = node.get("node_id")
        keep = True
        if node_filter is not None:
            try:
                keep = bool(node_filter(node, t, model))
            except Exception:
                # if user's filter errors, conservatively keep the node
                keep = True
        else:
            stage = node.get("true_stage", node.get("stage", None))
            if stage == 3 and remove_recovered:
                ts = _extract_stage_time_from_node(node, 3)
                if ts is None:
                    keep = not prune_without_time
                else:
                    try:
                        keep = (float(t) - float(ts)) < float(inactive_days)
                    except Exception:
                        keep = True
            elif stage == 0 and remove_susceptible:
                ts = _extract_stage_time_from_node(node, 0)
                if ts is None:
                    keep = not prune_without_time
                else:
                    try:
                        keep = (float(t) - float(ts)) < float(inactive_days)
                    except Exception:
                        keep = True
            else:
                keep = True
        if keep and nid is not None:
            keep_ids.add(nid)

    new_nodes = [n for n in nodes if n.get("node_id") in keep_ids]
    new_edges = [
        e for e in edges if (e.get("u") in keep_ids and e.get("v") in keep_ids)
    ]

    new_snap = dict(snap)  # shallow copy other keys
    new_snap["nodes"] = new_nodes
    new_snap["edges"] = new_edges
    return new_snap


def _prune_igraph_vertices(
    g: ig.Graph,
    t: int,
    model: Any,
    remove_recovered: bool = True,
    remove_susceptible: bool = True,
    inactive_days: int = 7,
    prune_without_time: bool = False,
    node_filter: Optional[Callable[[Dict[str, Any], int, Any], bool]] = None,
):
    """
    Remove vertices from igraph Graph `g` in-place that satisfy pruning criteria.
    Vertex attributes are examined for the same keys looked for by _extract_stage_time_from_node.
    """
    to_delete_idx = []
    # get attribute names once
    attr_names = g.vs.attribute_names()
    for v in g.vs:
        # build a small dict of attributes for this vertex
        attr: Dict[str, Any] = {}
        for name in attr_names:
            try:
                attr[name] = v[name]
            except Exception:
                # if attribute access fails, skip it
                pass

        # allow user-provided callable
        keep = True
        if node_filter is not None:
            try:
                keep = bool(node_filter(attr, t, model))
            except Exception:
                keep = True
        else:
            stage = attr.get("true_stage", attr.get("stage", None))
            if stage == 3 and remove_recovered:
                ts = _extract_stage_time_from_node(attr, 3)
                if ts is None:
                    keep = not prune_without_time
                else:
                    try:
                        keep = (float(t) - float(ts)) < float(inactive_days)
                    except Exception:
                        keep = True
            elif stage == 0 and remove_susceptible:
                ts = _extract_stage_time_from_node(attr, 0)
                if ts is None:
                    keep = not prune_without_time
                else:
                    try:
                        keep = (float(t) - float(ts)) < float(inactive_days)
                    except Exception:
                        keep = True
            else:
                keep = True

        if not keep:
            to_delete_idx.append(v.index)

    if to_delete_idx:
        # delete in descending order to be safe, but igraph.delete_vertices handles a list
        g.delete_vertices(sorted(to_delete_idx))
