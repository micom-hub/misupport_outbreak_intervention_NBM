import numpy as np
import pandas as pd
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.simulation.outbreak_model import NetworkModel


def get_known_network_snapshot(
    model: "NetworkModel", run_number: int, t: int, include_truth: bool = False
) -> Dict[str, Any]:
    """
    Extracts a snapshot of the LHD's knowledge (reported cases and discovered edges)
    as it existed at time t for a specific stochastic run. Optionally includes
    ground truth data for comparison.

    Args:
        model: The simulated NetworkModel object.
        run_number: The replicate index.
        t: The time threshold for extraction (0 to Tmax).
        include_truth: If True, includes nodes that are truly E, I, or R even if unknown.

    Returns:
        A dictionary containing lists of known nodes and edges.
    """
    if run_number >= len(model.all_surveillance_batches):
        raise ValueError(f"Run {run_number} not found in model results.")

    batches = model.all_surveillance_batches[run_number]

    # 0. Get ground truth states for all nodes at time t
    # states_over_time[t] = [S_list, E_list, I_list, R_list]
    true_state_map = {}
    if t < len(model.all_states_over_time[run_number]):
        S, E, I, R = model.all_states_over_time[run_number][t]
        for nid in E:
            true_state_map[int(nid)] = 1  # Exposed
        for nid in I:
            true_state_map[int(nid)] = 2  # Infectious
        for nid in R:
            true_state_map[int(nid)] = 3  # Recovered
        # Susceptibles are implicitly 0 or not in map

    # 1. Reconstruct knowledge from surveillance batches
    known_nodes = {}  # nid -> metadata
    known_edges = []
    edge_seen = set()

    for batch in batches:
        batch_t = int(batch.get("t", -1))
        if batch_t > t:
            break

        # Process reported cases
        rep_nodes = batch.get("reported_cases", np.empty(0))
        rep_stages = batch.get("reported_stage", np.empty(0))
        rep_is_vax = batch.get("is_vax", np.empty(0))
        rep_sources = batch.get("reported_sources", np.empty(0, dtype=object))

        for i in range(len(rep_nodes)):
            nid = int(rep_nodes[i])
            source = rep_sources[i] if len(rep_sources) > i else ""
            if nid not in known_nodes:
                known_nodes[nid] = {
                    "node_id": nid,
                    "report_time": batch_t,
                    "is_reported": True,
                    "is_passively_reported": (source == "passive"),
                    "is_tested_positive": (source == "tested_positive"),
                    "is_traced_contact": False,
                    "stage_at_report": int(rep_stages[i]) if len(rep_stages) > i else 0,
                    "is_vax": bool(rep_is_vax[i]) if len(rep_is_vax) > i else False,
                }
            else:
                known_nodes[nid]["is_reported"] = True
                known_nodes[nid]["is_passively_reported"] = source == "passive"
                known_nodes[nid]["is_tested_positive"] = source == "tested_positive"
                known_nodes[nid]["is_traced_contact"] = False

        # Keep track of all reported nodes to distinguish from traced-only nodes
        all_reported_nodes = set(known_nodes.keys())

        # Process trace edges
        src = batch.get("trace_src", np.empty(0, np.int32))
        tgt = batch.get("trace_tgt", np.empty(0, np.int32))
        ct = batch.get("trace_ct", np.empty(0, np.int16))

        for i in range(len(src)):
            u, v, c = int(src[i]), int(tgt[i]), int(ct[i])
            for nid in (u, v):
                # If this node is not already a reported case, mark it as a traced contact
                if nid not in all_reported_nodes and nid not in known_nodes:
                    known_nodes[nid] = {
                        "node_id": nid,
                        "report_time": -1,
                        "is_reported": False,
                        "is_passively_reported": False,
                        "is_tested_positive": False,
                        "is_traced_contact": True,
                        "stage_at_report": 0,
                        "is_vax": False,
                    }

            a, b = (u, v) if u < v else (v, u)
            key = (a, b, c)
            if key not in edge_seen:
                edge_seen.add(key)
                known_edges.append(
                    {
                        "u": u,
                        "v": v,
                        "ct_id": c,
                        "discovery_time": batch_t,
                        "source": "trace",
                    }
                )

    # 3. (Optional) Add hidden cases that exist in truth but aren't in LHD knowledge
    if include_truth:
        for nid, stage in true_state_map.items():
            if nid not in known_nodes:
                known_nodes[nid] = {
                    "node_id": nid,
                    "report_time": -1,
                    "is_reported": False,
                    "is_passively_reported": False,
                    "is_tested_positive": False,
                    "is_traced_contact": False,
                }

    # Attach true stage to all nodes found
    for nid, meta in known_nodes.items():
        meta["true_stage"] = true_state_map.get(nid, 0)

    return {"t": t, "nodes": list(known_nodes.values()), "edges": known_edges}


def get_knowledge_growth_df(model: "NetworkModel", run_number: int) -> pd.DataFrame:
    """
    Returns a timeseries DataFrame of the LHD's cumulative knowledge count.
    """
    end_t = model.all_end_days[run_number]
    history = []
    for t in range(end_t + 1):
        snap = get_known_network_snapshot(model, run_number, t)
        history.append(
            {
                "t": t,
                "cumulative_reported_cases": sum(
                    1 for n in snap["nodes"] if n["is_reported"]
                ),
                "cumulative_discovered_edges": len(snap["edges"]),
            }
        )
    return pd.DataFrame(history)


def build_lhd_igraph(model: "NetworkModel", run_number: int, t: int):
    """
    Converts the LHD's knowledge at time t into an igraph object for visualization.
    """
    import igraph as ig

    snap = get_known_network_snapshot(model, run_number, t)

    sorted_nodes = sorted(snap["nodes"], key=lambda x: x["node_id"])
    node_ids = [n["node_id"] for n in sorted_nodes]
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    g = ig.Graph(directed=True)
    g.add_vertices(len(node_ids))
    g.vs["name"] = [str(nid) for nid in node_ids]
    g.vs["is_reported"] = [bool(n.get("is_reported", False)) for n in sorted_nodes]
    g.vs["is_passively_reported"] = [
        n.get("is_passively_reported", False) for n in sorted_nodes
    ]
    g.vs["is_tested_positive"] = [
        n.get("is_tested_positive", False) for n in sorted_nodes
    ]
    g.vs["is_traced_contact"] = [
        n.get("is_traced_contact", False) for n in sorted_nodes
    ]

    edges = [(id_map[e["u"]], id_map[e["v"]]) for e in snap["edges"]]
    g.add_edges(edges)
    return g
