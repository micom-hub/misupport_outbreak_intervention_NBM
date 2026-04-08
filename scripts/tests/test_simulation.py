import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.sparse import csr_matrix

from scripts.config import ModelConfig
from scripts.simulation.outbreak_model import NetworkModel
from scripts.graph.graph_utils import GraphData
from scripts.graph.graph_utils import (
    build_minimal_graphdata_from_edge_list,
)  # Assuming this function exists and can be used


def make_small_graphdata():
    """
    Build a tiny test GraphData object for N=4 with a few edges and contact types.
    Returns GraphData instance.
    """
    N = 4
    # undirected edges (source, target, weight, contact_type)
    rows = [
        (0, 1, 1.0, "hh"),
        (1, 2, 1.0, "wp"),
        (2, 3, 1.0, "sch"),
    ]
    edge_df = pd.DataFrame(rows, columns=["source", "target", "weight", "contact_type"]).astype({"source": int, "target": int})

    # adjacency CSR (symmetric)
    src = edge_df["source"].to_numpy(dtype=np.int32)
    tgt = edge_df["target"].to_numpy(dtype=np.int32)
    wts = edge_df["weight"].to_numpy(dtype=np.float32)
    row = np.concatenate([src, tgt])
    col = np.concatenate([tgt, src])
    dat = np.concatenate([wts, wts])
    adj = csr_matrix((dat, (row, col)), shape=(N, N))

    # simple individual lookup: age and sex
    individual_lookup = pd.DataFrame({
        "age": [30, 6, 70, 40],
        "sex": ["M", "F", "M", "F"]
    })

    ages = individual_lookup["age"].to_numpy()
    sexes = individual_lookup["sex"].to_numpy()

    # build neighbor_map and fast_neighbor_map
    neighbor_map = {i: [] for i in range(N)}
    for _, r in edge_df.iterrows():
        s = int(r.source)
        t = int(r.target)
        w = float(r.weight)
        ct = str(r.contact_type)
        neighbor_map[s].append((t, w, ct))
        neighbor_map[t].append((s, w, ct))
    fast_neighbor_map = {src: {tgt: (w, ct) for (tgt, w, ct) in nbrs} for src, nbrs in neighbor_map.items()}

    # Use build_minimal_graphdata_from_edge_list to ensure consistency with production code
    # This function is assumed to exist in scripts.graph.graph_utils
    minimal_gd_from_util = build_minimal_graphdata_from_edge_list(edge_df, N=N)

    # Extract relevant parts from the utility-built GraphData
    csr_by_type = minimal_gd_from_util.csr_by_type
    contact_types = minimal_gd_from_util.contact_types
    ct_to_id = minimal_gd_from_util.ct_to_id
    id_to_ct = minimal_gd_from_util.id_to_ct
    full_node_list = minimal_gd_from_util.full_node_list
    degrees_arr = minimal_gd_from_util.degrees_arr

    # The utility function might also set compliances, but we'll use the explicit one here if needed
    compliances = np.ones(N, dtype=np.float32)  # Assuming default compliance

    # Build GraphData dataclass
    gd = GraphData(
        N=int(N),
        edge_list=edge_df,
        adj_matrix=adj,
        # individual_lookup, ages, sexes are explicitly defined and can be kept
        individual_lookup=individual_lookup,
        ages=ages,
        sexes=sexes,
        compliances=None,
        neighbor_map=neighbor_map,
        fast_neighbor_map=fast_neighbor_map,
        csr_by_type=csr_by_type,
        contact_types=contact_types,
        ct_to_id=ct_to_id,
        id_to_ct=id_to_ct,
        full_node_list=full_node_list,
        degrees_arr=degrees_arr,
    )
    return gd


def make_base_config():
    cfg = ModelConfig()
    # use small replicates and short duration for unit tests
    cfg = cfg.copy_with({"sim": {"n_replicates": 2, "simulation_duration": 5, "I0": 1, "seed": 123}})
    
    # reduce vaccines so exposures are easier to see
    cfg = cfg.copy_with({"epi": {"vax_uptake": 0.0, "vax_efficacy": 0.0}})
    return cfg


def test_initialize_states_int_count(tmp_path):
    gd = make_small_graphdata()
    cfg = make_base_config().copy_with({"sim": {"I0": 2, "n_replicates": 1}})
    run_dir = str(tmp_path / "run")
    model = NetworkModel(config=cfg, graphdata=gd, run_dir=run_dir, seed=42, lhd_register_defaults=False, lhd_algorithm_map={}, lhd_action_factory_map={})
    model._initialize_states(0)
    # I0 normalized to a list of length 2
    assert isinstance(model.I0, list)
    assert len(model.I0) == 2
    # state should have two infectious (I) entries
    assert int((model.state == 2).sum()) == 2


def test_initialize_states_list(tmp_path):
    gd = make_small_graphdata()
    cfg = make_base_config().copy_with({"sim": {"I0": [0, 2], "n_replicates": 1}})
    run_dir = str(tmp_path / "run2")
    model = NetworkModel(config=cfg, graphdata=gd, run_dir=run_dir, seed = 42, lhd_register_defaults=False, lhd_algorithm_map={}, lhd_action_factory_map={})
    model._initialize_states(0)
    assert isinstance(model.I0, list)
    assert set(model.I0) == {0, 2}
    assert (model.state[0] == 2) and (model.state[2] == 2)


def test_assign_periods_return_length():
    gd = make_small_graphdata()
    cfg = make_base_config()
    model = NetworkModel(config=cfg, graphdata=gd, run_dir=".", seed = 42, lhd_register_defaults=False, lhd_algorithm_map={}, lhd_action_factory_map={})
    inds = [0, 1, 2]
    model._initialize_states(0)
    inc = model.assign_incubation_period(inds)
    inf = model.assign_infectious_period(inds)
    assert len(inc) == len(inds)
    assert len(inf) == len(inds)
    assert np.all(np.isfinite(inc))
    assert np.all(np.isfinite(inf))


def test_determine_new_exposures_prob_one():
    # ensure that when base_prob==1 and weights 1, susceptible neighbors become exposed
    gd = make_small_graphdata()
    # set base transmission prob to 1 and disable vaccination
    cfg = make_base_config().copy_with({"epi": {"base_transmission_prob": 1.0, "vax_uptake": 0.0}, "sim": {"I0": [0], "n_replicates": 1}})
    run_dir = "."
    model = NetworkModel(config=cfg, graphdata=gd, run_dir=run_dir, seed=2, lhd_register_defaults=False, lhd_algorithm_map={}, lhd_action_factory_map={})
    model._initialize_states(0)
    newly = model.determine_new_exposures(recorder=None)
    # node 0 connected to node 1 by hh in our small graph -> 1 should be exposed
    assert 1 in newly.tolist()


def test_simulate_and_results_to_df(tmp_path):
    gd = make_small_graphdata()
    cfg = make_base_config().copy_with({"sim": {"I0": 1, "n_replicates": 2, "simulation_duration": 5, "seed": 7}})
    run_dir = str(tmp_path / "run_sim")
    model = NetworkModel(config=cfg, graphdata=gd, run_dir=run_dir, seed=7, lhd_register_defaults=False, lhd_algorithm_map={}, lhd_action_factory_map={})
    model.simulate()
    df = model.results_to_df(["peakPrev", "peakTime", "outbreakSize"])
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == cfg.sim.n_replicates
    assert set(df.columns) >= {"run_number", "peakPrev", "peakTime", "outbreakSize"}
