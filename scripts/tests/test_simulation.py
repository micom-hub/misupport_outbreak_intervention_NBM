import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.sparse import csr_matrix

from scripts.config import ModelConfig
from scripts.simulation.outbreak_model import NetworkModel
from scripts.graph.graph_utils import GraphData


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

    # Build csr_by_type similar to build_minimal_graphdata_from_edge_list
    # For each ct, create indptr/indices/weights arrays of directed neighbors
    all_cts = edge_df["contact_type"].unique().tolist()
    csr_by_type = {}
    for ct in all_cts:
        # collect directed edges of this ct
        rows_ct = []
        for src_node in range(N):
            for (tgt, w, c) in neighbor_map[src_node]:
                if c == ct:
                    rows_ct.append((src_node, tgt, w))
        if len(rows_ct) == 0:
            indptr = np.zeros(N + 1, dtype=np.int64)
            indices = np.empty(0, dtype=np.int32)
            weights = np.empty(0, dtype=np.float32)
        else:
            rows_ct = np.array(rows_ct, dtype=object)
            srcs = np.array([r[0] for r in rows_ct], dtype=np.int32)
            tgts = np.array([r[1] for r in rows_ct], dtype=np.int32)
            wvec = np.array([r[2] for r in rows_ct], dtype=np.float32)
            # sort by src to create indptr
            order = np.argsort(srcs, kind="mergesort")
            srcs_s = srcs[order]
            tgts_s = tgts[order]
            wts_s = wvec[order]
            counts = np.bincount(srcs_s, minlength=N)
            indptr = np.empty(N + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])
            indices = tgts_s.astype(np.int32, copy=True)
            weights = wts_s.astype(np.float32, copy=True)
        csr_by_type[str(ct)] = (indptr, indices, weights)

    contact_types = sorted(list(csr_by_type.keys()))
    ct_to_id = {ct: i for i, ct in enumerate(contact_types)}
    id_to_ct = {i: ct for ct, i in ct_to_id.items()}
    full_node_list = list(range(N))
    degrees_arr = adj.getnnz(axis=1).tolist()

    # Build GraphData dataclass
    gd = GraphData(
        N=int(N),
        edge_list=edge_df,
        adj_matrix=adj,
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
    model = NetworkModel(config=cfg, graphdata=gd, run_dir=run_dir, rng=np.random.default_rng(42), lhd_register_defaults=False, lhd_algorithm_map={}, lhd_action_factory_map={})
    model._initialize_states()
    # I0 normalized to a list of length 2
    assert isinstance(model.I0, list)
    assert len(model.I0) == 2
    # state should have two infectious (I) entries
    assert int((model.state == 2).sum()) == 2


def test_initialize_states_list(tmp_path):
    gd = make_small_graphdata()
    cfg = make_base_config().copy_with({"sim": {"I0": [0, 2], "n_replicates": 1}})
    run_dir = str(tmp_path / "run2")
    model = NetworkModel(config=cfg, graphdata=gd, run_dir=run_dir, rng=np.random.default_rng(123), lhd_register_defaults=False, lhd_algorithm_map={}, lhd_action_factory_map={})
    model._initialize_states()
    assert isinstance(model.I0, list)
    assert set(model.I0) == {0, 2}
    assert (model.state[0] == 2) and (model.state[2] == 2)


def test_assign_periods_return_length():
    gd = make_small_graphdata()
    cfg = make_base_config()
    model = NetworkModel(config=cfg, graphdata=gd, run_dir=".", rng=np.random.default_rng(1), lhd_register_defaults=False, lhd_algorithm_map={}, lhd_action_factory_map={})
    inds = [0, 1, 2]
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
    model = NetworkModel(config=cfg, graphdata=gd, run_dir=run_dir, rng=np.random.default_rng(2), lhd_register_defaults=False, lhd_algorithm_map={}, lhd_action_factory_map={})
    model._initialize_states()
    newly = model.determine_new_exposures(recorder=None)
    # node 0 connected to node 1 by hh in our small graph -> 1 should be exposed
    assert 1 in newly.tolist()


def test_simulate_and_results_to_df(tmp_path):
    gd = make_small_graphdata()
    cfg = make_base_config().copy_with({"sim": {"I0": 1, "n_replicates": 2, "simulation_duration": 5, "seed": 7}})
    run_dir = str(tmp_path / "run_sim")
    model = NetworkModel(config=cfg, graphdata=gd, run_dir=run_dir, rng=np.random.default_rng(7), lhd_register_defaults=False, lhd_algorithm_map={}, lhd_action_factory_map={})
    model.simulate()
    df = model.results_to_df(["peakPrev", "peakTime", "outbreakSize"])
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == cfg.sim.n_replicates
    assert set(df.columns) >= {"run_number", "peakPrev", "peakTime", "outbreakSize"}