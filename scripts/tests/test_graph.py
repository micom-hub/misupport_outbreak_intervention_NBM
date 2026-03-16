import numpy as np
import pandas as pd
import pytest

from scripts.graph.graph_utils import (
    build_minimal_graphdata_from_edge_list,
    build_graph_data,
    sample_from_master_graphdata
)
from scripts.config import ModelConfig


def make_toy_contacts_and_edges():
    """
    Build a small toy population (6 people) and an edge list covering hh/wp/sch/gq/cas types.
    Returns: contacts_df, edge_list_df, N
    """
    N = 6
    contacts = pd.DataFrame([
        {"PID": "0", "hh_id": "H1", "wp_id": "W1", "sch_id": "S1", "gq": False, "age": 34, "sex": "M"},
        {"PID": "1", "hh_id": "H1", "wp_id": "W1", "sch_id": "S1", "gq": False, "age": 6,  "sex": "F"},
        {"PID": "2", "hh_id": "H2", "wp_id": "W2", "sch_id": "X",  "gq": False, "age": 70, "sex": "M"},
        {"PID": "3", "hh_id": "H3", "wp_id": "X",  "sch_id": "X",  "gq": False, "age": 25, "sex": "F"},
        {"PID": "4", "hh_id": "H4", "wp_id": "W1", "sch_id": "S1", "gq": False, "age": 20, "sex": "M"},
        {"PID": "5", "hh_id": "H5", "wp_id": "X",  "sch_id": "X",  "gq": True,  "age": 50, "sex": "F"},
    ])

    # edges chosen to exercise multiple contact types
    rows = [
        (0, 1, 1.0, "hh"),   # household between 0 and 1
        (1, 2, 0.8, "wp"),   # workplace connecting 1 and 2
        (0, 4, 0.6, "sch"),  # school contact 0-4
        (3, 5, 0.9, "gq"),   # group quarter 3-5
        (2, 4, 0.2, "cas"),  # casual 2-4
    ]
    edge_df = pd.DataFrame(rows, columns=["source", "target", "weight", "contact_type"])
    edge_df = edge_df.astype({"source": int, "target": int, "weight": float, "contact_type": str})
    return contacts, edge_df, N


def test_build_minimal_graphdata_and_csrs():
    contacts, edges, N = make_toy_contacts_and_edges()
    minimal = build_minimal_graphdata_from_edge_list(edges, N=N)

    assert "csr_by_type" in minimal
    csr_by_type = minimal["csr_by_type"]
    # expected contact types
    expected_cts = set(edges["contact_type"].unique().tolist())
    assert set(csr_by_type.keys()) == expected_cts

    # For each contact type, check indptr length N+1
    for ct, (indptr, indices, weights) in csr_by_type.items():
        indptr = np.asarray(indptr)
        assert indptr.ndim == 1
        assert indptr.size == N + 1
        # indices length equals indptr[-1]
        assert indices.size == int(indptr[-1])


def test_build_graph_data_basic(tmp_path):
    contacts, edges, N = make_toy_contacts_and_edges()
    cfg = ModelConfig()
    # disable vaccination to make exposures easier to reason about if needed
    cfg = cfg.copy_with({"epi": {"vax_uptake": 0.0, "vax_efficacy": 0.0}})

    gd = build_graph_data(edge_list=edges, contacts_df=contacts, config=cfg, seed=1, N=N)

    # GraphData fields
    assert gd.N == N
    assert hasattr(gd, "adj_matrix")
    adj = gd.adj_matrix
    # adjacency symmetry and weights present
    for _, r in edges.iterrows():
        s = int(r.source)
        t = int(r.target)
        w = float(r.weight)
        assert adj[s, t] == pytest.approx(w)
        assert adj[t, s] == pytest.approx(w)

    # degrees array length and values
    assert hasattr(gd, "degrees_arr")
    assert gd.degrees_arr.shape[0] == N
    # neighbor map keys include each node
    assert isinstance(gd.neighbor_map, dict)
    for node in range(N):
        assert node in gd.neighbor_map

    # csr_by_type keys and shapes
    assert isinstance(gd.csr_by_type, dict)
    for ct, (indptr, indices, weights) in gd.csr_by_type.items():
        assert np.asarray(indptr).shape[0] == N + 1


def test_sample_from_master_graphdata_includes_household_and_samples():
    contacts, edges, N = make_toy_contacts_and_edges()
    minimal = build_minimal_graphdata_from_edge_list(edges, N=N)

    # Choose config with large mean_k so sampling selects all non-household peers (capped by deg)
    cfg = ModelConfig().copy_with({
        "population": {
            "wp_contacts": 100,
            "sch_contacts": 100,
            "cas_contacts": 100,
            "gq_contacts": 100
        },
        "epi": {"vax_uptake": 0.0, "vax_efficacy": 0.0}  # disable vax so no reductions
    })

    seed = 12345
    sampled = sample_from_master_graphdata(minimal, cfg, seed=seed)

    # schema ok
    assert set(sampled.columns) == {"source", "target", "weight", "contact_type"}
    # household pair (0,1) should be present
    hh_row = sampled[(sampled["source"] == 0) & (sampled["target"] == 1) & (sampled["contact_type"] == "hh")]
    assert not hh_row.empty

    # ensure undirected canonicalization (source <= target)
    assert (sampled["source"] <= sampled["target"]).all()

    # ensure uniqueness by (source,target)
    assert sampled.shape[0] == sampled[["source", "target"]].drop_duplicates().shape[0]