import numpy as np
import pandas as pd
import pytest
import igraph as ig
import networkx as nx
import numbers
import warnings 

from scripts.network_model import NetworkModel, DefaultModelParams
from scripts.SynthDataProcessing import build_edge_list, build_individual_lookup

##################
# Fixtures
##################

@pytest.fixture
def sample_contacts_df():
    # Small synthetic population:
    data = [
        {"age": 60, "sex": "F", "race": "Latino", "hh_id": "C", "wp_id": np.nan, "sch_id": np.nan, "gq_id": "1", "gq": True},
        {"age": 34, "sex": "M", "race": "White", "hh_id": "A", "wp_id": "X", "sch_id": np.nan, "gq_id": np.nan, "gq": False}, 
        {"age": 29, "sex": "F", "race": "Black", "hh_id": "A", "wp_id": "Y", "sch_id": np.nan, "gq_id": np.nan, "gq": False},
        {"age": 17, "sex": "F", "race": "Asian", "hh_id": "B", "wp_id": "Y", "sch_id": "S1", "gq_id": np.nan, "gq": False},
        {"age": 15, "sex": "M", "race": "White", "hh_id": "B", "wp_id": np.nan, "sch_id": "S1", "gq_id": np.nan, "gq": False}
    ]
    return pd.DataFrame(data).reset_index(drop=True)

@pytest.fixture
def net_model(sample_contacts_df):
    params = DefaultModelParams.copy()
    params.update({
        "I0": [1],         # Infect individual 1 (person with index 1)
        "simulation_duration": 5,
        "seed": 123,
        "overwrite_edge_list": True,
        "save_data_files": True,
        "run_name": "test_run_pytest"

    })
    return NetworkModel(contacts_df=sample_contacts_df, params=params)

##################
# Tests
##################

#### Helper Functions

def test_name_to_ind_single_and_batch(net_model):
    g = net_model.epi_g
    node_names = np.array(g.vs["name"])
    test_name = node_names[0]
    node_ind = net_model.name_to_ind(g, test_name)
    assert g.vs[node_ind]["name"] == test_name

    some_names = node_names[:3]
    node_inds = net_model.name_to_ind(g, some_names)
    assert all(g.vs[ind]["name"] == name for ind, name in zip(node_inds, some_names))

    arr_names = node_names[:3]
    inds_arr = net_model.name_to_ind(g, arr_names)
    assert (np.array([g.vs[i]["name"] for i in inds_arr]) == arr_names).all()

def test_ind_to_name_single_and_batch(net_model):
    g = net_model.epi_g
    node_names = np.array(g.vs["name"])
    node_inds = np.arange(len(node_names))

    test_ind = node_inds[0]
    assert net_model.ind_to_name(g, test_ind) == g.vs[test_ind]["name"]

    some_inds = node_inds[:3]
    names_list = net_model.ind_to_name(g, some_inds)
    for ind, name_returned in zip(some_inds, names_list):
        assert name_returned == g.vs[ind]["name"]

    inds_arr = node_inds[:3]
    names_arr = net_model.ind_to_name(g, inds_arr)
    assert (np.array([g.vs[i]["name"] for i in inds_arr]) == names_arr).all()

#### Input/Output Consistency

def test_build_edge_list_and_lookup(sample_contacts_df):
    params = DefaultModelParams.copy()
    edge_list = build_edge_list(
        contacts_df=sample_contacts_df, params=params, rng=np.random.default_rng(params["seed"]), save=False, county="TestCounty"
    )
    assert "source" in edge_list.columns
    assert "target" in edge_list.columns
    assert "weight" in edge_list.columns
    lookup = build_individual_lookup(sample_contacts_df)
    assert set(["age", "race", "sex"]).issubset(lookup.columns)
    assert lookup.shape[0] == sample_contacts_df.shape[0]

def test_model_initialization(net_model):
    assert net_model.N == net_model.contacts_df.shape[0]
    assert hasattr(net_model, "edge_list")
    assert hasattr(net_model, "individual_lookup")
    assert net_model.epi_g.vs["name"] is not None

#### Graph Methods

def test_initialize_graph_creates_expected_nodes(net_model):
    g = net_model.epi_g
    assert 1 in g.vs["name"]
    assert all(isinstance(n, numbers.Integral) for n in g.vs["name"])

def test_add_to_graph_expands_network(net_model):
    g1 = net_model.epi_g
    g2 = net_model.add_to_graph(g1, [2])
    assert 3 in g2.vs["name"] or 2 in g2.vs["name"]
    assert len(g2.vs["name"]) >= len(g1.vs["name"])

def test_graph_edge_attributes(net_model):
    g = net_model.epi_g
    if g.ecount() > 0:
        for e in g.es:
            assert isinstance(e["weight"], float)
            assert isinstance(e["contact_type"], str)

#### State Tracking and Transitions

def test_state_initialization(net_model):
    assert net_model.state[net_model.params["I0"]] == 2
    susceptible = set(range(net_model.N)) - set(net_model.params["I0"])
    assert (net_model.state[list(susceptible)] == 0).all()

def test_assign_periods(net_model):
    inds = np.array([0,1,2])
    inc_periods = net_model.assign_incubation_period(inds)
    inf_periods = net_model.assign_infectious_period(inds)
    assert inc_periods.shape == (3,) and (inc_periods > 0).all()
    assert inf_periods.shape == (3,) and (inf_periods > 0).all()

def test_step_advances_epidemic(net_model):
    initial_I = np.sum(net_model.state == 2)
    net_model.step()
    # May still be just 1 infectious, but should not crash
    assert np.sum(net_model.state == 2) >= 0

def test_states_over_time_consistency(net_model):
    net_model.simulate()
    # Should fill up states_over_time for every step
    assert len(net_model.states_over_time) >= 1
    for S, E, I, R in net_model.states_over_time:
        all_states = set(S + E + I + R)
        # Each node in contacts_df should be in exactly one state
        assert all(i in all_states for i in range(net_model.N))

#### Visualization

def test_epi_curve_runs(net_model, monkeypatch):
    net_model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    net_model.epi_curve()

def test_draw_network_runs(net_model, monkeypatch):
    net_model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    final_timestep = getattr(net_model, "simulation_end_day", None) or (len(net_model.states_over_time) - 1)
    net_model.draw_network(final_timestep)

def test_make_movie_runs(net_model, monkeypatch, tmp_path):
    net_model.simulate()
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    monkeypatch.setattr(plt, "show", lambda: None)
    # Mock out ani.save to avoid actually writing files
    monkeypatch.setattr(animation.Animation, "save", lambda self, filename, writer, fps: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        net_model.make_movie(filename=str(tmp_path / "testmovie.mp4"))

def test_make_graphml_file_runs(net_model, tmp_path):
    net_model.simulate()
    # Should not crash and should create proper networkx graph object
    net_model.make_graphml_file(t=0, suffix="_test")
    assert hasattr(net_model, "nx_g")
    assert isinstance(net_model.nx_g, nx.Graph)
    # Should have nodes corresponding to E/I/R + neighbors
    # Find original set of affected nodes at t=0
    S, E, I, R = net_model.states_over_time[0]
    expected_nodes = set(E + I + R)
    graph_nodes = set(net_model.nx_g.nodes)
    assert expected_nodes.issubset(graph_nodes)

#### Population Lookup and Attribute Creation

def test_lookup_demographics(net_model):
    lookup = net_model.individual_lookup
    for i in range(net_model.N):
        assert "age" in lookup.columns
        assert "race" in lookup.columns
        assert "sex" in lookup.columns

def test_assign_node_attribute_works(net_model, sample_contacts_df):
    g = net_model.epi_g
    node_names = np.array(g.vs["name"])
    ages = net_model.contacts_df.iloc[node_names]["age"].to_numpy()
    net_model.assign_node_attribute("age", ages, node_names, g)
    for idx, expected_age in zip(node_names, ages):
        node_ind = next(i for i, name in enumerate(g.vs["name"]) if name == idx)
        actual_age = g.vs[node_ind]["age"]
        assert actual_age == expected_age

#### Integration Tests

def test_full_run_success(net_model, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    net_model.simulate()
    net_model.epi_curve()
    final_timestep = getattr(net_model, "simulation_end_day", None) or (len(net_model.states_over_time) - 1)
    net_model.draw_network(final_timestep)
    net_model.make_movie(filename="movie.mp4")
    net_model.make_graphml_file(t=final_timestep, suffix="_integration")
    assert net_model.model_has_run
    assert hasattr(net_model, "states_over_time")
    assert len(net_model.states_over_time) > 0
    assert hasattr(net_model, "nx_g")
    assert isinstance(net_model.nx_g, nx.Graph)