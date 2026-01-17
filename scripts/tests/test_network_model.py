import numpy as np
import pandas as pd
import pytest
import igraph as ig
from scripts.network_model import NetworkModel, DefaultModelParams
from scripts.SynthDataProcessing import build_edge_list, build_individual_lookup

##################
# Fixtures
##################

@pytest.fixture
def sample_contacts_df():
    # Small synthetic population:
    data = [
        {"age": 34, "sex": "M", "race": "White", "hh_id": "A", "wp_id": "X", "sch_id": np.nan, "gq_id": np.nan},
        {"age": 29, "sex": "F", "race": "Black", "hh_id": "A", "wp_id": "Y", "sch_id": np.nan, "gq_id": np.nan},
        {"age": 17, "sex": "F", "race": "Asian", "hh_id": "B", "wp_id": "Y", "sch_id": "S1", "gq_id": np.nan},
        {"age": 15, "sex": "M", "race": "White", "hh_id": "B", "wp_id": np.nan, "sch_id": "S1", "gq_id": np.nan},
        {"age": 60, "sex": "F", "race": "Latino", "hh_id": "C", "wp_id": np.nan, "sch_id": np.nan, "gq_id": "G1"},
    ]
    return pd.DataFrame(data).reset_index(drop=True)

@pytest.fixture
def net_model(sample_contacts_df):
    params = DefaultModelParams.copy()
    params.update({
        "I0": [1],         # Infect individual 1 (person with index 1)
        "simulation_duration": 5,
        "seed": 123
    })
    return NetworkModel(contacts_df=sample_contacts_df, params=params)

##################
# Tests
##################

#### Input/Output Consistency

def test_build_edge_list_and_lookup(sample_contacts_df):
    params = DefaultModelParams.copy()
    edge_list = build_edge_list(
        contacts_df=sample_contacts_df, params=params, rng=np.random.default_rng(params["seed"]), save=False, county="TestCounty"
    )
    assert "source" in edge_list.columns
    assert "target" in edge_list.columns
    assert "weight" in edge_list.columns
    assert "contact_type" in edge_list.columns
    lookup = build_individual_lookup(sample_contacts_df)
    assert set(["age", "race", "sex"]).issubset(lookup.columns)
    assert lookup.shape[0] == sample_contacts_df.shape[0]

def test_model_initialization(net_model):
    # Model should contain the number of individuals matching df shape
    assert net_model.N == net_model.contacts_df.shape[0]
    assert hasattr(net_model, "edge_list")
    assert hasattr(net_model, "individual_lookup")
    assert net_model.epi_g.vs["name"] is not None

#### Graph Methods

def test_initialize_graph_creates_expected_nodes(net_model):
    # Infectious at init: [1]
    g = net_model.epi_g
    # Graph must contain index 1 and neighbors (likely 0 and/or 2, depending on edge_list logic)
    assert 1 in g.vs["name"]
    # Check that node names are all integers
    assert all(isinstance(n, int) for n in g.vs["name"])

def test_add_to_graph_expands_network(net_model):
    # Try adding node 3 as "infectious"
    g1 = net_model.epi_g
    g2 = net_model.add_to_graph(g1, [2])
    # Node 3 should be present, as well as its neighbors
    assert 3 in g2.vs["name"]
    # Check if any new nodes were added
    assert len(g2.vs["name"]) >= len(g1.vs["name"])

def test_graph_edge_attributes(net_model):
    # The edges should have weights and contact types as attributes
    g = net_model.epi_g
    if g.ecount() > 0:
        for e in g.es:
            assert isinstance(e["weight"], float)
            assert e["contact_type"] in ("hh", "wp", "sch", "gq")

#### State Tracking and Transitions

def test_state_initialization(net_model):
    # Only I0 should be infectious, all others susceptible
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
    # After step, there should be a valid state transition (may still be just 1 infectious if not enough contacts in dummy data)
    assert np.sum(net_model.state == 2) >= initial_I

def test_states_over_time_consistency(net_model):
    net_model.simulate()
    # Should fill up states_over_time for every step
    assert len(net_model.states_over_time) >= 1
    for S, E, I, R in net_model.states_over_time:
        all_states = S + E + I + R
        # Every node in the graph should be represented in some state list at each time
        present_names = set([int(v["name"]) for v in net_model.epi_g.vs])
        assert set(present_names).issubset(all_states)

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
    final_timestep = getattr(net_model, "simulation_end_day", None) or (len(net_model.epi_graphs)-1)
    net_model.draw_network(final_timestep)

#### Population Lookup

def test_lookup_demographics(net_model):
    lookup = net_model.individual_lookup
    # Demographics are accessible per individual index
    for i in range(net_model.N):
        assert "age" in lookup.columns
        assert "race" in lookup.columns
        assert "sex" in lookup.columns

#### Integration/Sanity

def test_full_run_success(net_model, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    net_model.simulate()
    net_model.epi_curve()
    net_model.draw_network(getattr(net_model, "simulation_end_day", None) or (len(net_model.epi_graphs)-1))
    # Check that simulation ran and epidemiological states updated
    assert net_model.model_has_run
    assert hasattr(net_model, "states_over_time")
    assert len(net_model.states_over_time) > 0