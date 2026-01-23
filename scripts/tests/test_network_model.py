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

def test_network_structures(net_model):
    model = net_model
    assert model.N == len(model.contacts_df)
    assert isinstance(model.edge_list, pd.DataFrame)
    assert model.adj_matrix.shape == (model.N, model.N)
    assert isinstance(model.individual_lookup, pd.DataFrame)
    assert model.individual_lookup.shape[0] == model.N
    assert isinstance(model.neighbor_map, dict)
    for i, v in model.neighbor_map.items():
        assert isinstance(v, list)

def test_state_initialization(net_model):
    net_model.initialize_states()
    model = net_model
    assert isinstance(model.state, np.ndarray)
    assert len(model.state) == model.N
    assert model.state[model.params["I0"]] == 2
    susceptible = set(range(model.N)) - set(model.params["I0"])
    assert (model.state[list(susceptible)] == 0).all()

def test_assign_periods(net_model):
    model = net_model
    model.initialize_states()
    inds = np.array([0,1,2])
    inc_periods = model.assign_incubation_period(inds)
    inf_periods = model.assign_infectious_period(inds)
    assert inc_periods.shape == (3,) and (inc_periods > 0).all()
    assert inf_periods.shape == (3,) and (inf_periods > 0).all()


def test_graph_attribute_assignment(net_model):
    model = net_model
    g = model.g_full
    names = np.array(g.vs["name"])
    ages = model.contacts_df.iloc[names]["age"].to_numpy()
    model.assign_node_attribute("age", ages, names, g)
    for idx, expected_age in zip(names, ages):
        ind = model.name_to_ind(g, idx)
        # igraph stores numeric attributes correctly
        assert g.vs[ind]["age"] == expected_age

def test_single_run_and_trajectories(net_model):
    model = net_model
    model.simulate()
    for run in range(model.n_runs):
        all_states = model.all_states_over_time[run]
        exposures = model.all_new_exposures[run]
        assert isinstance(all_states, list)
        assert isinstance(exposures, list)
        assert abs(len(exposures) == len(all_states)) 
        for timestep, (S,E,I,R) in enumerate(all_states):
            all_ids = set(S+E+I+R)
            assert all(i in all_ids for i in range(model.N))
            assert len(exposures) == len(all_states)
    # At least initial infection happens
    for run in range(model.n_runs):
        exposures = model.all_new_exposures[run]
        assert len(exposures) > 0
        assert len(exposures[0]) >= 0

def test_stochastic_dieout_flagged(net_model):
    model = net_model
    model.simulate()
    # At least one run will have dieout or not
    assert any(isinstance(flag, (np.bool_, bool)) for flag in model.all_stochastic_dieout)

def test_end_days_consistent(net_model):
    model = net_model
    model.simulate()
    # End day never exceeds Tmax
    for d in model.all_end_days:
        assert d <= model.Tmax

def test_epi_curve_runs(net_model, monkeypatch):
    model = net_model
    model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    for run in range(model.n_runs):
        model.epi_curve(run_number=run)

def test_cumulative_incidence_plot_runs(net_model, monkeypatch):
    model = net_model
    model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    for run in range(model.n_runs):
        model.cumulative_incidence_plot(run_number=run, strata=None)
        model.cumulative_incidence_plot(run_number=run, strata="age")
        model.cumulative_incidence_plot(run_number=run, strata="sex")

def test_cumulative_incidence_spaghetti_runs(net_model, monkeypatch):
    model = net_model
    model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    model.cumulative_incidence_spaghetti()

def test_draw_network_runs(net_model, monkeypatch):
    model = net_model
    model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    for run in range(model.n_runs):
        t = len(model.all_states_over_time[run]) // 2
        model.draw_network(t=t, run_number=run)

def test_make_movie_runs(net_model, monkeypatch, tmp_path):
    model = net_model
    model.simulate()
    import matplotlib.pyplot as plt, matplotlib.animation as animation
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(animation.Animation, "save", lambda self, filename, writer, fps: None)
    model.results_folder = tmp_path
    model.params["save_data_files"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.make_movie(run_number=0, filename=str(tmp_path / "testmovie.mp4"), fps=1)

def test_make_graphml_file_runs(net_model, tmp_path):
    model = net_model
    model.simulate()
    model.results_folder = tmp_path
    last_t = len(model.all_states_over_time[0]) - 1
    model.make_graphml_file(t=last_t, run_number=0)
    assert hasattr(model, "nx_g")
    assert isinstance(model.nx_g, nx.Graph)
    S, E, I, R = model.all_states_over_time[0][last_t]
    expected_nodes = set(E + I + R)
    graph_nodes = set(model.nx_g.nodes)
    # Test that all relevant nodes are included after conversion
    assert expected_nodes.issubset(graph_nodes)

def test_population_lookup(net_model):
    lookup = net_model.individual_lookup
    assert {"age","race","sex"}.issubset(set(lookup.columns))
    assert len(lookup)==net_model.N

def test_full_batch_run_monkeypatched_visualizations(net_model, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    model = net_model
    model.simulate()
    model.cumulative_incidence_spaghetti()
    for run in range(model.n_runs):
        model.epi_curve(run_number=run)
        model.cumulative_incidence_plot(run_number=run, strata=None)
        t = len(model.all_states_over_time[run]) // 2
        model.draw_network(t=t, run_number=run)