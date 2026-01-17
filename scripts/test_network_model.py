import numpy as np
import igraph as ig
import scipy.sparse as sp
import pytest

from scripts.syntheticAM import df_to_adjacency
from scripts.network_model import NetworkModel, DefaultModelParams

@pytest.fixture
def simple_adj_csr():
    # Create a 5-node undirected adjacency matrix, with weights, in CSR format
    adj = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0.5, 0, 0],
        [0, 0.5, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ])
    return sp.csr_matrix(adj)

@pytest.fixture
def net_model(simple_adj_csr):
    params = DefaultModelParams.copy()
    params["I0"] = [1]   # Start with one infection, node 1
    params["simulation_duration"] = 10
    params["seed"] = 1234
    return NetworkModel(adj_csr=simple_adj_csr, params=params)

###NETWORK CONSTRUCTION TESTS
def test_initialize_graph_nodes_and_names(net_model):
    g = net_model.initialize_graph([1])
    assert set(g.vs['name']) == set([0, 1, 2])
    assert all(isinstance(name, int) for name in g.vs['name'])

def test_initialize_graph_edges_and_weights(net_model):
    g = net_model.initialize_graph([1])
    edges = set((g.vs[e.source]['name'], g.vs[e.target]['name']) for e in g.es)
    assert (1, 0) in edges or (0, 1) in edges
    assert (1, 2) in edges or (2, 1) in edges
    expected_weights = []
    for tgt in net_model.adj_csr[1].indices:
        expected_weights.append(net_model.adj_csr[1, tgt])
    actual_weights = [e['weight'] for e in g.es]
    for w in expected_weights:
        assert w in actual_weights

def test_initialize_graph_multiple_indices(net_model):
    g = net_model.initialize_graph([1, 2])
    assert set(g.vs['name']) == set([0, 1, 2, 3])

def test_add_to_graph_adds_new_nodes(net_model):
    g = net_model.initialize_graph([1])
    g2 = net_model.add_to_graph(g, [2])
    assert set(g2.vs['name']) == set([0, 1, 2, 3])

def test_add_to_graph_no_duplicate_nodes(net_model):
    g = net_model.initialize_graph([1])
    g2 = net_model.add_to_graph(g, [1])
    assert len(g2.vs) == len(set(g2.vs['name']))

def test_add_to_graph_correct_edge_weights(net_model):
    g = net_model.initialize_graph([1])
    g2 = net_model.add_to_graph(g, [2])
    idx_2 = [v.index for v in g2.vs if v['name'] == 2][0]
    idx_3 = [v.index for v in g2.vs if v['name'] == 3][0]
    found = False
    for e in g2.es:
        if (e.source == idx_2 and e.target == idx_3) or (e.source == idx_3 and e.target == idx_2):
            assert e['weight'] == float(net_model.adj_csr[2, 3])
            found = True
    assert found, "Edge between 2 and 3 not found"

def test_add_to_graph_duplicates_if_expected(net_model):
    g = net_model.initialize_graph([1])
    g2 = net_model.add_to_graph(g, [2])
    num_edges_before = g2.ecount()
    g3 = net_model.add_to_graph(g2, [2])
    num_edges_after = g3.ecount()
    assert num_edges_after >= num_edges_before

def test_vertex_names_consistent_after_repeated_add(net_model):
    g = net_model.initialize_graph([1])
    g2 = net_model.add_to_graph(g, [2])
    g3 = net_model.add_to_graph(g2, [3])
    assert len(set(g3.vs['name'])) == len(g3.vs['name'])
    for n in g3.vs['name']:
        assert isinstance(n, (int, np.integer))
        assert 0 <= n < net_model.adj_csr.shape[0]

### MODEL FLOW TESTS

def test_assign_incubation_period(net_model):
    inds = np.array([0,1,2])
    inc_period = net_model.assign_incubation_period(inds)
    assert inc_period.shape == (3,)
    assert (inc_period > 0).all()

def test_assign_infectious_period(net_model):
    inds = [0, 1, 2]
    inf_period = net_model.assign_infectious_period(inds)
    assert inf_period.shape == (3,)
    assert (inf_period > 0).all()

def test_initial_state_assignment(net_model):
    state = net_model.state
    assert state[1] == 2, "Node 1 should be infectious"
    assert (state[[0,2,3,4]] == 0).all(), "Other nodes should be susceptible"

def test_simulation_runs_and_stops(net_model):
    net_model.simulate()
    assert net_model.model_has_run
    final_states = net_model.state
    valid_states = set([0, 1, 2, 3])
    assert set(final_states).issubset(valid_states)


### ANALYSIS METHOD TESTS

def test_epi_curve_plot_runs(net_model, monkeypatch):
    net_model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    net_model.epi_curve()

def test_draw_network_runs(net_model, monkeypatch):
    net_model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    t = net_model.simulation_end_day or (len(net_model.epi_graphs) - 1)
    net_model.draw_network(t)



###########################
# Integration (Sanity) Test
###########################
def test_full_keweenaw_run(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    Keweenaw_adj = df_to_adjacency(county = "Keweenaw", saveFile = False, sparse = True)
    testModel = NetworkModel(adj_csr = Keweenaw_adj)
    testModel.simulate()
    testModel.epi_curve()
    testModel.draw_network(testModel.simulation_end_day)