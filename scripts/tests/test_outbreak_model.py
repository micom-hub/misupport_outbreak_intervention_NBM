import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scripts.config import ModelConfig
from scripts.simulation.outbreak_model import NetworkModel
from scripts.graph.graph_utils import GraphData


@pytest.fixture
def simple_graph():
    """Create a 3-node linear graph: 0-1-2."""
    N = 3
    edge_list = pd.DataFrame(
        [(0, 1, 1.0, "hh"), (1, 2, 1.0, "hh")],
        columns=["source", "target", "weight", "contact_type"],
    )

    # Build required matrices for Numba determine_transmissions
    row = np.array([0, 1, 1, 2])
    col = np.array([1, 0, 2, 1])
    dat = np.ones(4, dtype=np.float32)
    adj = csr_matrix((dat, (row, col)), shape=(N, N))

    neighbor_map = {
        0: [(1, 1.0, "hh")],
        1: [(0, 1.0, "hh"), (2, 1.0, "hh")],
        2: [(1, 1.0, "hh")],
    }

    return GraphData(
        N=N,
        edge_list=edge_list,
        adj_matrix=adj,
        individual_lookup=pd.DataFrame({"age": [20, 20, 20], "sex": ["M", "F", "M"]}),
        ages=np.array([20, 20, 20], dtype=np.int32),
        sexes=np.array(["M", "F", "M"]),
        compliances=np.ones(N),
        neighbor_map=neighbor_map,
        fast_neighbor_map={},
        csr_by_type={"hh": (adj.indptr, adj.indices, adj.data)},
        contact_types=["hh"],
        ct_to_id={"hh": 0},
        id_to_ct={0: "hh"},
        full_node_list=[0, 1, 2],
        degrees_arr=np.array([1, 2, 1], dtype=np.int32),
    )


@pytest.fixture
def outbreak_config():
    return ModelConfig().copy_with(
        {
            "sim": {"n_replicates": 1, "simulation_duration": 10, "I0": [0]},
            "epi": {
                "base_transmission_prob": 1.0,
                "vax_uptake": 0.0,
            },  # Guarantee transmission
        }
    )


def test_model_initialization(outbreak_config, simple_graph):
    """Check if seeding I0 works correctly."""
    model = NetworkModel(config=outbreak_config, graphdata=simple_graph, run_dir=".")
    model._initialize_replicate(0)

    assert model.state[0] == 2  # Node 0 is Infectious
    assert model.state[1] == 0  # Node 1 is Susceptible
    assert 0 in model.states_over_time[0][2]  # Index 2 is the 'I' list


def test_transmission_logic(outbreak_config, simple_graph):
    """Verify that determine_transmissions spreads infection across edges."""
    model = NetworkModel(config=outbreak_config, graphdata=simple_graph, run_dir=".")
    model._initialize_replicate(0)

    # Node 0 is I, Node 1 is S. With p=1.0, 1 should be exposed.
    src, tgt, ct = model.determine_transmissions()

    assert 1 in tgt
    assert 0 in src


def test_state_transitions(outbreak_config, simple_graph):
    """Verify the progression S -> E -> I -> R."""
    # Force incubation and infectious periods to 1 day for rapid testing
    outbreak_config = outbreak_config.copy_with(
        {
            "epi": {
                "incubation_period": 1.0,
                "infectious_period": 1.0,
                "gamma_alpha": 100.0,
            }
        }
    )

    model = NetworkModel(config=outbreak_config, graphdata=simple_graph, run_dir=".")
    model._initialize_replicate(0)

    # Day 0: Node 0 is I.
    # Step to Day 1: 0 transmits to 1.
    model.step()
    assert model.state[1] == 1  # Node 1 is Exposed (E)

    # Step to Day 2: Node 1 moves E -> I, Node 0 moves I -> R
    model.step()
    assert model.state[1] == 2  # Node 1 is now Infectious (I)
    assert model.state[0] == 3  # Node 0 is now Recovered (R)


def test_stochastic_dieout(outbreak_config, simple_graph):
    """Check if simulation terminates correctly when no E/I cases remain."""
    # Set high recovery, low transmission to force dieout
    dieout_cfg = outbreak_config.copy_with(
        {"epi": {"base_transmission_prob": 0.0}, "sim": {"simulation_duration": 100}}
    )

    model = NetworkModel(config=dieout_cfg, graphdata=simple_graph, run_dir=".")
    model.simulate()

    # The simulation should end before Tmax
    assert model.all_end_days[0] < 100
    assert model.all_stochastic_dieout[0] == True


def test_results_aggregation(outbreak_config, simple_graph):
    """Verify the metrics in results_to_df."""
    model = NetworkModel(config=outbreak_config, graphdata=simple_graph, run_dir=".")
    model.simulate()
    df = model.results_to_df()

    assert "outbreakSize" in df.columns
    assert df["outbreakSize"].iloc[0] >= 1
    assert df["peakPrev"].iloc[0] <= 1.0
