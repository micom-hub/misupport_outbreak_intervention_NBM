import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from scripts.config import ModelConfig
from scripts.lhd.lhd import LocalHealthDepartment
from scripts.lhd.surveillance import STAGE_INF, STAGE_PRE


# --- Mock NetworkModel for Policy Tests ---
class MockNetworkModel:
    """
    A mock NetworkModel that provides necessary attributes and allows
    LHD to modify multipliers.
    """

    def __init__(self, N, config, neighbor_map, ct_to_id):
        self.N = N
        self.config = config
        self.neighbor_map = neighbor_map
        self.ct_to_id = ct_to_id
        self.id_to_ct = {v: k for k, v in ct_to_id.items()}
        self.ages = np.random.randint(0, 80, N)
        self.is_vaccinated = np.zeros(N, dtype=bool)
        # These are modified by LHD actions
        self.in_multiplier = {
            ct: np.ones(N, dtype=np.float32) for ct in ct_to_id.keys()
        }
        self.out_multiplier = {
            ct: np.ones(N, dtype=np.float32) for ct in ct_to_id.keys()
        }
        self.replicate_ind = 0  # For _log_day
        self.state = np.zeros(N, dtype=np.int8)
        # Mock methods that LHD might call (e.g., for results)
        self.results_to_df = MagicMock(return_value=pd.DataFrame())
        self.all_states_over_time = {0: []}  # For surveillance to look up truth


@pytest.fixture
def base_policy_config():
    """A base ModelConfig for LHD policies."""
    return ModelConfig().copy_with(
        {
            "sim": {"n_replicates": 1, "simulation_duration": 10, "I0": [0]},
            "lhd": {
                "lhd_daily_capacity": 100,
                "p_detect_inf": 1.0,  # Ensure detection
                "report_delay_days": 0,  # No delay for simplicity
                "lhd_default_int_reduction": 0.5,
                "lhd_default_int_duration": 5,
                "trace_recall_prob": 1.0,  # Make knowledge gain deterministic for tests
            },
        }
    )


@pytest.fixture
def simple_network_data():
    """A simple network for testing tracing and isolation."""
    N = 10
    # Node 0 is connected to 1 (hh), 2 (wp)
    # Node 1 is connected to 0 (hh), 3 (sch)
    # Node 2 is connected to 0 (wp)
    # Node 3 is connected to 1 (sch)
    neighbor_map = {
        0: [(1, 1.0, "hh"), (2, 1.0, "wp")],
        1: [(0, 1.0, "hh"), (3, 1.0, "sch")],
        2: [(0, 1.0, "wp")],
        3: [(1, 1.0, "sch")],
        4: [],  # Isolated node
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
    }
    ct_to_id = {"hh": 0, "wp": 1, "sch": 2, "cas": 3}
    return N, neighbor_map, ct_to_id


@pytest.fixture
def mock_lhd(base_policy_config, simple_network_data):
    N, neighbor_map, ct_to_id = simple_network_data
    mock_model = MockNetworkModel(N, base_policy_config, neighbor_map, ct_to_id)
    lhd = LocalHealthDepartment(seed=1, surv_seed=2, model=mock_model)
    # Attach LHD to mock_model for _log_day to work
    mock_model.lhd = lhd
    return lhd


# --- Helper to simulate LHD steps ---
def simulate_lhd_days(lhd: LocalHealthDepartment, epi_states_per_day: list):
    """
    Simulates LHD steps over multiple days.
    epi_states_per_day: List of dicts, each dict is the epi_state for that day.
    """
    for t, epi_state in enumerate(epi_states_per_day):
        # Update mock_model's all_states_over_time for surveillance truth lookup
        # This is a simplification; in real model, this is updated by NetworkModel.step
        S = np.where(lhd.model.state == 0)[0]
        E = np.where(lhd.model.state == 1)[0]
        I = np.where(lhd.model.state == 2)[0]
        R = np.where(lhd.model.state == 3)[0]
        lhd.model.all_states_over_time[0].append([S, E, I, R])

        lhd.step(t=t, epi_state=epi_state)

        # Update mock_model's state based on epi_state for next day's truth lookup
        # This is a very simplified update, assuming new_inf_ids become I, and pre_ids become E
        for nid in epi_state.get("new_inf_ids", []):
            lhd.model.state[nid] = STAGE_INF
        for nid in epi_state.get("new_pre_ids", []):
            lhd.model.state[nid] = STAGE_PRE

    return lhd


# --- Policy Test Cases ---


def test_trace_and_test_policy(base_policy_config, simple_network_data):
    """
    Test 'trace_and_test' policy:
    1. Node 0 reported (Day 0).
    2. Node 0 isolated, Node 0 traced.
    3. Tracing discovers Node 1 (hh contact). Node 1 is truly infectious.
    4. Node 1 is tested (due to TestContactsOfKnownCases).
    5. Node 1 reported (Day 1).
    6. Node 1 isolated, Node 1 traced.
    """
    N, neighbor_map, ct_to_id = simple_network_data
    config = base_policy_config.copy_with({"lhd": {"policy_name": "trace_and_test"}})
    mock_model = MockNetworkModel(N, config, neighbor_map, ct_to_id)
    lhd = LocalHealthDepartment(seed=1, surv_seed=2, model=mock_model)
    mock_model.lhd = lhd

    # Day 0: Node 0 reported as infectious
    epi_state_day0 = {
        "new_inf_ids": np.array([0], dtype=np.int32),
        "inf_ids": np.array([0], dtype=np.int32),
    }
    # Day 1: Node 1 is truly infectious (discovered via tracing from Day 0)
    epi_state_day1 = {
        "new_inf_ids": np.array([1], dtype=np.int32),
        "inf_ids": np.array([0, 1], dtype=np.int32),
    }
    # Day 2: No new infections, but Node 1 is now known.
    epi_state_day2 = {
        "new_inf_ids": np.empty(0, dtype=np.int32),
        "inf_ids": np.array([0, 1], dtype=np.int32),
    }

    simulate_lhd_days(lhd, [epi_state_day0, epi_state_day1, epi_state_day2])

    log_df = lhd.action_log_to_df()

    # Day 0 actions: Node 0 isolated, Node 0 traced
    day0_actions = log_df[log_df["t"] == 0]
    assert len(day0_actions) == 2
    assert {"isolate", "trace"} == set(day0_actions["action"])
    assert all(a.node == 0 for a in day0_actions.itertuples())

    # Day 1 actions: Node 1 tested (from TestContactsOfKnownCases), Node 1 isolated, Node 1 traced
    day1_actions = log_df[log_df["t"] == 1]
    assert len(day1_actions) == 3  # Isolate 1, Trace 1, Test 1
    assert {"isolate", "trace", "test"} == set(day1_actions["action"])

    # Check specific targets for day 1
    isolated_day1 = day1_actions[day1_actions["action"] == "isolate"]["node"].iloc[0]
    traced_day1 = day1_actions[day1_actions["action"] == "trace"]["node"].iloc[0]
    tested_day1 = day1_actions[day1_actions["action"] == "test"]["node"].iloc[0]

    assert isolated_day1 == 1
    assert traced_day1 == 1
    assert tested_day1 == 2  # Node 1 already known, so Node 2 is tested


def test_network_crawl_policy(base_policy_config, simple_network_data):
    """
    Test 'network_crawl' policy:
    1. Node 0 reported (Day 0).
    2. Node 0 traced. Tracing discovers Node 1 (hh contact).
    3. Day 1: TraceEdgeEndpoints proposes tracing Node 1.
    """
    N, neighbor_map, ct_to_id = simple_network_data
    config = base_policy_config.copy_with({"lhd": {"policy_name": "network_crawl"}})
    mock_model = MockNetworkModel(N, config, neighbor_map, ct_to_id)
    lhd = LocalHealthDepartment(seed=1, surv_seed=2, model=mock_model)
    mock_model.lhd = lhd

    # Day 0: Node 0 reported as infectious
    epi_state_day0 = {
        "new_inf_ids": np.array([0], dtype=np.int32),
        "inf_ids": np.array([0], dtype=np.int32),
    }
    # Day 1: No new infections, but LHD should crawl
    epi_state_day1 = {
        "new_inf_ids": np.empty(0, dtype=np.int32),
        "inf_ids": np.array([0], dtype=np.int32),
    }

    simulate_lhd_days(lhd, [epi_state_day0, epi_state_day1])

    log_df = lhd.action_log_to_df()

    # Day 0 actions: Node 0 traced (from TraceNewCases)
    day0_actions = log_df[log_df["t"] == 0]
    assert len(day0_actions) == 1
    assert day0_actions["action"].iloc[0] == "trace"
    assert day0_actions["node"].iloc[0] == 0

    # Day 1 actions: Node 1 traced (from TraceEdgeEndpoints, as it was discovered from Node 0's trace on Day 0)
    day1_actions = log_df[log_df["t"] == 1]
    assert len(day1_actions) == 2  # Traces node 1 and 2 (contacts of node 0)
    assert day1_actions["action"].iloc[0] == "trace"
    assert set(day1_actions["node"]) == {1, 2}


def test_network_crawl_isolate_policy(base_policy_config, simple_network_data):
    """
    Test 'network_crawl_isolate' policy:
    1. Node 0 reported (Day 0).
    2. Node 0's neighbors (Node 1, Node 2) are isolated (due to IsolateNeighborsOfHighDegreeCases).
    3. Node 0 is also isolated (due to IsolateNewCases, which is higher priority).
    """
    N, neighbor_map, ct_to_id = simple_network_data
    config = base_policy_config.copy_with(
        {"lhd": {"policy_name": "network_crawl_isolate"}}
    )
    mock_model = MockNetworkModel(N, config, neighbor_map, ct_to_id)
    lhd = LocalHealthDepartment(seed=1, surv_seed=2, model=mock_model)
    mock_model.lhd = lhd

    # Pre-populate known adjacency so LHD knows about neighbors 1 and 2 on Day 0
    lhd.state.known_edges.append((0, 1, 0, -1, "prior"))
    lhd.state.known_edges.append((0, 2, 1, -1, "prior"))
    lhd.state.known_adj[0].extend([(1, 0, -1, "prior"), (2, 1, -1, "prior")])
    lhd.state.known_adj[1].append((0, 0, -1, "prior"))
    lhd.state.known_adj[2].append((0, 1, -1, "prior"))

    # Day 0: Node 0 reported as infectious. Node 0 has 2 neighbors (1, 2) in the known_adj.
    epi_state_day0 = {
        "new_inf_ids": np.array([0], dtype=np.int32),
        "inf_ids": np.array([0], dtype=np.int32),
    }
    # Day 1: No new infections, but LHD should crawl and isolate
    epi_state_day1 = {
        "new_inf_ids": np.empty(0, dtype=np.int32),
        "inf_ids": np.array([0], dtype=np.int32),
    }

    simulate_lhd_days(lhd, [epi_state_day0, epi_state_day1])

    log_df = lhd.action_log_to_df()

    # Day 0 actions: Node 0 isolated (from IsolateNewCases), Node 1 isolated, Node 2 isolated (from IsolateNeighborsOfHighDegreeCases)
    day0_actions = log_df[log_df["t"] == 0]

    # Expected actions: isolate 0, isolate 1, isolate 2
    assert len(day0_actions) == 3
    assert all(a.action == "isolate" for a in day0_actions.itertuples())
    assert {p.node for p in day0_actions.itertuples()} == {0, 1, 2}

    # Verify isolation multipliers were applied
    # Node 0, 1, 2 should have their out_multiplier reduced for 'wp' (default iso contact types)
    # Initial value is 1.0, reduction is 0.5, so factor is 0.5
    expected_factor = 0.5
    # 'hh' is excluded from isolation by default, so check 'wp'
    assert mock_model.out_multiplier["wp"][0] == expected_factor
    assert mock_model.out_multiplier["wp"][1] == expected_factor
    assert mock_model.out_multiplier["wp"][2] == expected_factor

    # Other nodes should remain 1.0
    assert mock_model.out_multiplier["hh"][3] == 1.0


def test_lhd_capacity_limits_actions(base_policy_config, simple_network_data):
    """
    Verify that LHD daily capacity limits the number of actions taken.
    Using 'trace_and_test' policy with very low capacity.
    """
    N, neighbor_map, ct_to_id = simple_network_data
    config = base_policy_config.copy_with(
        {
            "lhd": {
                "policy_name": "trace_and_test",
                "lhd_daily_capacity": 1,  # Very low capacity
            }
        }
    )
    mock_model = MockNetworkModel(N, config, neighbor_map, ct_to_id)
    lhd = LocalHealthDepartment(seed=1, surv_seed=2, model=mock_model)
    mock_model.lhd = lhd

    # Day 0: Node 0 reported as infectious
    epi_state_day0 = {
        "new_inf_ids": np.array([0], dtype=np.int32),
        "inf_ids": np.array([0], dtype=np.int32),
    }

    simulate_lhd_days(lhd, [epi_state_day0])

    log_df = lhd.action_log_to_df()

    # Check daily log for selected actions
    daily_log_df = lhd.lhd_daily_log_to_df()
    assert (
        daily_log_df["selected_total"].iloc[0] == 1
    )  # Only 1 action should be selected

    # Check action log for what was actually applied
    day0_actions = log_df[log_df["t"] == 0]
    assert len(day0_actions) == 1  # Only one action should be logged
    # The 'isolate_new_cases' algorithm has priority 2.0, 'trace_new_cases' has 1.5.
    # So isolation should be prioritized.
    assert day0_actions["action"].iloc[0] == "isolate"
    assert day0_actions["node"].iloc[0] == 0
