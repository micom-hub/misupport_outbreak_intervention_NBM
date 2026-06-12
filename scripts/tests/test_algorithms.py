import pytest
import numpy as np
from unittest.mock import MagicMock

from scripts.lhd.state import LHDState
from scripts.lhd.surveillance import STAGE_INF, STAGE_PRE
from scripts.lhd.algorithms_state import (
    IsolateNewCases,
    TraceNewCases,
    TestContactsOfKnownCases,
    TraceEdgeEndpoints,
    IsolateNeighborsOfHighDegreeCases,
)
from scripts.lhd.response_types import ActionProposal


@pytest.fixture
def mock_lhd_state():
    """A clean LHDState for each test."""
    N = 20
    state = LHDState(N=N)
    # Mock a basic neighbor map for LHDState.neighbors to work
    state.known_adj = {
        0: [(1, 0, 0, "trace"), (2, 0, 0, "trace")],  # Node 0 connected to 1, 2
        1: [(0, 0, 0, "trace"), (3, 0, 0, "trace")],  # Node 1 connected to 0, 3
        2: [(0, 0, 0, "trace")],
        3: [(1, 0, 0, "trace")],
    }
    return state


def test_isolate_new_cases_proposals(mock_lhd_state):
    """Verify IsolateNewCases proposes isolation for new cases."""
    mock_lhd_state.new_cases_today = np.array([5, 6], dtype=np.int32)
    algo = IsolateNewCases(priority=1.0, cost_per_case=1)
    proposals = algo.propose(mock_lhd_state)

    assert len(proposals) == 2
    assert all(p.action == "isolate" for p in proposals)
    assert {p.target for p in proposals} == {5, 6}
    assert all(p.priority == 1.0 for p in proposals)
    assert all(p.reason == "newly reported case" for p in proposals)


def test_trace_new_cases_proposals(mock_lhd_state):
    """Verify TraceNewCases proposes tracing for new cases."""
    mock_lhd_state.new_cases_today = np.array([7], dtype=np.int32)
    algo = TraceNewCases(priority=0.8, cost_per_case=2)
    proposals = algo.propose(mock_lhd_state)

    assert len(proposals) == 1
    assert proposals[0].action == "trace"
    assert proposals[0].target == 7
    assert proposals[0].priority == 0.8
    assert proposals[0].cost_units == 2
    assert proposals[0].reason == "trace newly reported case"


def test_test_contacts_of_known_cases_proposals(mock_lhd_state):
    """Verify TestContactsOfKnownCases proposes testing for neighbors of known cases."""
    # Node 0 is a known case, connected to 1 and 2
    mock_lhd_state.known_case[0] = True
    mock_lhd_state.known_case_list.append(0)
    mock_lhd_state.known_adj[0] = [(1, 0, 0, "trace"), (2, 0, 0, "trace")]
    mock_lhd_state.known_adj[1] = [(0, 0, 0, "trace")]
    mock_lhd_state.known_adj[2] = [(0, 0, 0, "trace")]

    # Node 1 is also a known case, connected to 0 and 3
    mock_lhd_state.known_case[1] = True
    mock_lhd_state.known_case_list.append(1)
    mock_lhd_state.known_adj[1].append((3, 0, 0, "trace"))
    mock_lhd_state.known_adj[3] = [(1, 0, 0, "trace")]

    algo = TestContactsOfKnownCases(priority=0.5)
    proposals = algo.propose(mock_lhd_state)

    # Expected: test 2 (contact of 0), test 3 (contact of 1)
    # Node 0 and 1 are already known cases, so no test proposals for them.
    assert len(proposals) == 2
    assert all(p.action == "test" for p in proposals)
    assert {p.target for p in proposals} == {2, 3}
    assert all(p.priority == 0.5 for p in proposals)

    # Test that already pending tests are not proposed again
    mock_lhd_state.pending_tests[2] = [MagicMock()]  # Node 2 is already pending test
    proposals_after_pending = algo.propose(mock_lhd_state)
    assert len(proposals_after_pending) == 1
    assert proposals_after_pending[0].target == 3


def test_trace_edge_endpoints_proposals(mock_lhd_state):
    """Verify TraceEdgeEndpoints proposes tracing for endpoints of newly discovered edges."""
    mock_lhd_state.last_t = 5
    # Edge (10, 11) discovered at t=5
    mock_lhd_state.known_edges.append((10, 11, 0, 5, "trace"))
    # Edge (11, 12) discovered at t=4 (not new today)
    mock_lhd_state.known_edges.append((11, 12, 0, 4, "trace"))

    algo = TraceEdgeEndpoints(priority=0.7)
    proposals = algo.propose(mock_lhd_state)

    assert len(proposals) == 2
    assert all(p.action == "trace" for p in proposals)
    assert {p.target for p in proposals} == {10, 11}
    assert all(p.priority == 0.7 for p in proposals)
    assert all(p.reason == "crawl discovered edge endpoint" for p in proposals)


def test_isolate_neighbors_of_high_degree_cases_proposals(mock_lhd_state):
    """Verify IsolateNeighborsOfHighDegreeCases proposes isolation for neighbors of high-degree new cases."""
    mock_lhd_state.new_cases_today = np.array([0, 1], dtype=np.int32)

    # Node 0 has 2 known neighbors (1, 2)
    mock_lhd_state.known_adj[0] = [(1, 0, 0, "trace"), (2, 0, 0, "trace")]
    mock_lhd_state.known_adj[1] = [(0, 0, 0, "trace"), (3, 0, 0, "trace")]
    mock_lhd_state.known_adj[2] = [(0, 0, 0, "trace")]
    mock_lhd_state.known_adj[3] = [(1, 0, 0, "trace")]

    # Node 1 has 2 known neighbors (0, 3)
    # Both 0 and 1 have degree 2 in the known graph.
    # Let's make node 0 higher degree by adding more neighbors
    mock_lhd_state.known_adj[0].extend([(4, 0, 0, "trace"), (5, 0, 0, "trace")])
    mock_lhd_state.known_adj[4] = [(0, 0, 0, "trace")]
    mock_lhd_state.known_adj[5] = [(0, 0, 0, "trace")]

    algo = IsolateNeighborsOfHighDegreeCases(priority=1.5)
    proposals = algo.propose(mock_lhd_state)

    # Node 0 (degree 4) is higher priority than Node 1 (degree 2)
    # Neighbors of 0: {1, 2, 4, 5}
    # Neighbors of 1: {0, 3}
    # Total expected proposals: 4 (from 0) + 2 (from 1) = 6
    assert len(proposals) == 6
    assert all(p.action == "isolate" for p in proposals)

    # Check targets: should be neighbors of 0 and 1
    expected_targets = {1, 2, 4, 5, 0, 3}
    assert {p.target for p in proposals} == expected_targets

    # Check priority: proposals from higher degree case should come first
    # The algorithm sorts cases by degree, then proposes for their neighbors.
    # So, proposals for neighbors of node 0 should appear before neighbors of node 1.
    # The actual order of proposals for neighbors of the *same* case is not guaranteed
    # by the current implementation, but the grouping by source case is.

    # Let's check that the reasons reflect the source case and degree
    node0_proposals = [p for p in proposals if "case 0" in p.reason]
    node1_proposals = [p for p in proposals if "case 1" in p.reason]

    assert len(node0_proposals) == 4
    assert len(node1_proposals) == 2

    # Verify sorting by degree (higher degree case's neighbors come first)
    # This is implicitly tested by the order of proposals if the list is built sequentially
    # from the sorted `case_degrees`.
    # The current implementation adds all neighbors of the first case, then all neighbors of the second.
    # So, the first 4 proposals should be for neighbors of node 0, and the next 2 for neighbors of node 1.

    # To be more robust, we can check the targets of the first few proposals.
    # The exact order of targets within a case's neighbors is not defined, so we check sets.
    first_proposals_targets = {p.target for p in proposals[:4]}
    assert first_proposals_targets == {1, 2, 4, 5}  # Neighbors of node 0

    remaining_proposals_targets = {p.target for p in proposals[4:]}
    assert remaining_proposals_targets == {0, 3}  # Neighbors of node 1


def test_no_proposals_when_no_new_cases_or_edges(mock_lhd_state):
    """Verify algorithms propose nothing when their triggers are not met."""
    mock_lhd_state.new_cases_today = np.empty(0, dtype=np.int32)
    mock_lhd_state.known_edges = []
    mock_lhd_state.known_case_list = []
    mock_lhd_state.last_t = 0

    assert not IsolateNewCases().propose(mock_lhd_state)
    assert not TraceNewCases().propose(mock_lhd_state)
    assert not TestContactsOfKnownCases().propose(mock_lhd_state)
    assert not TraceEdgeEndpoints().propose(mock_lhd_state)
    assert not IsolateNeighborsOfHighDegreeCases().propose(mock_lhd_state)
