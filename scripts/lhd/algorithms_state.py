from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

from scripts.lhd.response_types import ActionProposal

#
class AlgorithmBase:
    name: str = "base_state"
    def propose(self, state) -> List[ActionProposal]:
        return []


class IsolateNewCases(AlgorithmBase):
    name = "isolate_new_cases"

    def __init__(self, *, cost_per_case: int = 1, priority: float = 1.0, params: Optional[Dict[str, Any]] = None):
        self.cost_per_case = int(cost_per_case)
        self.priority = float(priority)
        self.params = params or {}

    def propose(self, state) -> List[ActionProposal]:
        nodes = np.asarray(getattr(state, "new_cases_today", np.empty(0, np.int32)), dtype=np.int32)
        if nodes.size == 0:
            return []
        return [
            ActionProposal(
                action="isolate",
                target_kind="node",
                target=int(n),
                priority=self.priority,
                cost_units=max(self.cost_per_case, 1),
                params=self.params,
                source_algo=self.name,
                reason="newly reported case",
            )
            for n in nodes
        ]

class TraceNewCases(AlgorithmBase):
    name = "trace_new_cases"

    def __init__(self, *, cost_per_case: int = 1, priority: float = 1.0, params: Optional[Dict[str, Any]] = None):
        self.cost_per_case = int(cost_per_case)
        self.priority = float(priority)
        self.params = params or {}

    def propose(self, state) -> List[ActionProposal]:
        nodes = np.asarray(getattr(state, "new_cases_today", np.empty(0, np.int32)), dtype=np.int32)
        if nodes.size == 0:
            return []
        return [
            ActionProposal(
                action="trace",
                target_kind="node",
                target=int(n),
                priority=self.priority,
                cost_units=max(self.cost_per_case, 1),
                params=self.params,
                source_algo=self.name,
                reason="trace newly reported case",
            )
            for n in nodes
        ]


class TestContactsOfKnownCases(AlgorithmBase):
    """
    Look at known cases and propose testing for their neighbors in the known network
    who are not yet known to be infected.
    """

    # pytest wants to look at this lol
    __test__ = False
    name = "test_contacts_of_known_cases"

    def __init__(
        self,
        *,
        cost_per_node: int = 1,
        priority: float = 0.5,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.cost_per_node = int(cost_per_node)
        self.priority = float(priority)
        self.params = params or {}

    def propose(self, state) -> List[ActionProposal]:
        proposals = []
        # We look at all known cases to find their contacts
        for case_id in state.known_case_list:
            for nbr_info in state.neighbors(case_id):
                nbr_id = nbr_info[0]
                # Only test if they aren't already a known case or currently being tested
                if not state.known_case[nbr_id] and nbr_id not in state.pending_tests:
                    proposals.append(
                        ActionProposal(
                            action="test",
                            target_kind="node",
                            target=int(nbr_id),
                            priority=self.priority,
                            cost_units=self.cost_per_node,
                            params=self.params,
                            source_algo=self.name,
                            reason=f"contact of known case {case_id}",
                        )
                    )
        return proposals


class TraceEdgeEndpoints(AlgorithmBase):
    """
    Crawl the network by tracing the endpoints of edges discovered today.
    """

    name = "trace_edge_endpoints"

    def __init__(
        self,
        *,
        cost_per_node: int = 1,
        priority: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.cost_per_node = int(cost_per_node)
        self.priority = float(priority)
        self.params = params or {}

    def propose(self, state) -> List[ActionProposal]:
        # known_edges format: (u, v, ct_id, discovery_time, source)
        newly_found_nodes = set()
        for u, v, _, t_disc, _ in state.known_edges:
            if t_disc == state.last_t:
                # Only propose tracing endpoints that aren't already known cases
                if not state.known_case[u]:
                    newly_found_nodes.add(u)
                if not state.known_case[v]:
                    newly_found_nodes.add(v)

        if not newly_found_nodes:
            return []

        return [
            ActionProposal(
                action="trace",
                target_kind="node",
                target=int(n),
                priority=self.priority,
                cost_units=self.cost_per_node,
                params=self.params,
                source_algo=self.name,
                reason="crawl discovered edge endpoint",
            )
            for n in newly_found_nodes
        ]


class IsolateNeighborsOfHighDegreeCases(AlgorithmBase):
    """
    Identify newly reported cases, rank them by their known degree,
    and propose isolation for all their known contacts.
    """

    name = "isolate_neighbors_of_high_degree_cases"

    def __init__(
        self,
        *,
        cost_per_node: int = 1,
        priority: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.cost_per_node = int(cost_per_node)
        self.priority = float(priority)
        self.params = params or {}

    def propose(self, state) -> List[ActionProposal]:
        new_cases = getattr(state, "new_cases_today", np.empty(0, np.int32))
        if new_cases.size == 0:
            return []

        # Rank cases by known degree
        case_degrees = [(cid, len(state.known_adj[cid])) for cid in new_cases]
        case_degrees.sort(key=lambda x: x[1], reverse=True)

        proposals = []
        for cid, degree in case_degrees:
            for nbr_info in state.neighbors(cid):
                nbr_id = nbr_info[0]
                proposals.append(
                    ActionProposal(
                        action="isolate",
                        target_kind="node",
                        target=int(nbr_id),
                        priority=self.priority,
                        cost_units=self.cost_per_node,
                        params=self.params,
                        source_algo=self.name,
                        reason=f"contact of high-degree case {cid} (deg={degree})",
                    )
                )
        return proposals
