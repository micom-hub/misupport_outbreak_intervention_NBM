# scripts/lhd/policy_catalog.py


from __future__ import annotations
from typing import Dict, Any

from scripts.lhd.algorithms_state import (
    IsolateNewCases,
    TraceNewCases,
    TestContactsOfKnownCases,
    TraceEdgeEndpoints,
    IsolateNeighborsOfHighDegreeCases,
)
from scripts.lhd.planner import GreedyPlanner

# Registry of algorithms currently implemented
ALGO_REGISTRY = {
    "isolate_new_cases": IsolateNewCases,
    "trace_new_cases": TraceNewCases,
    "test_contacts_of_known_cases": TestContactsOfKnownCases,
    "trace_edge_endpoints": TraceEdgeEndpoints,
    "isolate_neighbors_of_high_degree_cases": IsolateNeighborsOfHighDegreeCases,
}

# Registry of planners currently implemented
PLANNER_REGISTRY = {
    "greedy": GreedyPlanner,
}


# LHD Policies, which are combinations of algorithms (prioritizing individuals) and planners (resource allocation strategies)
POLICIES: Dict[str, Dict[str, Any]] = {
    "observe_only": {
        "planner": "greedy",
        "algorithms": [],
    },
    "isolate_only": {
        "planner": "greedy",
        "algorithms": [
            ("isolate_new_cases", {}),
        ],
    },
    "trace_only": {
        "planner": "greedy",
        "algorithms": [
            ("trace_new_cases", {}),
        ],
    },
    "trace_then_isolate": {
        "planner": "greedy",
        "algorithms": [
            ("trace_new_cases", {}),
            ("isolate_new_cases", {}),
        ],
    },
    "trace_and_test": {
        "planner": "greedy",
        "algorithms": [
            ("isolate_new_cases", {"priority": 2.0}),  # Highest priority
            ("trace_new_cases", {"priority": 1.5}),
            ("test_contacts_of_known_cases", {"priority": 1.0}),
        ],
    },
    "network_crawl": {
        "planner": "greedy",
        "algorithms": [
            ("trace_new_cases", {}),
            ("trace_edge_endpoints", {}),
        ],
    },
    "network_crawl_isolate": {
        "planner": "greedy",
        "algorithms": [
            ("trace_edge_endpoints", {}),
            ("isolate_new_cases", {"priority": 2.0}),
            ("isolate_neighbors_of_high_degree_cases", {"priority": 1.5}),
        ],
    },
}


def build_policy(policy_name: str, *, default_algo_params: dict) -> tuple[list, object]:
    name = str(policy_name or "observe_only")
    spec = POLICIES.get(name, POLICIES["observe_only"])

    planner_cls = PLANNER_REGISTRY[spec.get("planner", "greedy")]
    planner = planner_cls()

    algos = []
    for algo_name, params in spec.get("algorithms", []):
        cls = ALGO_REGISTRY[algo_name]

        # Get the default configuration for this algorithm (cost, priority, and action-specific params)
        algo_config = default_algo_params.get(algo_name, {})
        # Policy-specific overrides (from POLICIES dict) can be merged here if needed, but currently 'params' is empty.
        algo_params = {
            **algo_config,
            **params,
        }  # Merge policy-specific params (if any) over defaults
        algos.append(cls(**algo_params))

    return algos, planner, name
