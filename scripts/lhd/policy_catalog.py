#scripts/lhd/policy_catalog.py
from __future__ import annotations
from typing import Dict, Any

from scripts.lhd.algorithms_state import IsolateNewCases, TraceNewCases
from scripts.lhd.planner import GreedyPlanner

#Registry of algorithms currently implemented 
ALGO_REGISTRY = {
    "isolate_new_cases": IsolateNewCases,
    "trace_new_cases": TraceNewCases,
}

#Registry of planners currently implemented
PLANNER_REGISTRY = {
    "greedy": GreedyPlanner,
}


#LHD Policies, which are combinations of algorithms (prioritizing individuals) and planners (resource allocation strategies)
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
}


def build_policy(policy_name: str, *, default_algo_params: dict) -> tuple[list, object]:
    name = str(policy_name or "observe_only")
    spec = POLICIES.get(name, POLICIES["observe_only"])

    planner_cls = PLANNER_REGISTRY[spec.get("planner", "greedy")]
    planner = planner_cls()

    algos = []
    for algo_name, params in spec.get("algorithms", []):
        cls = ALGO_REGISTRY[algo_name]
        # allow injecting sensible defaults
        if algo_name == "isolate_new_cases":
            params = dict(params)
            params.setdefault("params", default_algo_params)
        algos.append(cls(**params))

    return algos, planner, name