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
                action="trace_contacts",
                target_kind="node",
                target=int(n),
                priority=self.priority,
                cost_units=max(self.cost_per_case, 1),
                params=self.params,
                source_algo=self.name,
                reason="trace contacts for newly reported case",
            )
            for n in nodes
        ]