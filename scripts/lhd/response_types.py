#scripts/lhd/response_types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

#Responses may target nodes or locations
TargetKind = Literal["node", "location"]

#Algorithms return proposals, which contain an action, who it should be done to, its relative priority, and its relative cost
@dataclass(frozen=True)
class ActionProposal:
    action: str
    target_kind: TargetKind
    target: Any
    priority: float
    cost_units: int
    params: Dict[str, Any] = field(default_factory=dict)
    source_algo: str = ""
    reason: str = ""

@dataclass
class ActionPlan:
    selected: List[ActionProposal]
    capacity_available: int
    capacity_used: int

@dataclass
class ExecutionSummary:
    attempted_by_action: Dict[str, int] = field(default_factory=dict)
    applied_by_action: Dict[str, int] = field(default_factory=dict)
    info_orders_by_action: Dict[str, int] = field(default_factory=dict)
    tokens_scheduled: int = 0