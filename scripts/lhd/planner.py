#scripts/lhd/planner.py
from __future__ import annotations
from typing import List, Dict, Tuple
from scripts.lhd.response_types import ActionProposal, ActionPlan
"""
Container for various planners, which take a list of action proposals and allocate a set amount of resources toward these actions, filtering them to a list of actions that will be executed
"""

class GreedyPlanner:
    name = "greedy"

    def select(self, proposals: List[ActionProposal], capacity: int) -> ActionPlan:
        capacity = int(capacity)
        if capacity <= 0 or not proposals:
            return ActionPlan(selected=[], capacity_available=capacity, capacity_used=0)

        # dedup (action, target_kind, target)
        best: Dict[Tuple[str, str, int], ActionProposal] = {}
        for p in proposals:
            key = (p.action, p.target_kind, int(p.target))
            if key not in best:
                best[key] = p
            else:
                cur = best[key]
                if (p.priority > cur.priority) or (p.priority == cur.priority and p.cost_units < cur.cost_units):
                    best[key] = p

        uniq = list(best.values())

        def score(p: ActionProposal):
            c = max(int(p.cost_units), 1)
            return (p.priority / c, p.priority, -c, p.action)

        uniq.sort(key=score, reverse=True)

        selected = []
        used = 0
        for p in uniq:
            c = max(int(p.cost_units), 1)
            if used + c <= capacity:
                selected.append(p)
                used += c

        return ActionPlan(selected=selected, capacity_available=capacity, capacity_used=used)