from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple


import numpy as np

from scripts.utils.dict_utilities import params_key
from scripts.lhd.response_types import ActionPlan, ExecutionSummary, ActionProposal


#Class to execute actions, contains logic for each action and target kind, with how to do them
class Executor:
    def execute(self, *, lhd, t: int, plan: ActionPlan) -> ExecutionSummary:
        summary = ExecutionSummary()
        if not plan.selected:
            return summary

        groups = defaultdict(list)
        params_map = {}
        for p in plan.selected:
            k = (p.action, p.target_kind, params_key(p.params))
            groups[k].append(p.target)
            params_map[k] = p.params

        for (action, target_kind, _), targets in groups.items():
            summary.attempted_by_action[action] = summary.attempted_by_action.get(action, 0) + len(targets)
            params = params_map[(action, target_kind, _)]

            if action == "isolate" and target_kind == "node":
                nodes = lhd._as_nodes(targets)
                applied, tokens_added = lhd._apply_isolation(t=t, nodes=nodes, params=params)
                summary.applied_by_action[action] = summary.applied_by_action.get(action, 0) + applied
                summary.tokens_scheduled += tokens_added

            elif action == "trace_contacts" and target_kind == "node":
                nodes = lhd._as_nodes(targets)
                lhd.surveillance.order_trace(t=t, cases=nodes, params=params)
                summary.info_orders_by_action[action] = summary.info_orders_by_action.get(action, 0) + int(nodes.size)

            elif action == "test_nodes" and target_kind == "node":
                nodes = lhd._as_nodes(targets)
                lhd.surveillance.order_test(t=t, nodes=nodes, params=params)
                summary.info_orders_by_action[action] = summary.info_orders_by_action.get(action, 0) + int(nodes.size)

        return summary

