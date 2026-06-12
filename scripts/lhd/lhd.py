# scripts/lhd/lhd.py
from typing import TYPE_CHECKING, Optional, Dict, List, Any
from collections import defaultdict, Counter
import numpy as np
import pandas as pd  # only if you want results_to_df here; otherwise import inside method

from scripts.lhd.state import LHDState
from scripts.lhd.surveillance import SurveillanceModel
from scripts.lhd.policy_catalog import build_policy
from scripts.lhd.executor import Executor
from scripts.lhd.tokens import MultiplierToken
from scripts.lhd.response_types import ActionProposal, ActionPlan, ExecutionSummary


if TYPE_CHECKING:
    from scripts.simulation.outbreak_model import NetworkModel


class LocalHealthDepartment:
    """
    Current LHD (3/17):

    - step(t, epi_state) 
        - Expires actions due to expire
        - Observes updated epi_state through surveillance
        - Updates LHD knowledge
        - Uses algorithms to propose actions
        - Uses planner to allocate resources to actions
        - Executes actions

    - Algorithms use only the current LHDState to prioritize individuals
    - Planner selects actions to remain under capacity
    - Executor applies control actions (multipliers on in/out transmission) and info (surveillance orders)
    
    - LHD now builds its own daily results frame to hand to outbreak_model for cost-efficiency metrics
    """
    def __init__(
        self,
        *,
        seed: int,
        surv_seed: int,
        model: NetworkModel,
        capacity: Optional[int] = None,
        policy_name: Optional[str] = None,
    ):
        self.model = model
        self.seed = seed
        self.surv_seed = surv_seed
        self.rng = np.random.default_rng(seed)

        # capacity
        self.daily_capacity = int(capacity) if capacity is not None else int(self.model.config.lhd.lhd_daily_capacity)

        # baseline surveillance parameters
        self.p_detect_inf = self.model.config.lhd.p_detect_inf
        self.report_delay_days = self.model.config.lhd.report_delay_days

        # Default parameter values (control + info)
        self.min_factor = 1e-6
        self.default_iso_reduction = float(self.model.config.lhd.lhd_default_int_reduction)
        self.default_iso_duration = int(self.model.config.lhd.lhd_default_int_duration)
        self.default_iso_contact_types = ["cas", "sch", "wp"]

        self.default_trace_params = {
            "delay_days": 0, #do day-of tracing
            "recall_prob": 0.25, #25% recall prob for any given contact
            "max_per_case": 25,  #limit on total recalled
            "contact_types": ["hh", "sch", "wp"],
        }

        self.default_test_params = {
            "delay_days": 0, #day-of testing
            "sens_pre": 0.5, #pick up half of pre-infectious
            "sens_inf": 0.99, #pick up 99% of post-infectious
            "spec": 1.0, #no false-positives (yet)
            "report_delay_days": self.report_delay_days,
        }
        # Instantiate surveillance object
        self.surveillance = SurveillanceModel(
            neighbor_map=self.model.neighbor_map,
            ct_to_id=self.model.ct_to_id,
            seed=self.surv_seed,
            N=self.model.N,
            ages=self.model.ages,
            is_vax=self.model.is_vaccinated,
            p_detect_inf=self.p_detect_inf,
            report_delay_days=self.report_delay_days,
        )

        # Instantiate LHDState
        self.state = LHDState(N = self.model.N)

        # Assemble the LHD policy (algo + planner)
        cfg_name = getattr(self.model.config.lhd, "policy_name", "observe_only")
        self.policy_name = str(policy_name or cfg_name)

        default_algo_params = {
            "isolate_new_cases": {
                "cost_per_case": 1,
                "priority": 1.0,
                "params": {  # These are the parameters for the _apply_isolation method
                    "reduction": self.default_iso_reduction,
                    "duration": self.default_iso_duration,
                    "contact_types": self.default_iso_contact_types,
                },
            },
            "trace_new_cases": {
                "cost_per_case": 1,
                "priority": 1.0,
                "params": self.default_trace_params,  # Pass the actual trace params here
            },
        }
        self.algorithms, self.planner, self.policy_name = build_policy(
            self.policy_name, default_algo_params=default_algo_params
        )

        self.executor = Executor()

        # expiry tokens day -> list[tokens]
        self._expiry_tokens_by_day = defaultdict(list)

        self._results_rows = []
        self._action_log = []
        self._lhd_daily_log = []

    # -------------------------------------------
    # Helpers for scheduling and expiring actions
    # -------------------------------------------
    def process_expirations(self, t: int) -> int:
        """
        Revert control tokens due at day t. Returns count expired.
        """
        due = self._expiry_tokens_by_day.pop(int(t), [])
        for tok in due:
            tok.revert(self.model)
        return int(len(due))

    # ------------
    # Surveillance
    # ------------
    def observe(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
        scheduled_actions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        # Provide state change to surveillance, get back observations
        return self.surveillance.step(
            t=int(t), epi_state=epi_state, scheduled_actions=scheduled_actions
        )

    # ---------------
    # Action Handling
    # ---------------
    def _apply_isolation(
        self, *, t: int, nodes: np.ndarray, params: Dict[str, Any]
    ) -> tuple[int, int]:
        """
        For a given set of nodes and isolation parameters, order isolation
        """
        nodes = np.asarray(nodes, dtype=np.int32)
        if nodes.size == 0:
            return 0, 0

        # merge params with defaults
        reduction = float(params.get("reduction", self.default_iso_reduction))
        duration = int(params.get("duration", self.default_iso_duration))
        cts = params.get("contact_types", self.default_iso_contact_types)

        # reduction is fraction removed (reduction .2 means .8 left)
        reduction = min(max(reduction, 0.0), 1.0)
        factor = max(self.min_factor, 1.0 - reduction)

        # apply in/out multipliers
        for ct in cts:
            if ct in self.model.in_multiplier:
                self.model.in_multiplier[ct][nodes] *= factor
            if ct in self.model.out_multiplier:
                self.model.out_multiplier[ct][nodes] *= factor

        tokens_added = 0
        if duration > 0:
            tok = MultiplierToken(
                expires_at=int(t + duration),
                nodes=nodes.copy(),
                contact_types=tuple(cts),
                in_factor=factor,
                out_factor=factor,
                action="isolate",
            )
            self._schedule_token(tok)
            tokens_added = 1

        # Log specific actions to the node-level log
        for node in nodes:
            self._action_log.append(
                {
                    "t": t,
                    "node": int(node),
                    "action": "isolate",
                    "duration": duration,
                    "reduction": reduction,
                }
            )

        return int(nodes.size), tokens_added

    def _order_trace(
        self, *, t: int, cases: np.ndarray, params: Dict[str, Any]
    ) -> tuple[int, int]:
        # Order contact tracing on given node(s)
        cases = np.asarray(cases, dtype=np.int32)
        if cases.size == 0:
            return 0, 0

        # Ensure params is a dict and merge carefully
        passed_params = params if isinstance(params, dict) else {}
        merged = {**self.default_trace_params, **passed_params}

        self.surveillance.order_trace(t=int(t), cases=cases, params=merged)

        for node in cases:
            self._action_log.append(
                {"t": t, "node": int(node), "action": "trace", "params": merged}
            )
        return int(cases.size), 0

    def _order_test(
        self, *, t: int, nodes: np.ndarray, params: Dict[str, Any]
    ) -> tuple[int, int]:
        # Order tests for given node(s)
        nodes = np.asarray(nodes, dtype=np.int32)
        if nodes.size == 0:
            return 0, 0
        merged = dict(self.default_test_params)
        merged.update(params or {})
        self.surveillance.order_test(t=int(t), nodes=nodes, params=merged)

        for node in nodes:
            self._action_log.append(
                {"t": t, "node": int(node), "action": "test", "params": merged}
            )
        return int(nodes.size), 0

    def step(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
        scheduled_surv_actions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        One step for the LHD where it:
        1) Processes old interventions
        2) Observes reported cases through surveillance
        3)
        """

        t = int(t)

        # 1) Process old interventions that are expiring
        expired = self.process_expirations(t)

        # 2) Conduct surveillance on daily updates
        batch = self.observe(
            t=t, epi_state=epi_state, scheduled_actions=scheduled_surv_actions
        )

        # 3) Integrate findings to knowledge state
        self.state.process_batch(batch)

        # 4) Have algorithms propose actions to take
        proposals = []
        for algo in self.algorithms:
            proposals.extend(algo.propose(self.state))

        # 5) Use planner to allocate resources to proposals
        plan: ActionPlan = self.planner.select(proposals, capacity=self.daily_capacity)

        # 7) Pass ActionPlan to executor
        exec_summary: ExecutionSummary = self.executor.execute(lhd=self, t=t, plan=plan)
        exec_summary.tokens_expired = expired

        # 8) Record results of the day
        self._log_day(
            t=t,
            batch=batch,
            proposals=proposals,
            plan=plan,
            summary=exec_summary,
            run_number=self.model.replicate_ind,
        )

        return batch

    # Results writer helper
    def _log_day(
        self,
        *,
        t: int,
        batch: Dict[str, np.ndarray],
        proposals: List[ActionProposal],
        plan: ActionPlan,
        run_number: int,
        summary: ExecutionSummary,
    ) -> None:
        rep = np.asarray(
            batch.get("reported_cases", np.empty(0, np.int32)), dtype=np.int32
        )
        proposed_actions = Counter([p.action for p in proposals])
        selected_actions = Counter([p.action for p in plan.selected])

        # Extract nodes for specific actions
        nodes_isolated_today = [
            p.target
            for p in plan.selected
            if p.action == "isolate" and p.target_kind == "node"
        ]
        nodes_contact_traced_today = [
            p.target
            for p in plan.selected
            if p.action == "trace" and p.target_kind == "node"
        ]

        # Extract newly discovered nodes from tracing
        trace_src = batch.get("trace_src", np.empty(0, dtype=np.int32))
        trace_tgt = batch.get("trace_tgt", np.empty(0, dtype=np.int32))
        newly_discovered_trace_nodes = (
            np.unique(np.concatenate([trace_src, trace_tgt])).tolist()
            if trace_src.size > 0
            else []
        )

        row = {
            "run_number": run_number,
            "t": int(t),
            "policy_name": self.policy_name,
            "capacity_available": int(plan.capacity_available),
            "capacity_used": int(plan.capacity_used),
            "reported_cases_today": int(rep.size),
            "new_cases_today": int(
                getattr(self.state, "new_cases_today", np.empty(0, np.int32)).size
            ),
            "known_cases_total": int(len(getattr(self.state, "known_case_list", []))),
            "new_edges_today": int(getattr(self.state, "new_edges_today", 0)),
            "known_edges_total": int(len(getattr(self.state, "known_edges", []))),
            "proposals_total": int(len(proposals)),
            "selected_total": int(len(plan.selected)),
            "tokens_scheduled": int(summary.tokens_scheduled),
            "tokens_expired": int(summary.tokens_expired),
            "isolation_actions_taken_today": summary.applied_by_action.get(
                "isolate", 0
            ),
            "nodes_isolated_today": nodes_isolated_today,
            "tracing_actions_taken_today": summary.info_orders_by_action.get(
                "trace", 0
            ),
            "nodes_contact_traced_today": nodes_contact_traced_today,
            "newly_reported_cases_today": rep.tolist(),
            "newly_discovered_trace_nodes_today": newly_discovered_trace_nodes,
        }

        # flatten action counts
        for k, v in proposed_actions.items():
            row[f"proposed_{k}"] = int(v)
        for k, v in selected_actions.items():
            row[f"selected_{k}"] = int(v)
        for k, v in summary.applied_by_action.items():
            row[f"applied_{k}"] = int(v)
        for k, v in summary.info_orders_by_action.items():
            row[f"info_{k}"] = int(v)

        self._results_rows.append(row)
        self._lhd_daily_log.append(row)

    def lhd_daily_log_to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._lhd_daily_log)

    def results_to_df(self) -> pd.DataFrame:
        # Export results to dataframe
        return pd.DataFrame(self._results_rows)

    def action_log_to_df(self) -> pd.DataFrame:
        # Export detailed node-level actions to dataframe
        return pd.DataFrame(self._action_log)

    def _schedule_token(self, tok: MultiplierToken) -> None:
        self._expiry_tokens_by_day[int(tok.expires_at)].append(tok)

    def _as_nodes(self, targets) -> np.ndarray:
        return np.asarray(targets, dtype=np.int32)

    # Reset Helper
    def reset_for_run(self):
        """
        Reset LHD state for new model run
        """

        self._expiry_tokens_by_day.clear()
        self._results_rows.clear()
        self._action_log.clear()
        self._lhd_daily_log.clear()
        if hasattr(self, "surveillance") and self.surveillance is not None:
            self.surveillance.reset_for_run(seed=self.surveillance.seed, is_vax = self.model.is_vaccinated)
        if hasattr(self, "state") and self.state is not None:
            self.state.reset_for_run()
