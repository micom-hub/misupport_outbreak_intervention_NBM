# scripts/lhd/lhd.py
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.lhd.executor import TaskExecutor
from scripts.lhd.planner import WorkScheduler
from scripts.lhd.policy_catalog import get_policy_spec
from scripts.lhd.state import LHDState
from scripts.lhd.surveillance import SurveillanceModel
from scripts.lhd.tasks import LHDTask
from scripts.lhd.tokens import (
    GlobalMultiplierToken,
    MultiplierToken,
)


class LocalHealthDepartment:
    """
    Capacity-constrained local health department response.

    Each day, surveillance updates the LHD's partial knowledge, new reports
    generate operational tasks, the scheduler allocates available work units,
    and unfinished work remains in a persistent backlog.
    """

    def __init__(
        self,
        *,
        seed: int,
        surv_seed: int,
        model,
        capacity: Optional[float] = None,
        policy_name: Optional[str] = None,
    ):
        self.model = model
        self.seed = int(seed)
        self.surv_seed = int(surv_seed)
        self.rng = np.random.default_rng(self.seed)

        cfg = self.model.config.lhd

        self.daily_capacity = float(
            capacity
            if capacity is not None
            else cfg.lhd_daily_capacity
        )
        self.policy_name = str(
            policy_name or cfg.policy_name
        )
        self.policy_spec = get_policy_spec(
            self.policy_name
        )

        # Policy phases:
        # 0 = routine surveillance
        # 1 = case investigation and tracing
        # 2 = escalated emergency response
        self.policy_phase = (
            1
            if self.policy_spec["perfect_information"]
            else 0
        )
        self.phase1_start_day: Optional[int] = (
            0
            if self.policy_spec["perfect_information"]
            else None
        )
        self.escalation_order_day: Optional[int] = None
        self.escalation_effective_day: Optional[int] = None
        self.escalation_end_day: Optional[int] = None
        self.escalation_trigger_name: Optional[str] = None
        self.escalation_trigger_value: Optional[float] = None

        self._trigger_streak = 0
        self._scheduled_escalations_by_day = defaultdict(list)
        self._expiry_tokens_by_day = defaultdict(list)

        # Counters for work performed before the regular LHD step.
        self._tokens_expired_today = 0
        self._tasks_cancelled_before_step = 0

        self.min_factor = 1e-6

        # Individual isolation deliberately leaves household and
        # group-quarter contacts unchanged.
        self.default_iso_reduction = float(
            cfg.lhd_default_int_reduction
        )
        self.default_iso_duration = int(
            cfg.lhd_default_int_duration
        )
        self.default_iso_contact_types = [
            "cas",
            "sch",
            "wp",
        ]

        self.default_trace_params = {
            "delay_days": int(
                cfg.trace_delay_days
            ),
            "recall_prob": float(
                cfg.trace_recall_prob
            ),
            "max_per_case": int(
                cfg.trace_max_per_case
            ),
            "contact_types": [
                "hh",
                "sch",
                "wp",
            ],
        }

        self.default_test_params = {
            "delay_days": int(
                cfg.test_turnaround_days
            ),
            "sensitivity_exposed": float(
                cfg.test_sensitivity_exposed
            ),
            "sensitivity_infectious": float(
                cfg.test_sensitivity_infectious
            ),
            "specificity": float(
                cfg.test_specificity
            ),
            "report_delay_days": int(
                cfg.test_report_delay_days
            ),
        }

        self.task_costs = {
            "case_investigation": float(
                cfg.case_investigation_cost
            ),
            "isolation_support": float(
                cfg.isolation_support_cost
            ),
            "contact_interview": float(
                cfg.contact_interview_cost
            ),
            "contact_notification": float(
                cfg.contact_notification_cost
            ),
            "testing_referral": float(
                cfg.testing_referral_cost
            ),
            "automated_notification": float(
                cfg.automated_notification_cost
            ),
        }

        self.task_priorities = {
            "case_investigation": float(
                cfg.case_investigation_priority
            ),
            "isolation_support": float(
                cfg.isolation_support_priority
            ),
            "contact_interview": float(
                cfg.contact_interview_priority
            ),
            "contact_notification": float(
                cfg.contact_notification_priority
            ),
            "testing_referral": float(
                cfg.testing_referral_priority
            ),
            "automated_notification": float(
                cfg.automated_notification_priority
            ),
        }

        self.task_deadline_days = {
            "case_investigation": int(
                cfg.case_investigation_deadline_days
            ),
            "isolation_support": int(
                cfg.isolation_support_deadline_days
            ),
            "contact_interview": int(
                cfg.contact_interview_deadline_days
            ),
            "contact_notification": int(
                cfg.contact_notification_deadline_days
            ),
            "testing_referral": int(
                cfg.testing_referral_deadline_days
            ),
            "automated_notification": int(
                cfg.automated_notification_deadline_days
            ),
        }

        self.surveillance = SurveillanceModel(
            neighbor_map=self.model.neighbor_map,
            ct_to_id=self.model.ct_to_id,
            seed=self.surv_seed,
            N=self.model.N,
            ages=self.model.ages,
            is_vax=self.model.is_vaccinated,
            p_detect_symptomatic_daily=float(
                cfg.p_detect_symptomatic_daily
            ),
            p_detect_asymptomatic_daily=float(
                cfg.p_detect_asymptomatic_daily
            ),
            p_detect_imported_daily=float(
                cfg.p_detect_imported_daily
            ),
            imported_detection_start_day=int(
                cfg.imported_detection_start_day
            ),
            passive_report_delay_days=int(
                cfg.passive_report_delay_days
            ),
        )
        self.surveillance.attach_model(self.model)

        self.state = LHDState(N=self.model.N)
        self.scheduler = WorkScheduler(
            aging_priority_per_day=float(
                cfg.task_aging_priority_per_day
            )
        )
        self.executor = TaskExecutor()

        self._daily_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Daily workflow
    # ------------------------------------------------------------------

    def pre_epi_step(
        self,
        day: int,
    ) -> None:
        """
        Apply changes that must occur before today's transmission.
        """
        day = int(day)

        self._tokens_expired_today = (
            self.process_expirations(day)
        )
        self._tasks_cancelled_before_step = 0

        self._activate_scheduled_escalations(day)

        if self.policy_spec["perfect_information"]:
            self._apply_perfect_information_response(
                day
            )

    def observe(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
        scheduled_actions: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, np.ndarray]:
        return self.surveillance.step(
            t=int(t),
            epi_state=epi_state,
            scheduled_actions=scheduled_actions,
        )

    def step(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
        scheduled_surv_actions: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, np.ndarray]:
        """Process one post-transmission LHD workday."""
        day = int(t)

        pending_tasks_start = len(
            self.state.task_queue.pending_tasks()
        )
        pending_units_start = (
            self.state.task_queue.pending_work_units()
        )

        batch = self.observe(
            t=day,
            epi_state=epi_state,
            scheduled_actions=scheduled_surv_actions,
        )
        self.state.process_batch(batch)

        reported_count = self._array_size(
            batch,
            "reported_cases",
        )
        self._update_response_phase_from_reports(
            day=day,
            reported_count=reported_count,
        )

        new_report_tasks = self._ingest_new_reports(
            day=day
        )
        new_contact_tasks = (
            self._ingest_elicited_contacts(
                day=day,
                batch=batch,
            )
        )
        expired_tasks = self._expire_tasks(
            day=day
        )

        schedule = self.scheduler.select(
            self.state.task_queue.eligible_tasks(
                day
            ),
            capacity=self.daily_capacity,
            day=day,
        )

        (
            completed_tasks,
            cancelled_tasks,
            work_units_used,
            tokens_scheduled,
            child_tasks_created,
        ) = self._execute_schedule(
            day=day,
            selected_tasks=schedule.selected,
        )

        row = self._build_daily_row(
            day=day,
            batch=batch,
            schedule=schedule,
            pending_tasks_start=pending_tasks_start,
            pending_units_start=pending_units_start,
            new_report_tasks=new_report_tasks,
            new_contact_tasks=new_contact_tasks,
            child_tasks_created=child_tasks_created,
            completed_tasks=completed_tasks,
            cancelled_tasks=cancelled_tasks,
            expired_tasks=expired_tasks,
            work_units_used=work_units_used,
            tokens_scheduled=tokens_scheduled,
        )

        trigger_value = (
            self._evaluate_escalation_trigger(
                day=day,
                current_row=row,
            )
        )

        row.update(
            {
                "policy_phase": int(
                    self.policy_phase
                ),
                "escalation_trigger_metric": (
                    self.model.config.lhd
                    .escalation_trigger_metric
                    if self.policy_spec[
                        "adaptive_escalation"
                    ]
                    else None
                ),
                "escalation_trigger_value":
                    trigger_value,
                "escalation_triggered_today":
                    bool(
                        self.escalation_order_day
                        == day
                    ),
                "escalation_order_day":
                    self.escalation_order_day,
                "escalation_effective_day":
                    self.escalation_effective_day,
                "escalation_active": bool(
                    self.policy_phase == 2
                ),
                "escalation_profile": (
                    self.model.config.lhd
                    .escalation_intervention_profile
                    if (
                        self.policy_phase == 2
                        or self.escalation_order_day
                        is not None
                    )
                    else None
                ),
            }
        )

        self._daily_log.append(row)
        return batch

    def _execute_schedule(
        self,
        *,
        day: int,
        selected_tasks: List[LHDTask],
    ) -> Tuple[
        List[LHDTask],
        List[LHDTask],
        float,
        int,
        int,
    ]:
        completed_tasks: List[LHDTask] = []
        cancelled_tasks: List[LHDTask] = []
        work_units_used = 0.0
        tokens_scheduled = 0
        child_tasks_created = 0

        for task in selected_tasks:
            self.state.task_queue.mark_selected(
                task.task_id,
                day,
            )
            self._record_task_event(
                day=day,
                event_type="task_selected",
                task=task,
            )

            self.state.task_queue.mark_started(
                task.task_id,
                day,
            )
            self._record_task_event(
                day=day,
                event_type="task_started",
                task=task,
            )

            result = self.executor.execute(
                lhd=self,
                day=day,
                task=task,
            )

            if result.success:
                self.state.task_queue.mark_completed(
                    task.task_id,
                    day,
                )
                completed_tasks.append(task)
                work_units_used += float(
                    result.work_units_used
                )
                tokens_scheduled += int(
                    result.tokens_scheduled
                )
                child_tasks_created += int(
                    result.child_tasks_created
                )

                self._record_task_event(
                    day=day,
                    event_type="task_completed",
                    task=task,
                    metadata={
                        "work_units_used":
                            result.work_units_used,
                    },
                )
            else:
                self.state.task_queue.mark_cancelled(
                    task.task_id,
                    day,
                    result.message,
                )
                cancelled_tasks.append(task)

                self._record_task_event(
                    day=day,
                    event_type="task_cancelled",
                    task=task,
                    metadata={
                        "reason": result.message,
                    },
                )

        return (
            completed_tasks,
            cancelled_tasks,
            work_units_used,
            tokens_scheduled,
            child_tasks_created,
        )

    # ------------------------------------------------------------------
    # Persistent tasks
    # ------------------------------------------------------------------

    def enqueue_task(
        self,
        *,
        task_type: str,
        target: int,
        created_day: int,
        source_case: Optional[int] = None,
        reason: str = "",
        source_algorithm: str = "workflow",
        params: Optional[Dict[str, Any]] = None,
        earliest_start_day: Optional[int] = None,
    ) -> Tuple[LHDTask, bool]:
        if task_type not in self.task_costs:
            raise ValueError(
                "No work-unit cost configured for "
                f"task type '{task_type}'"
            )

        created_day = int(created_day)

        if earliest_start_day is None:
            earliest_start_day = created_day + 1

        deadline_offset = (
            self.task_deadline_days.get(
                task_type
            )
        )
        deadline_day = (
            None
            if deadline_offset is None
            else created_day
            + int(deadline_offset)
        )

        task_params = self._task_parameters(
            task_type=task_type,
            params=params,
        )

        task = LHDTask(
            task_id=(
                self.state.task_queue.next_task_id()
            ),
            task_type=task_type,
            target_kind="node",
            target=int(target),
            source_case=(
                None
                if source_case is None
                else int(source_case)
            ),
            created_day=created_day,
            earliest_start_day=int(
                earliest_start_day
            ),
            deadline_day=deadline_day,
            base_priority=float(
                self.task_priorities[task_type]
            ),
            work_cost=float(
                self.task_costs[task_type]
            ),
            source_algorithm=source_algorithm,
            reason=reason,
            params=task_params,
        )

        stored_task, created = (
            self.state.task_queue.add(task)
        )

        if created:
            self._record_task_event(
                day=created_day,
                event_type="task_created",
                task=task,
                metadata={
                    "earliest_start_day":
                        task.earliest_start_day,
                    "deadline_day":
                        task.deadline_day,
                    "reason": reason,
                },
            )

        return stored_task, created

    def _task_parameters(
        self,
        *,
        task_type: str,
        params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        overrides = dict(params or {})

        if task_type == "isolation_support":
            return {
                "reduction":
                    self.default_iso_reduction,
                "duration":
                    self.default_iso_duration,
                "contact_types":
                    list(
                        self.default_iso_contact_types
                    ),
                **overrides,
            }

        if task_type == "contact_interview":
            values = {
                **self.default_trace_params,
                **overrides,
            }

            if (
                self.policy_phase == 2
                and self.model.config.lhd
                .phase2_trace_mode
                == "targeted"
            ):
                values["contact_types"] = ["hh"]

            return values

        if task_type == "testing_referral":
            return {
                **self.default_test_params,
                **overrides,
            }

        if task_type == "automated_notification":
            return {
                "reduction": float(
                    self.model.config.lhd
                    .automated_notification_reduction
                ),
                "duration": int(
                    self.model.config.lhd
                    .automated_notification_duration_days
                ),
                "contact_types": [
                    "cas",
                    "sch",
                    "wp",
                ],
                **overrides,
            }

        return overrides

    def _ingest_new_reports(
        self,
        *,
        day: int,
    ) -> int:
        if (
            self.policy_spec[
                "perfect_information"
            ]
            or not self.policy_spec[
                "investigate_cases"
            ]
        ):
            return 0

        created = 0

        for node in self.state.new_cases_today:
            _, was_created = self.enqueue_task(
                task_type="case_investigation",
                target=int(node),
                source_case=int(node),
                created_day=day,
                earliest_start_day=day,
                reason="new report received by LHD",
                source_algorithm="report_intake",
            )
            created += int(was_created)

        return created

    def _ingest_elicited_contacts(
        self,
        *,
        day: int,
        batch: Dict[str, np.ndarray],
    ) -> int:
        sources = np.asarray(
            batch.get(
                "trace_src",
                np.empty(0, dtype=np.int32),
            ),
            dtype=np.int32,
        )
        targets = np.asarray(
            batch.get(
                "trace_tgt",
                np.empty(0, dtype=np.int32),
            ),
            dtype=np.int32,
        )
        contact_types = np.asarray(
            batch.get(
                "trace_ct",
                np.empty(0, dtype=np.int16),
            ),
            dtype=np.int16,
        )

        created = 0

        for source, target, type_id in zip(
            sources,
            targets,
            contact_types,
        ):
            source = int(source)
            target = int(target)
            type_id = int(type_id)

            contact_type = (
                self.model.id_to_ct.get(
                    type_id,
                    str(type_id),
                )
            )

            self.model.record_event(
                day=day,
                event_type="contact_elicited",
                node_id=target,
                source_node_id=source,
                contact_type=contact_type,
            )

            if (
                self.policy_phase == 2
                and self.policy_spec[
                    "automated_notification_in_phase2"
                ]
            ):
                task_type = (
                    "automated_notification"
                )
            elif self.policy_spec[
                "notify_contacts"
            ]:
                task_type = (
                    "contact_notification"
                )
            else:
                continue

            _, was_created = self.enqueue_task(
                task_type=task_type,
                target=target,
                source_case=source,
                created_day=day,
                earliest_start_day=day,
                reason=(
                    f"contact elicited from "
                    f"case {source}"
                ),
                source_algorithm="trace_result",
                params={
                    "contact_type": contact_type,
                },
            )
            created += int(was_created)

        return created

    def _expire_tasks(
        self,
        *,
        day: int,
    ) -> List[LHDTask]:
        expired = (
            self.state.task_queue
            .expire_overdue(day)
        )

        for task in expired:
            self._record_task_event(
                day=day,
                event_type="task_expired",
                task=task,
                metadata={
                    "reason":
                        task.expiration_reason,
                },
            )

        return expired

    def record_task_semantic_event(
        self,
        *,
        day: int,
        event_type: str,
        task: LHDTask,
        node_id: Optional[int] = None,
        source_node_id: Optional[int] = None,
    ) -> None:
        self.model.record_event(
            day=day,
            event_type=event_type,
            node_id=(
                int(task.target)
                if node_id is None
                else int(node_id)
            ),
            source_node_id=(
                task.source_case
                if source_node_id is None
                else source_node_id
            ),
            task_id=task.task_id,
            metadata={
                "task_type": task.task_type,
                "work_cost": task.work_cost,
                "queue_age": task.age(day),
            },
        )

    def _record_task_event(
        self,
        *,
        day: int,
        event_type: str,
        task: LHDTask,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        event_metadata = {
            "task_type": task.task_type,
            "work_cost": task.work_cost,
            "queue_age": task.age(day),
        }
        event_metadata.update(metadata or {})

        self.model.record_event(
            day=day,
            event_type=event_type,
            node_id=int(task.target),
            source_node_id=task.source_case,
            task_id=task.task_id,
            metadata=event_metadata,
        )

    # ------------------------------------------------------------------
    # Individual interventions and surveillance orders
    # ------------------------------------------------------------------

    def _apply_isolation(
        self,
        *,
        t: int,
        nodes: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[int, int]:
        nodes = np.unique(
            np.asarray(nodes, dtype=np.int32)
        )

        if nodes.size == 0:
            return 0, 0

        reduction = float(
            np.clip(
                params.get(
                    "reduction",
                    self.default_iso_reduction,
                ),
                0.0,
                1.0,
            )
        )
        duration = int(
            params.get(
                "duration",
                self.default_iso_duration,
            )
        )
        contact_types = list(
            params.get(
                "contact_types",
                self.default_iso_contact_types,
            )
        )
        factor = max(
            self.min_factor,
            1.0 - reduction,
        )

        eligible = nodes[
            self.state.isolated_until[nodes]
            <= t
        ]

        if eligible.size == 0:
            return 0, 0

        self._multiply_node_contacts(
            nodes=eligible,
            contact_types=contact_types,
            factor=factor,
        )

        expires_at = int(
            t + max(duration, 1)
        )
        self.state.isolated_until[
            eligible
        ] = expires_at

        tokens_added = 0

        if duration > 0:
            self._schedule_token(
                MultiplierToken(
                    expires_at=expires_at,
                    nodes=eligible.copy(),
                    contact_types=tuple(
                        contact_types
                    ),
                    in_factor=factor,
                    out_factor=factor,
                    action="isolate",
                )
            )
            tokens_added = 1

        for node in eligible:
            self.model.record_event(
                day=t,
                event_type="isolation_started",
                node_id=int(node),
                metadata={
                    "duration": duration,
                    "expires_at": expires_at,
                    "reduction": reduction,
                    "contact_types":
                        contact_types,
                },
            )

        return int(eligible.size), tokens_added

    def _apply_automated_notification(
        self,
        *,
        t: int,
        nodes: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[int, int]:
        nodes = np.unique(
            np.asarray(nodes, dtype=np.int32)
        )

        if nodes.size == 0:
            return 0, 0

        reduction = float(
            np.clip(
                params.get("reduction", 0.30),
                0.0,
                1.0,
            )
        )
        duration = max(
            1,
            int(params.get("duration", 5)),
        )
        contact_types = list(
            params.get(
                "contact_types",
                ["cas", "sch", "wp"],
            )
        )
        factor = max(
            self.min_factor,
            1.0 - reduction,
        )

        eligible = nodes[
            self.state.quarantined_until[
                nodes
            ] <= t
        ]

        if eligible.size == 0:
            return 0, 0

        self._multiply_node_contacts(
            nodes=eligible,
            contact_types=contact_types,
            factor=factor,
        )

        expires_at = int(t + duration)
        self.state.quarantined_until[
            eligible
        ] = expires_at

        self._schedule_token(
            MultiplierToken(
                expires_at=expires_at,
                nodes=eligible.copy(),
                contact_types=tuple(
                    contact_types
                ),
                in_factor=factor,
                out_factor=factor,
                action="automated_notification",
            )
        )

        for node in eligible:
            self.model.record_event(
                day=t,
                event_type=(
                    "automated_notification_"
                    "reduction_started"
                ),
                node_id=int(node),
                metadata={
                    "reduction": reduction,
                    "expires_at": expires_at,
                    "contact_types":
                        contact_types,
                },
            )

        return int(eligible.size), 1

    def _multiply_node_contacts(
        self,
        *,
        nodes: np.ndarray,
        contact_types: List[str],
        factor: float,
    ) -> None:
        for contact_type in contact_types:
            if contact_type in self.model.in_multiplier:
                self.model.in_multiplier[
                    contact_type
                ][nodes] *= factor

            if contact_type in self.model.out_multiplier:
                self.model.out_multiplier[
                    contact_type
                ][nodes] *= factor

    def _order_trace(
        self,
        *,
        t: int,
        cases: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[int, int]:
        cases = np.unique(
            np.asarray(cases, dtype=np.int32)
        )
        merged = {
            **self.default_trace_params,
            **dict(params or {}),
        }

        delay = max(
            1,
            int(merged.get("delay_days", 1)),
        )
        merged["delay_days"] = delay
        due_day = int(t + delay)

        eligible = np.asarray(
            [
                int(node)
                for node in cases
                if not self.state
                .is_trace_pending(int(node))
                and not self.state
                .has_been_traced(int(node))
            ],
            dtype=np.int32,
        )

        if eligible.size == 0:
            return 0, 0

        for node in eligible:
            self.state.mark_trace_pending(
                int(node),
                due_day,
            )

        self.surveillance.order_trace(
            t=int(t),
            cases=eligible,
            params=merged,
        )

        return int(eligible.size), 0

    def _order_test(
        self,
        *,
        t: int,
        nodes: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[int, int]:
        nodes = np.unique(
            np.asarray(nodes, dtype=np.int32)
        )
        merged = {
            **self.default_test_params,
            **dict(params or {}),
        }

        delay = max(
            1,
            int(merged.get("delay_days", 1)),
        )
        merged["delay_days"] = delay
        due_day = int(t + delay)

        eligible = np.asarray(
            [
                int(node)
                for node in nodes
                if not self.state
                .is_test_pending(int(node))
                and not self.state
                .known_case[int(node)]
            ],
            dtype=np.int32,
        )

        if eligible.size == 0:
            return 0, 0

        for node in eligible:
            self.state.mark_test_pending(
                int(node),
                due_day,
            )

        self.surveillance.order_test(
            t=int(t),
            nodes=eligible,
            params=merged,
        )

        return int(eligible.size), 0

    # ------------------------------------------------------------------
    # Policy escalation
    # ------------------------------------------------------------------

    def _update_response_phase_from_reports(
        self,
        *,
        day: int,
        reported_count: int,
    ) -> None:
        if reported_count <= 0:
            return

        if (
            self.policy_phase == 0
            and self.policy_spec[
                "activate_on_first_report"
            ]
        ):
            self.policy_phase = 1
            self.phase1_start_day = int(day)

            self.model.record_event(
                day=day,
                event_type=(
                    "response_phase_started"
                ),
                result="phase_1",
                value=1.0,
            )

        if (
            self.policy_spec[
                "immediate_escalation"
            ]
            and self.escalation_order_day
            is None
        ):
            self._order_escalation(
                day=day,
                trigger_name="first_report",
                trigger_value=float(
                    reported_count
                ),
            )

    def _evaluate_escalation_trigger(
        self,
        *,
        day: int,
        current_row: Dict[str, Any],
    ) -> Optional[float]:
        if (
            not self.policy_spec[
                "adaptive_escalation"
            ]
            or self.policy_phase != 1
            or self.escalation_order_day
            is not None
        ):
            return None

        cfg = self.model.config.lhd
        value = self._current_trigger_value(
            current_row=current_row
        )

        if value >= float(
            cfg.escalation_trigger_threshold
        ):
            self._trigger_streak += 1
        else:
            self._trigger_streak = 0

        required_days = max(
            1,
            int(
                cfg.escalation_consecutive_days
            ),
        )

        if self._trigger_streak >= required_days:
            self._order_escalation(
                day=day,
                trigger_name=(
                    cfg.escalation_trigger_metric
                ),
                trigger_value=value,
            )

        return value

    def _current_trigger_value(
        self,
        *,
        current_row: Dict[str, Any],
    ) -> float:
        cfg = self.model.config.lhd
        metric = str(
            cfg.escalation_trigger_metric
        )
        window = max(
            1,
            int(cfg.escalation_window_days),
        )

        if metric == "new_reports":
            return float(
                current_row["cases_reported"]
            )

        if metric == "backlog_to_capacity_ratio":
            return float(
                current_row[
                    "backlog_to_capacity_ratio"
                ]
            )

        if metric == "capacity_utilization":
            capacity = float(
                current_row[
                    "capacity_available"
                ]
            )
            return (
                float(
                    current_row["capacity_used"]
                )
                / capacity
                if capacity > 0
                else 0.0
            )

        if metric == "maximum_queue_age":
            return float(
                current_row["max_queue_age"]
            )

        report_history = np.asarray(
            [
                float(row["cases_reported"])
                for row in [
                    *self._daily_log,
                    current_row,
                ]
            ],
            dtype=float,
        )

        if metric == "rolling_mean_reports":
            return float(
                np.mean(
                    report_history[-window:]
                )
            )

        if metric == "rolling_report_growth":
            if report_history.size < 2 * window:
                return 0.0

            previous = float(
                np.mean(
                    report_history[
                        -2 * window:-window
                    ]
                )
            )
            current = float(
                np.mean(
                    report_history[-window:]
                )
            )

            if current < float(
                cfg.escalation_minimum_reports
            ):
                return 0.0

            if previous <= 0.0:
                return (
                    float("inf")
                    if current > 0.0
                    else 0.0
                )

            return (
                current - previous
            ) / previous

        raise ValueError(
            f"Unknown escalation metric: "
            f"{metric}"
        )

    def _order_escalation(
        self,
        *,
        day: int,
        trigger_name: str,
        trigger_value: float,
    ) -> bool:
        if self.escalation_order_day is not None:
            return False

        cfg = self.model.config.lhd
        delay = max(
            0,
            int(
                cfg.escalation_implementation_delay_days
            ),
        )

        # Orders occur after transmission, so activation cannot precede
        # the next day's pre-transmission step.
        effective_day = max(
            int(day + delay),
            int(day + 1),
        )

        self.escalation_order_day = int(day)
        self.escalation_effective_day = (
            effective_day
        )
        self.escalation_trigger_name = str(
            trigger_name
        )
        self.escalation_trigger_value = float(
            trigger_value
        )

        self._scheduled_escalations_by_day[
            effective_day
        ].append(
            {
                "trigger_name": trigger_name,
                "trigger_value": float(
                    trigger_value
                ),
                "profile": (
                    cfg.escalation_intervention_profile
                ),
            }
        )

        self.model.record_event(
            day=day,
            event_type="escalation_ordered",
            result=trigger_name,
            value=float(trigger_value),
            metadata={
                "effective_day": effective_day,
                "profile": (
                    cfg.escalation_intervention_profile
                ),
            },
        )

        return True

    def _activate_scheduled_escalations(
        self,
        day: int,
    ) -> None:
        scheduled = (
            self._scheduled_escalations_by_day
            .pop(int(day), [])
        )

        for record in scheduled:
            factors = (
                self._build_escalation_contact_factors()
            )

            for contact_type, factor in (
                factors.items()
            ):
                self.model.in_multiplier[
                    contact_type
                ] *= factor
                self.model.out_multiplier[
                    contact_type
                ] *= factor

            duration = int(
                self.model.config.lhd
                .escalation_duration_days
            )
            expires_at = None

            if duration > 0:
                expires_at = int(
                    day + duration
                )
                self._schedule_token(
                    GlobalMultiplierToken(
                        expires_at=expires_at,
                        contact_factors=dict(
                            factors
                        ),
                        action=(
                            "community_mitigation"
                        ),
                    )
                )

            self.escalation_end_day = expires_at
            self.policy_phase = 2

            if (
                self.model.config.lhd
                .cancel_tracing_backlog_on_escalation
            ):
                self._tasks_cancelled_before_step += (
                    self._cancel_tracing_backlog(
                        day
                    )
                )

            self.model.record_event(
                day=day,
                event_type="escalation_started",
                result=record["trigger_name"],
                value=record["trigger_value"],
                metadata={
                    "profile": record["profile"],
                    "contact_factors": factors,
                    "expires_at": expires_at,
                },
            )

    def _build_escalation_contact_factors(
        self,
    ) -> Dict[str, float]:
        cfg = self.model.config.lhd
        profile = str(
            cfg.escalation_intervention_profile
        )

        profile_reductions = {
            "combined_emergency": {
                "hh":
                    cfg.emergency_household_reduction,
                "sch":
                    cfg.emergency_school_reduction,
                "wp":
                    cfg.emergency_workplace_reduction,
                "cas":
                    cfg.emergency_casual_reduction,
                "gq":
                    cfg.emergency_group_quarter_reduction,
            },
            "stay_home": {
                "hh":
                    cfg.emergency_household_reduction,
                "sch":
                    cfg.emergency_school_reduction,
                "wp":
                    cfg.emergency_workplace_reduction,
                "cas":
                    cfg.emergency_casual_reduction,
                "gq":
                    cfg.emergency_group_quarter_reduction,
            },
            "school_closure": {
                "sch":
                    cfg.emergency_school_reduction,
            },
            "workplace_mitigation": {
                "wp":
                    cfg.emergency_workplace_reduction,
            },
            "gathering_restrictions": {
                "cas":
                    cfg.emergency_casual_reduction,
            },
        }

        if profile not in profile_reductions:
            raise ValueError(
                f"Unknown escalation profile: "
                f"{profile}"
            )

        factors = {}

        for contact_type, reduction in (
            profile_reductions[
                profile
            ].items()
        ):
            if (
                contact_type
                not in self.model.contact_types
            ):
                continue

            reduction = float(
                np.clip(
                    reduction,
                    0.0,
                    1.0,
                )
            )

            if reduction > 0.0:
                factors[contact_type] = max(
                    self.min_factor,
                    1.0 - reduction,
                )

        return factors

    def _cancel_tracing_backlog(
        self,
        day: int,
    ) -> int:
        trace_mode = str(
            self.model.config.lhd
            .phase2_trace_mode
        )

        if trace_mode == "continue":
            return 0

        cancelled = 0
        cancellable_types = {
            "contact_interview",
            "contact_notification",
            "testing_referral",
        }

        for task in list(
            self.state.task_queue.pending_tasks()
        ):
            if (
                task.task_type
                not in cancellable_types
            ):
                continue

            if (
                trace_mode == "targeted"
                and task.task_type
                == "contact_interview"
            ):
                continue

            self.state.task_queue.mark_cancelled(
                task.task_id,
                day,
                "policy_escalation",
            )
            cancelled += 1

            self._record_task_event(
                day=day,
                event_type="task_cancelled",
                task=task,
                metadata={
                    "reason":
                        "policy_escalation",
                },
            )

        return cancelled

    def tracing_enabled_for_new_case(
        self,
        node: int,
    ) -> bool:
        del node  # Reserved for future case-specific prioritization.

        if not self.policy_spec["trace_cases"]:
            return False

        if self.policy_phase < 2:
            return True

        return (
            self.model.config.lhd
            .phase2_trace_mode
            in {"continue", "targeted"}
        )

    # ------------------------------------------------------------------
    # Perfect-information upper bound
    # ------------------------------------------------------------------

    def _apply_perfect_information_response(
        self,
        day: int,
    ) -> None:
        """
        Isolate every true exposed or infectious person before transmission.

        This deliberately violates the ordinary LHD information boundary and
        serves only as an upper-bound benchmark.
        """
        active_nodes = np.where(
            (self.model.state == 1)
            | (self.model.state == 2)
        )[0].astype(
            np.int32,
            copy=False,
        )

        if active_nodes.size == 0:
            return

        new_targets = active_nodes[
            self.state.isolated_until[
                active_nodes
            ] <= day
        ]

        if new_targets.size == 0:
            return

        self._apply_isolation(
            t=day,
            nodes=new_targets,
            params={
                "reduction": float(
                    self.model.config.lhd
                    .perfect_information_reduction
                ),
                "duration": int(
                    self.model.config.lhd
                    .perfect_information_isolation_duration_days
                ),
                "contact_types": list(
                    self.model.contact_types
                ),
            },
        )

        for node in new_targets:
            self.model.record_event(
                day=day,
                event_type=(
                    "perfect_information_"
                    "identification"
                ),
                node_id=int(node),
                disease_stage=(
                    "exposed"
                    if self.model.state[
                        int(node)
                    ] == 1
                    else "infectious"
                ),
                metadata={
                    "truth_access": True,
                    "full_graph_access": True,
                    "unlimited_capacity": True,
                },
            )

    # ------------------------------------------------------------------
    # Token expiration
    # ------------------------------------------------------------------

    def process_expirations(
        self,
        day: int,
    ) -> int:
        day = int(day)
        due_tokens = (
            self._expiry_tokens_by_day.pop(
                day,
                [],
            )
        )

        for token in due_tokens:
            token.revert(self.model)

            if token.action == "community_mitigation":
                self.model.record_event(
                    day=day,
                    event_type="escalation_ended",
                )
                continue

            nodes = np.asarray(
                token.nodes,
                dtype=np.int32,
            )

            if token.action == "isolate":
                expired = nodes[
                    self.state.isolated_until[
                        nodes
                    ] <= day
                ]
                self.state.isolated_until[
                    expired
                ] = -1

                for node in expired:
                    self.model.record_event(
                        day=day,
                        event_type="isolation_ended",
                        node_id=int(node),
                    )

            elif (
                token.action
                == "automated_notification"
            ):
                expired = nodes[
                    self.state.quarantined_until[
                        nodes
                    ] <= day
                ]
                self.state.quarantined_until[
                    expired
                ] = -1

                for node in expired:
                    self.model.record_event(
                        day=day,
                        event_type=(
                            "automated_notification_"
                            "reduction_ended"
                        ),
                        node_id=int(node),
                    )

        return len(due_tokens)

    def _schedule_token(
        self,
        token,
    ) -> None:
        self._expiry_tokens_by_day[
            int(token.expires_at)
        ].append(token)

    # ------------------------------------------------------------------
    # Daily output
    # ------------------------------------------------------------------

    def _build_daily_row(
        self,
        *,
        day: int,
        batch: Dict[str, np.ndarray],
        schedule,
        pending_tasks_start: int,
        pending_units_start: float,
        new_report_tasks: int,
        new_contact_tasks: int,
        child_tasks_created: int,
        completed_tasks: List[LHDTask],
        cancelled_tasks: List[LHDTask],
        expired_tasks: List[LHDTask],
        work_units_used: float,
        tokens_scheduled: int,
    ) -> Dict[str, Any]:
        pending_tasks_end = (
            self.state.task_queue.pending_tasks()
        )
        pending_units_end = (
            self.state.task_queue.pending_work_units()
        )
        queue_ages = (
            self.state.task_queue.queue_ages(
                day
            )
        )

        completed_counts = Counter(
            task.task_type
            for task in completed_tasks
        )
        pending_counts = (
            self.state.task_queue
            .pending_count_by_type()
        )
        pending_units_by_type = (
            self.state.task_queue
            .pending_units_by_type()
        )

        capacity_available = float(
            schedule.capacity_available
        )
        capacity_committed = float(
            schedule.capacity_committed
        )
        capacity_used = float(
            work_units_used
        )

        direct_notifications = int(
            completed_counts.get(
                "contact_notification",
                0,
            )
        )
        automated_notifications = int(
            completed_counts.get(
                "automated_notification",
                0,
            )
        )

        row = {
            "run_number": int(
                self.model.replicate_ind
            ),
            "t": int(day),
            "policy_name": self.policy_name,
            "new_detections": self._array_size(
                batch,
                "detected_cases",
            ),
            "cases_reported": self._array_size(
                batch,
                "reported_cases",
            ),
            "known_cases_total": (
                len(self.state.known_case_list)
            ),
            "cases_awaiting_investigation":
                int(
                    pending_counts.get(
                        "case_investigation",
                        0,
                    )
                ),
            "cases_investigated": int(
                completed_counts.get(
                    "case_investigation",
                    0,
                )
            ),
            "contacts_elicited": self._array_size(
                batch,
                "trace_tgt",
            ),
            "contacts_awaiting_notification":
                int(
                    pending_counts.get(
                        "contact_notification",
                        0,
                    )
                    + pending_counts.get(
                        "automated_notification",
                        0,
                    )
                ),
            "contacts_notified": (
                direct_notifications
                + automated_notifications
            ),
            "direct_contacts_notified":
                direct_notifications,
            "automated_contacts_notified":
                automated_notifications,
            "tests_ordered": int(
                completed_counts.get(
                    "testing_referral",
                    0,
                )
            ),
            "tests_completed": self._array_size(
                batch,
                "tests_completed",
            ),
            "positive_tests": int(
                np.count_nonzero(
                    np.asarray(
                        batch.get(
                            "test_results_positive",
                            np.empty(
                                0,
                                dtype=bool,
                            ),
                        ),
                        dtype=bool,
                    )
                )
            ),
            "isolation_orders": int(
                completed_counts.get(
                    "isolation_support",
                    0,
                )
            ),
            "active_isolations": int(
                np.count_nonzero(
                    self.state.isolated_until
                    > day
                )
            ),
            "active_automated_reductions":
                int(
                    np.count_nonzero(
                        self.state
                        .quarantined_until
                        > day
                    )
                ),
            "new_edges_today": int(
                self.state.new_edges_today
            ),
            "known_edges_total": int(
                len(self.state.known_edges)
            ),
            "pending_tests": int(
                len(self.state.pending_tests)
            ),
            "pending_traces": int(
                len(self.state.pending_traces)
            ),
            "capacity_available":
                capacity_available,
            "capacity_committed":
                capacity_committed,
            "capacity_used": capacity_used,
            "unused_capacity": max(
                0.0,
                capacity_available
                - capacity_used,
            ),
            "pending_tasks_start": int(
                pending_tasks_start
            ),
            "pending_tasks_end": int(
                len(pending_tasks_end)
            ),
            "pending_work_units_start":
                float(
                    pending_units_start
                ),
            "pending_work_units_end":
                float(pending_units_end),
            "backlog_to_capacity_ratio": (
                float(pending_units_end)
                / capacity_available
                if capacity_available > 0
                else (
                    np.inf
                    if pending_units_end > 0
                    else 0.0
                )
            ),
            "new_report_tasks": int(
                new_report_tasks
            ),
            "new_contact_tasks": int(
                new_contact_tasks
            ),
            "child_tasks_created": int(
                child_tasks_created
            ),
            "tasks_selected": int(
                len(schedule.selected)
            ),
            "tasks_completed": int(
                len(completed_tasks)
            ),
            "tasks_cancelled": int(
                len(cancelled_tasks)
                + self._tasks_cancelled_before_step
            ),
            "tasks_expired": int(
                len(expired_tasks)
            ),
            "mean_queue_age": (
                float(np.mean(queue_ages))
                if queue_ages
                else 0.0
            ),
            "median_queue_age": (
                float(np.median(queue_ages))
                if queue_ages
                else 0.0
            ),
            "max_queue_age": (
                int(max(queue_ages))
                if queue_ages
                else 0
            ),
            "tokens_scheduled": int(
                tokens_scheduled
            ),
            "tokens_expired": int(
                self._tokens_expired_today
            ),
        }

        task_types = sorted(
            set(self.task_costs)
            | set(pending_counts)
            | set(completed_counts)
        )

        for task_type in task_types:
            row[
                f"pending_{task_type}"
            ] = int(
                pending_counts.get(
                    task_type,
                    0,
                )
            )
            row[
                f"pending_units_{task_type}"
            ] = float(
                pending_units_by_type.get(
                    task_type,
                    0.0,
                )
            )
            row[
                f"completed_{task_type}"
            ] = int(
                completed_counts.get(
                    task_type,
                    0,
                )
            )

        return row

    @staticmethod
    def _array_size(
        self,
        mapping: Dict[str, Any],
        key: str,
    ) -> int:
        return int(
            np.asarray(
                mapping.get(key, np.empty(0))
            ).size
        )

    # ------------------------------------------------------------------
    # Export and reset
    # ------------------------------------------------------------------

    def lhd_daily_log_to_df(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self._daily_log
        )

    def tasks_to_df(
        self,
    ) -> pd.DataFrame:
        return (
            self.state.task_queue
            .to_dataframe()
        )

    def reset_for_run(self) -> None:
        self._expiry_tokens_by_day.clear()
        self._scheduled_escalations_by_day.clear()
        self._daily_log.clear()

        self.surveillance.reset_for_run(
            seed=self.surveillance.seed,
            is_vax=self.model.is_vaccinated,
        )
        self.state.reset_for_run(
            run_number=int(
                self.model.replicate_ind
            )
        )

        self.policy_phase = (
            1
            if self.policy_spec[
                "perfect_information"
            ]
            else 0
        )
        self.phase1_start_day = (
            0
            if self.policy_spec[
                "perfect_information"
            ]
            else None
        )
        self.escalation_order_day = None
        self.escalation_effective_day = None
        self.escalation_end_day = None
        self.escalation_trigger_name = None
        self.escalation_trigger_value = None
        self._trigger_streak = 0
        self._tokens_expired_today = 0
        self._tasks_cancelled_before_step = 0