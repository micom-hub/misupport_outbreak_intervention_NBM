from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.lhd.executor import TaskExecutor
from scripts.lhd.planner import WorkScheduler
from scripts.lhd.state import LHDState
from scripts.lhd.surveillance import SurveillanceModel
from scripts.lhd.tasks import LHDTask
from scripts.lhd.policy_catalog import get_policy_spec
from scripts.lhd.tokens import (
    GlobalMultiplierToken,
    MultiplierToken,
)


class LocalHealthDepartment:
    """
    Capacity-constrained operational LHD workflow.

    Daily process:
      1. Expire interventions.
      2. Process surveillance and receive reports/results.
      3. Update partial LHD knowledge.
      4. Convert new reports and elicited contacts into tasks.
      5. Expire overdue tasks.
      6. Schedule eligible tasks under daily work-unit capacity.
      7. Execute selected tasks.
      8. Preserve unselected tasks in the backlog.
      9. Record daily operational summaries.
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
            capacity if capacity is not None else cfg.lhd_daily_capacity
        )

        self.policy_name = str(policy_name or cfg.policy_name)

        # Transitional Phase 2 workflow profiles. Phase 3 will replace the
        # policy catalog completely.
        self.policy_spec = get_policy_spec(self.policy_name)

        # Policy Phases defined
        # 0 - Routine surveillance, no outbreak detected
        # 1 - case investigation, tracing responses, outbreak investigation
        # 2 - Public health emergency, escalated response, intractable tracing

        # perfect info policy starts in investigation phase
        self.policy_phase = 1 if self.policy_spec["perfect_information"] else 0

        # Define triggers for shift between phases
        self.phase1_start_day: Optional[int] = (
            0 if self.policy_spec["perfect_information"] else None
        )
        self.escalation_order_day: Optional[int] = None
        self.escalation_effective_day: Optional[int] = None
        self.escalation_end_day: Optional[int] = None
        self.escalation_trigger_name: Optional[str] = None
        self.escalation_trigger_value: Optional[float] = None

        self._trigger_streak = 0
        self._scheduled_escalations_by_day = defaultdict(list)
        self._active_global_interventions = []

        # Surveillance
        self.p_detect_symptomatic_daily = float(cfg.p_detect_symptomatic_daily)
        self.p_detect_asymptomatic_daily = float(cfg.p_detect_asymptomatic_daily)
        self.p_detect_imported_daily = float(cfg.p_detect_imported_daily)
        self.imported_detection_start_day = int(cfg.imported_detection_start_day)
        self.passive_report_delay_days = int(cfg.passive_report_delay_days)

        self.default_trace_params = {
            "delay_days": int(cfg.trace_delay_days),
            "recall_prob": float(cfg.trace_recall_prob),
            "max_per_case": int(cfg.trace_max_per_case),
            "contact_types": [
                "hh",
                "sch",
                "wp",
            ],
        }

        self.default_test_params = {
            "delay_days": int(cfg.test_turnaround_days),
            "sensitivity_exposed": float(cfg.test_sensitivity_exposed),
            "sensitivity_infectious": float(cfg.test_sensitivity_infectious),
            "specificity": float(cfg.test_specificity),
            "report_delay_days": int(cfg.test_report_delay_days),
        }

        self.default_iso_reduction = float(cfg.lhd_default_int_reduction)
        self.default_iso_duration = int(cfg.lhd_default_int_duration)

        # Deliberately excludes household and group-quarter contacts.
        self.default_iso_contact_types = [
            "cas",
            "sch",
            "wp",
        ]
        self.min_factor = 1e-6

        # Work-unit profiles
        self.task_costs = {
            "case_investigation": float(cfg.case_investigation_cost),
            "isolation_support": float(cfg.isolation_support_cost),
            "contact_interview": float(cfg.contact_interview_cost),
            "contact_notification": float(cfg.contact_notification_cost),
            "testing_referral": float(cfg.testing_referral_cost),
            "automated_notification": float(cfg.automated_notification_cost),
            "community_mitigation": float(cfg.community_mitigation_cost),
        }

        self.task_priorities = {
            "case_investigation": float(cfg.case_investigation_priority),
            "isolation_support": float(cfg.isolation_support_priority),
            "contact_interview": float(cfg.contact_interview_priority),
            "contact_notification": float(cfg.contact_notification_priority),
            "automated_notification": float(cfg.automated_notification_priority),
            "testing_referral": float(cfg.testing_referral_priority),
        }

        self.task_deadline_days = {
            "case_investigation": int(cfg.case_investigation_deadline_days),
            "isolation_support": int(cfg.isolation_support_deadline_days),
            "contact_interview": int(cfg.contact_interview_deadline_days),
            "contact_notification": int(cfg.contact_notification_deadline_days),
            "automated_notification": int(cfg.automated_notification_deadline_days),
            "testing_referral": int(cfg.testing_referral_deadline_days),
        }

        self.task_priorities["automated_notification"] = float(
            cfg.automated_notification_priority
        )

        self.surveillance = SurveillanceModel(
            neighbor_map=self.model.neighbor_map,
            ct_to_id=self.model.ct_to_id,
            seed=self.surv_seed,
            N=self.model.N,
            ages=self.model.ages,
            is_vax=self.model.is_vaccinated,
            p_detect_symptomatic_daily=(self.p_detect_symptomatic_daily),
            p_detect_asymptomatic_daily=(self.p_detect_asymptomatic_daily),
            p_detect_imported_daily=(self.p_detect_imported_daily),
            imported_detection_start_day=(self.imported_detection_start_day),
            passive_report_delay_days=(self.passive_report_delay_days),
        )
        self.surveillance.attach_model(self.model)

        self.state = LHDState(N=self.model.N)

        self.scheduler = WorkScheduler(
            aging_priority_per_day=float(cfg.task_aging_priority_per_day)
        )
        self.executor = TaskExecutor()

        self._expiry_tokens_by_day = defaultdict(list)

        self._results_rows: List[Dict[str, Any]] = []
        self._action_log: List[Dict[str, Any]] = []
        self._lhd_daily_log: List[Dict[str, Any]] = []

    def observe(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
        scheduled_actions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        return self.surveillance.step(
            t=int(t),
            epi_state=epi_state,
            scheduled_actions=scheduled_actions,
        )

    def pre_epi_step(
        self,
        day: int,
    ) -> None:
        """
        Apply intervention changes that occur before today's transmission step, including processes expiring, scheduling escalations, or the perfect info response
        """
        day = int(day)

        self.process_expirations(day)
        self._activate_scheduled_escalations(day)

        if self.policy_spec["perfect_information"]:
            self._apply_perfect_information_response(day)

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
                f"No work-unit cost configured for task type '{task_type}'"
            )

        if task_type not in self.task_priorities:
            raise ValueError(f"No priority configured for task type '{task_type}'")

        created_day = int(created_day)

        # Child tasks are deliberately unavailable until the next daily
        # scheduling cycle.
        if earliest_start_day is None:
            earliest_start_day = created_day + 1

        deadline_offset = self.task_deadline_days.get(task_type)
        deadline_day = (
            None if deadline_offset is None else created_day + int(deadline_offset)
        )

        task_params = dict(params or {})

        if task_type == "isolation_support":
            task_params = {
                "reduction": self.default_iso_reduction,
                "duration": self.default_iso_duration,
                "contact_types": self.default_iso_contact_types,
                **task_params,
            }

        elif task_type == "contact_interview":
            task_params = {
                **self.default_trace_params,
                **task_params,
            }
            if (
            self.policy_phase == 2
            and self.model.config.lhd.phase2_trace_mode
            == "targeted"):
              task_params["contact_types"] = ["hh"]

        elif task_type == "testing_referral":
            task_params = {
                **self.default_test_params,
                **task_params,
            }

        elif task_type == "automated_notification":
            task_params = {
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
                **task_params,
            }

        task = LHDTask(
            task_id=(self.state.task_queue.next_task_id()),
            task_type=task_type,
            target_kind="node",
            target=int(target),
            source_case=(None if source_case is None else int(source_case)),
            created_day=created_day,
            earliest_start_day=int(earliest_start_day),
            deadline_day=deadline_day,
            base_priority=float(self.task_priorities[task_type]),
            work_cost=float(self.task_costs[task_type]),
            source_algorithm=source_algorithm,
            reason=reason,
            params=task_params,
        )

        task_or_existing, created = self.state.task_queue.add(task)

        if created:
            self.model.record_event(
                day=created_day,
                event_type="task_created",
                node_id=int(target),
                source_node_id=source_case,
                task_id=task.task_id,
                metadata={
                    "task_type": task_type,
                    "work_cost": task.work_cost,
                    "priority": task.base_priority,
                    "earliest_start_day": task.earliest_start_day,
                    "deadline_day": task.deadline_day,
                    "reason": reason,
                },
            )

        return task_or_existing, created

    def _ingest_new_reports(
        self,
        *,
        day: int,
    ) -> int:

        if self.policy_spec["perfect_information"]:
          return 0

        if not self.policy_spec["investigate_cases"]:
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
        trace_source = np.asarray(
            batch.get(
                "trace_src",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )
        trace_target = np.asarray(
            batch.get(
                "trace_tgt",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )
        trace_type = np.asarray(
            batch.get(
                "trace_ct",
                np.empty(0, np.int16),
            ),
            dtype=np.int16,
        )

        created = 0
        n = min(
            trace_source.size,
            trace_target.size,
            trace_type.size,
        )

        for index in range(n):
            source = int(trace_source[index])
            target = int(trace_target[index])
            type_id = int(trace_type[index])
            contact_type = self.model.id_to_ct.get(
                type_id,
                str(type_id),
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
                and self.policy_spec["automated_notification_in_phase2"]):
                task_type = "automated_notification"

            elif self.policy_spec["notify_contacts"]:
                task_type = "contact_notification"

            else:
                continue

            _, was_created = self.enqueue_task(
                task_type=task_type,
                target=target,
                source_case=source,
                created_day=day,
                earliest_start_day=day,
                reason=(
                    f"contact elicited from case {source}"
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
        expired = self.state.task_queue.expire_overdue(day)

        for task in expired:
            self.model.record_event(
                day=day,
                event_type="task_expired",
                node_id=int(task.target),
                source_node_id=task.source_case,
                task_id=task.task_id,
                metadata={
                    "task_type": task.task_type,
                    "reason": task.expiration_reason,
                    "age_days": task.age(day),
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
            node_id=(int(task.target) if node_id is None else int(node_id)),
            source_node_id=(
                task.source_case if source_node_id is None else source_node_id
            ),
            task_id=task.task_id,
            metadata={
                "task_type": task.task_type,
                "work_cost": task.work_cost,
                "queue_age": task.age(day),
            },
        )

    def step(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
        scheduled_surv_actions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        day = int(t)

        tokens_expired = 0  # expiration occurs at the start of the day now

        pending_tasks_start = len(self.state.task_queue.pending_tasks())
        pending_units_start = self.state.task_queue.pending_work_units()

        batch = self.observe(
            t=day,
            epi_state=epi_state,
            scheduled_actions=(scheduled_surv_actions),
        )

        self.state.process_batch(batch)

        reported_count = int(np.asarray(batch.get(
            "reported_cases",
            np.empty(0, np.int32),
        )).size)
        # determine whether we escalate based on case reports
        self._update_response_phase_from_reports(
            day=day,
            reported_count=reported_count,
        )

        new_report_tasks = self._ingest_new_reports(day=day)
        new_contact_tasks = self._ingest_elicited_contacts(
            day=day,
            batch=batch,
        )

        expired_tasks = self._expire_tasks(day=day)

        eligible_tasks = self.state.task_queue.eligible_tasks(day)
        schedule = self.scheduler.select(
            eligible_tasks,
            capacity=self.daily_capacity,
            day=day,
        )

        work_units_used = 0.0
        completed_tasks: List[LHDTask] = []
        cancelled_tasks: List[LHDTask] = []
        tokens_scheduled = 0
        child_tasks_created = 0

        for task in schedule.selected:
            self.state.task_queue.mark_selected(
                task.task_id,
                day,
            )
            self.model.record_event(
                day=day,
                event_type="task_selected",
                node_id=int(task.target),
                source_node_id=task.source_case,
                task_id=task.task_id,
                metadata={
                    "task_type": task.task_type,
                    "work_cost": task.work_cost,
                    "queue_age": task.age(day),
                },
            )

            self.state.task_queue.mark_started(
                task.task_id,
                day,
            )
            self.model.record_event(
                day=day,
                event_type="task_started",
                node_id=int(task.target),
                source_node_id=task.source_case,
                task_id=task.task_id,
                metadata={
                    "task_type": task.task_type,
                },
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
                work_units_used += float(result.work_units_used)
                tokens_scheduled += int(result.tokens_scheduled)
                child_tasks_created += int(result.child_tasks_created)

                self.model.record_event(
                    day=day,
                    event_type="task_completed",
                    node_id=int(task.target),
                    source_node_id=task.source_case,
                    task_id=task.task_id,
                    metadata={
                        "task_type": task.task_type,
                        "work_units_used": result.work_units_used,
                        "queue_age": task.age(day),
                    },
                )

            else:
                self.state.task_queue.mark_cancelled(
                    task.task_id,
                    day,
                    result.message,
                )
                cancelled_tasks.append(task)

                self.model.record_event(
                    day=day,
                    event_type="task_cancelled",
                    node_id=int(task.target),
                    source_node_id=task.source_case,
                    task_id=task.task_id,
                    metadata={
                        "task_type": task.task_type,
                        "reason": result.message,
                    },
                )
        # get the log for the day
        row = self._log_day(
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
            tokens_expired=tokens_expired,
        )
        # extract values from the log
        trigger_value = self._evaluate_escalation_trigger(
            day=day,
            current_row=row,
        )

        row["policy_phase"] = int(
            self.policy_phase
        )
        row["escalation_trigger_metric"] = (
            self.model.config.lhd
            .escalation_trigger_metric
            if self.policy_spec[
                "adaptive_escalation"
            ]
            else None
        )
        row["escalation_trigger_value"] = (
            trigger_value
        )
        row["escalation_triggered_today"] = bool(
            self.escalation_order_day == day
        )
        row["escalation_order_day"] = (
            self.escalation_order_day
        )
        row["escalation_effective_day"] = (
            self.escalation_effective_day
        )
        row["escalation_active"] = bool(
            self.policy_phase == 2
        )
        row["escalation_profile"] = (
            self.model.config.lhd
            .escalation_intervention_profile
            if self.policy_phase == 2
            or self.escalation_order_day is not None
            else None
        )

        self._results_rows.append(row)
        self._lhd_daily_log.append(row)

        return batch

    def _log_day(
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
        tokens_expired: int,
    ) -> Dict[str, Any]:
        reported = np.asarray(
            batch.get(
                "reported_cases",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )
        tests_completed = np.asarray(
            batch.get(
                "tests_completed",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )
        positive_tests = np.asarray(
            batch.get(
                "test_results_positive",
                np.empty(0, dtype=bool),
            ),
            dtype=bool,
        )
        trace_targets = np.asarray(
            batch.get(
                "trace_tgt",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )

        pending_tasks_end = self.state.task_queue.pending_tasks()
        pending_units_end = self.state.task_queue.pending_work_units()
        queue_ages = self.state.task_queue.queue_ages(day)

        if queue_ages:
            mean_age = float(np.mean(queue_ages))
            median_age = float(np.median(queue_ages))
            max_age = int(max(queue_ages))
        else:
            mean_age = 0.0
            median_age = 0.0
            max_age = 0

        completed_counts = Counter(task.task_type for task in completed_tasks)
        pending_counts = self.state.task_queue.pending_count_by_type()
        pending_units_by_type = self.state.task_queue.pending_units_by_type()

        active_isolations = int(np.count_nonzero(self.state.isolated_until > day))

        capacity_available = float(schedule.capacity_available)
        capacity_committed = float(schedule.capacity_committed)
        capacity_used = float(work_units_used)
        unused_capacity = max(
            0.0,
            capacity_available - capacity_used,
        )

        backlog_ratio = (
            float(pending_units_end) / capacity_available
            if capacity_available > 0
            else (np.inf if pending_units_end > 0 else 0.0)
        )

        row = {
            "run_number": int(self.model.replicate_ind),
            "t": int(day),
            "policy_name": self.policy_name,
            "cases_reported": int(reported.size),
            "cases_awaiting_investigation": int(
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
            "contacts_elicited": int(trace_targets.size),
            "contacts_awaiting_notification": int(
                pending_counts.get(
                    "contact_notification",
                    0,
                )
            ),
            "contacts_notified": int(
                completed_counts.get(
                    "contact_notification",
                    0,
                )
            ),
            "tests_ordered": int(
                completed_counts.get(
                    "testing_referral",
                    0,
                )
            ),
            "tests_completed": int(tests_completed.size),
            "positive_tests": int(np.count_nonzero(positive_tests)),
            "isolation_orders": int(
                completed_counts.get(
                    "isolation_support",
                    0,
                )
            ),
            "active_isolations": active_isolations,
            "capacity_available": capacity_available,
            "capacity_committed": capacity_committed,
            "capacity_used": capacity_used,
            "unused_capacity": unused_capacity,
            "pending_tasks_start": int(pending_tasks_start),
            "pending_tasks_end": int(len(pending_tasks_end)),
            "pending_work_units_start": float(pending_units_start),
            "pending_work_units_end": float(pending_units_end),
            "backlog_to_capacity_ratio": float(backlog_ratio),
            "new_report_tasks": int(new_report_tasks),
            "new_contact_tasks": int(new_contact_tasks),
            "child_tasks_created": int(child_tasks_created),
            "tasks_selected": int(len(schedule.selected)),
            "tasks_completed": int(len(completed_tasks)),
            "tasks_cancelled": int(len(cancelled_tasks)),
            "tasks_expired": int(len(expired_tasks)),
            "mean_queue_age": mean_age,
            "median_queue_age": median_age,
            "max_queue_age": max_age,
            "tokens_scheduled": int(tokens_scheduled),
            "tokens_expired": int(tokens_expired),
            "policy_phase": (0 if self.policy_name == "observe_only" else 1),
            "escalation_trigger": None,
            "new_detections": int(np.asarray(batch.get("detected_cases",
            np.empty(0, np.int32),)).size),
            "known_cases_total": (len(self.state.known_case_list)),
            "known_edges_total": (len(self.state.known_edges)),
            "new_edges_today": int(self.state.new_edges_today),
            "pending_tests": (len(self.state.pending_tests)),
            "pending_traces": (len(self.state.pending_traces)),
            "active_automated_reductions": int(np.count_nonzero(
                    self.state.quarantined_until > day)),
            
        }

        task_types = sorted(
            set(self.task_costs) | set(pending_counts) | set(completed_counts)
        )

        for task_type in task_types:
            row[f"pending_{task_type}"] = int(
                pending_counts.get(
                    task_type,
                    0,
                )
            )
            row[f"pending_units_{task_type}"] = float(
                pending_units_by_type.get(
                    task_type,
                    0.0,
                )
            )
            row[f"completed_{task_type}"] = int(
                completed_counts.get(
                    task_type,
                    0,
                )
            )

        return row



    def _apply_isolation(
        self,
        *,
        t: int,
        nodes: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[int, int]:
        nodes = np.unique(np.asarray(nodes, dtype=np.int32))

        if nodes.size == 0:
            return 0, 0

        reduction = float(
            params.get(
                "reduction",
                self.default_iso_reduction,
            )
        )
        duration = int(
            params.get(
                "duration",
                self.default_iso_duration,
            )
        )
        contact_types = params.get(
            "contact_types",
            self.default_iso_contact_types,
        )

        reduction = float(np.clip(reduction, 0.0, 1.0))
        factor = max(
            self.min_factor,
            1.0 - reduction,
        )

        # Do not multiply an already-active isolation again.
        eligible = nodes[self.state.isolated_until[nodes] <= t]

        if eligible.size == 0:
            return 0, 0

        for contact_type in contact_types:
            if contact_type in self.model.in_multiplier:
                self.model.in_multiplier[contact_type][eligible] *= factor

            if contact_type in self.model.out_multiplier:
                self.model.out_multiplier[contact_type][eligible] *= factor

        expires_at = int(t + duration) if duration > 0 else int(t + 1)

        self.state.isolated_until[eligible] = expires_at

        tokens_added = 0

        if duration > 0:
            token = MultiplierToken(
                expires_at=expires_at,
                nodes=eligible.copy(),
                contact_types=tuple(contact_types),
                in_factor=factor,
                out_factor=factor,
                action="isolate",
            )
            self._schedule_token(token)
            tokens_added = 1

        for node in eligible:
            self._action_log.append(
                {
                    "t": int(t),
                    "node": int(node),
                    "action": "isolate",
                    "duration": int(duration),
                    "reduction": float(reduction),
                }
            )
            self.model.record_event(
                day=t,
                event_type="isolation_started",
                node_id=int(node),
                metadata={
                    "duration": duration,
                    "expires_at": expires_at,
                    "reduction": reduction,
                    "contact_types": list(contact_types),
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
        contact_types = params.get(
            "contact_types",
            ["cas", "sch", "wp"],
        )

        factor = max(
            self.min_factor,
            1.0 - reduction,
        )

        eligible = nodes[
            self.state.quarantined_until[nodes]
            <= t
        ]

        if eligible.size == 0:
            return 0, 0

        for contact_type in contact_types:
            if contact_type in self.model.in_multiplier:
                self.model.in_multiplier[
                    contact_type
                ][eligible] *= factor
            if contact_type in self.model.out_multiplier:
                self.model.out_multiplier[
                    contact_type
                ][eligible] *= factor

        expires_at = int(t + duration)

        self.state.quarantined_until[eligible] = (
            expires_at
        )

        token = MultiplierToken(
            expires_at=expires_at,
            nodes=eligible.copy(),
            contact_types=tuple(contact_types),
            in_factor=factor,
            out_factor=factor,
            action="automated_notification",
        )
        self._schedule_token(token)

        for node in eligible:
            self.model.record_event(
                day=t,
                event_type=(
                    "automated_notification_reduction_started"
                ),
                node_id=int(node),
                metadata={
                    "reduction": reduction,
                    "expires_at": expires_at,
                    "contact_types":
                        list(contact_types),
                },
            )

        return int(eligible.size), 1

    def _build_escalation_contact_factors(self) -> Dict[str, float]:
        # Extract contact factor adjustments based on escalation profile

        cfg = self.model.config.lhd
        profile = str(cfg.escalation_intervention_profile)
        reductions: Dict[str, float] = {}

        # these two are the same for now
        if profile == "combined_emergency" or profile == "stay_home":
            reductions = {
                "hh": cfg.emergency_household_reduction,
                "sch": cfg.emergency_school_reduction,
                "wp": cfg.emergency_workplace_reduction,
                "cas": cfg.emergency_casual_reduction,
                "gq": cfg.emergency_group_quarter_reduction,
            }

        elif profile == "school_closure":
            reductions = {
                "sch": cfg.emergency_school_reduction,
            }

        elif profile == "workplace_mitigation":
            reductions = {
                "wp": cfg.emergency_workplace_reduction,
            }

        elif profile == "gathering_restrictions":
            reductions = {
                "cas": cfg.emergency_casual_reduction,
            }

        else:
            raise ValueError(f"Unknown escalation profile: {profile}")

        factors = {}

        for contact_type, reduction in reductions.items():
            if contact_type not in self.model.contact_types:
                continue

            reduction = float(np.clip(reduction, 0.0, 1.0))

            if reduction <= 0.0:
                continue

            factors[contact_type] = max(
                self.min_factor,
                1.0 - reduction,
            )

        return factors

    def _order_trace(
        self,
        *,
        t: int,
        cases: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[int, int]:
        cases = np.unique(np.asarray(cases, dtype=np.int32))

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
                if not self.state.is_trace_pending(int(node))
                and not self.state.has_been_traced(int(node))
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

        for node in eligible:
            self._action_log.append(
                {
                    "t": int(t),
                    "node": int(node),
                    "action": "trace",
                    "params": dict(merged),
                }
            )

        return int(eligible.size), 0

    def _order_test(
        self,
        *,
        t: int,
        nodes: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[int, int]:
        nodes = np.unique(np.asarray(nodes, dtype=np.int32))

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
                if not self.state.is_test_pending(int(node))
                and not self.state.known_case[int(node)]
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

        for node in eligible:
            self._action_log.append(
                {
                    "t": int(t),
                    "node": int(node),
                    "action": "test",
                    "params": dict(merged),
                }
            )

        return int(eligible.size), 0


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
        effective_day = int(day + delay)

        # If ordered after today's transmission, even zero delay can only affect
        # the following day's transmission.
        effective_day = max(
            effective_day,
            int(day + 1),
        )

        self.escalation_order_day = int(day)
        self.escalation_effective_day = effective_day
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
                "profile":
                    cfg.escalation_intervention_profile,
            },
        )

        return True


    def _activate_scheduled_escalations(
        self,
        day: int,
    ) -> None:
        scheduled = self._scheduled_escalations_by_day.pop(
            int(day),
            [],
        )

        for record in scheduled:
            factors = (
                self._build_escalation_contact_factors()
            )

            for contact_type, factor in factors.items():
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

            if duration > 0:
                expires_at = int(day + duration)

                token = GlobalMultiplierToken(
                    expires_at=expires_at,
                    contact_factors=dict(factors),
                    action="community_mitigation",
                )
                self._schedule_token(token)
                self.escalation_end_day = expires_at
            else:
                expires_at = None
                self.escalation_end_day = None

            self.policy_phase = 2

            if (
                self.model.config.lhd
                .cancel_tracing_backlog_on_escalation
            ):
                self._cancel_tracing_backlog(day)

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

    def _cancel_tracing_backlog(
        self,
        day: int,
    ) -> None:
        trace_mode = str(
            self.model.config.lhd.phase2_trace_mode
        )

        if trace_mode == "continue":
            return

        cancellable_types = {
            "contact_interview",
            "contact_notification",
            "testing_referral",
        }

        for task in list(
            self.state.task_queue.pending_tasks()
        ):
            if task.task_type not in cancellable_types:
                continue

            if (
                trace_mode == "targeted"
                and task.task_type
                == "contact_interview"
            ):
                # Targeted tracing behavior is enforced when new tasks are
                # created; existing interviews are retained.
                continue

            self.state.task_queue.mark_cancelled(
                task.task_id,
                day,
                "policy_escalation",
            )

            self.model.record_event(
                day=day,
                event_type="task_cancelled",
                node_id=int(task.target),
                source_node_id=task.source_case,
                task_id=task.task_id,
                metadata={
                    "task_type": task.task_type,
                    "reason": "policy_escalation",
                },
            )

    def _current_trigger_value(
        self,
        *,
        day: int,
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

        prior_rows = list(self._lhd_daily_log)
        rows = prior_rows + [current_row]

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
                current_row["capacity_available"]
            )
            used = float(
                current_row["capacity_used"]
            )
            return (
                used / capacity
                if capacity > 0
                else 0.0
            )

        if metric == "maximum_queue_age":
            return float(
                current_row["max_queue_age"]
            )

        reports = np.asarray(
            [
                float(row["cases_reported"])
                for row in rows
            ],
            dtype=float,
        )

        if metric == "rolling_mean_reports":
            return float(
                np.mean(reports[-window:])
            )

        if metric == "rolling_report_growth":
            if reports.size < 2 * window:
                return 0.0

            previous = float(
                np.mean(
                    reports[-2 * window:-window]
                )
            )
            current = float(
                np.mean(reports[-window:])
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
            f"Unknown trigger metric: {metric}"
        )


    def _evaluate_escalation_trigger(
        self,
        *,
        day: int,
        current_row: Dict[str, Any],
    ) -> Optional[float]:
        if not self.policy_spec[
            "adaptive_escalation"
        ]:
            return None

        if self.policy_phase != 1:
            return None

        if self.escalation_order_day is not None:
            return None

        cfg = self.model.config.lhd
        value = self._current_trigger_value(
            day=day,
            current_row=current_row,
        )
        threshold = float(
            cfg.escalation_trigger_threshold
        )

        if value >= threshold:
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
                event_type="response_phase_started",
                result="phase_1",
                value=1.0,
            )

        if (
            self.policy_spec[
                "immediate_escalation"
            ]
            and self.escalation_order_day is None
        ):
            self._order_escalation(
                day=day,
                trigger_name="first_report",
                trigger_value=float(
                    reported_count
                ),
            )



    def process_expirations(
        self,
        t: int,
    ) -> int:
        t = int(t)
        due = self._expiry_tokens_by_day.pop(
            t,
            [],
        )

        for token in due:
            token.revert(self.model)

            if token.action == "community_mitigation":
              self.model.record_event(
                  day=t,
                  event_type="escalation_ended",
              )

            if token.action == "isolate":
                nodes = np.asarray(
                    token.nodes,
                    dtype=np.int32,
                )
                expired = nodes[self.state.isolated_until[nodes] <= t]
                self.state.isolated_until[expired] = -1

                for node in expired:
                    self.model.record_event(
                        day=t,
                        event_type="isolation_ended",
                        node_id=int(node),
                    )

            elif token.action == "automated_notification":
                nodes = np.asarray(
                    token.nodes,
                    dtype=np.int32,
                )
                expired = nodes[
                    self.state.quarantined_until[nodes]
                    <= t
                ]
                self.state.quarantined_until[
                    expired
                ] = -1

                for node in expired:
                    self.model.record_event(
                        day=t,
                        event_type=(
                            "automated_notification_reduction_ended"
                        ),
                        node_id=int(node),
                    )

        return len(due)

    def tracing_enabled_for_new_case(
        self,
        node: int,
    ) -> bool:
        if not self.policy_spec["trace_cases"]:
            return False

        if self.policy_phase < 2:
            return True

        mode = str(
            self.model.config.lhd.phase2_trace_mode
        )

        if mode == "continue":
            return True

        if mode == "stop":
            return False

        if mode == "targeted":  # noqa: SIM103
            # The case interview is retained, but surveillance tracing itself
            # is restricted to household contacts in enqueue_task().
            return True

        return False

    def _schedule_token(
        self,
        token: MultiplierToken,
    ) -> None:
        self._expiry_tokens_by_day[int(token.expires_at)].append(token)


    def _apply_perfect_information_response(
        self,
        day: int,
    ) -> None:
        """
        Upper bound for an LHD with "perfect" knowledge of the epidemic

        Before transmission each day, this policy observes the true E and I
        states and applies near-complete transmission reduction across every
        available contact type. It has no operational capacity limit.
        """
        active_truth_nodes = np.where(
            (self.model.state == 1)
            | (self.model.state == 2)
        )[0].astype(np.int32, copy=False)

        if active_truth_nodes.size == 0:
            return

        reduction = float(
            np.clip(
                self.model.config.lhd
                .perfect_information_reduction,
                0.0,
                1.0,
            )
        )
        duration = int(
            self.model.config.lhd
            .perfect_information_isolation_duration_days
        )

        newly_targeted = active_truth_nodes[
            self.state.isolated_until[
                active_truth_nodes
            ] <= day
        ]

        if newly_targeted.size == 0:
            return

        self._apply_isolation(
            t=day,
            nodes=newly_targeted,
            params={
                "reduction": reduction,
                "duration": duration,
                "contact_types": list(
                    self.model.contact_types
                ),
            },
        )

        for node in newly_targeted:
            self.model.record_event(
                day=day,
                event_type="perfect_information_identification",
                node_id=int(node),
                disease_stage=(
                    "exposed"
                    if self.model.state[int(node)] == 1
                    else "infectious"
                ),
                metadata={
                    "truth_access": True,
                    "full_graph_access": True,
                    "unlimited_capacity": True,
                },
            )

    def lhd_daily_log_to_df(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(self._lhd_daily_log)

    def results_to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._results_rows)

    def action_log_to_df(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(self._action_log)

    def tasks_to_df(self) -> pd.DataFrame:
        return self.state.task_queue.to_dataframe()

    def reset_for_run(self) -> None:
        self._expiry_tokens_by_day.clear()
        self._results_rows.clear()
        self._action_log.clear()
        self._lhd_daily_log.clear()

        self.surveillance.reset_for_run(
            seed=self.surveillance.seed,
            is_vax=self.model.is_vaccinated,
        )
        self.state.reset_for_run(run_number=int(self.model.replicate_ind))

        self.policy_phase = (
            1
            if self.policy_spec["perfect_information"]
            else 0
        )
        self.phase1_start_day = (
            0
            if self.policy_spec["perfect_information"]
            else None
        )
        self.escalation_order_day = None
        self.escalation_effective_day = None
        self.escalation_end_day = None
        self.escalation_trigger_name = None
        self.escalation_trigger_value = None
        self._trigger_streak = 0
        self._scheduled_escalations_by_day.clear()
        self._active_global_interventions.clear()
