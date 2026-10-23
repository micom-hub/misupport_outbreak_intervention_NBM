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
from scripts.lhd.tokens import MultiplierToken


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
            capacity
            if capacity is not None
            else cfg.lhd_daily_capacity
        )

        self.policy_name = str(
            policy_name or cfg.policy_name
        )

        # Transitional Phase 2 workflow profiles. Phase 3 will replace the
        # policy catalog completely.
        self.workflow_profile = (
            self._workflow_profile_for_policy(
                self.policy_name
            )
        )

        # Surveillance
        self.p_detect_symptomatic_daily = float(
            cfg.p_detect_symptomatic_daily
        )
        self.p_detect_asymptomatic_daily = float(
            cfg.p_detect_asymptomatic_daily
        )
        self.p_detect_imported_daily = float(
            cfg.p_detect_imported_daily
        )
        self.imported_detection_start_day = int(
            cfg.imported_detection_start_day
        )
        self.passive_report_delay_days = int(
            cfg.passive_report_delay_days
        )

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

        self.default_iso_reduction = float(
            cfg.lhd_default_int_reduction
        )
        self.default_iso_duration = int(
            cfg.lhd_default_int_duration
        )

        # Deliberately excludes household and group-quarter contacts.
        self.default_iso_contact_types = [
            "cas",
            "sch",
            "wp",
        ]
        self.min_factor = 1e-6

        # Work-unit profiles
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
            "community_mitigation": float(
                cfg.community_mitigation_cost
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
        }

        self.surveillance = SurveillanceModel(
            neighbor_map=self.model.neighbor_map,
            ct_to_id=self.model.ct_to_id,
            seed=self.surv_seed,
            N=self.model.N,
            ages=self.model.ages,
            is_vax=self.model.is_vaccinated,
            p_detect_symptomatic_daily=(
                self.p_detect_symptomatic_daily
            ),
            p_detect_asymptomatic_daily=(
                self.p_detect_asymptomatic_daily
            ),
            p_detect_imported_daily=(
                self.p_detect_imported_daily
            ),
            imported_detection_start_day=(
                self.imported_detection_start_day
            ),
            passive_report_delay_days=(
                self.passive_report_delay_days
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

        self._expiry_tokens_by_day = defaultdict(list)

        self._results_rows: List[Dict[str, Any]] = []
        self._action_log: List[Dict[str, Any]] = []
        self._lhd_daily_log: List[Dict[str, Any]] = []

    @staticmethod
    def _workflow_profile_for_policy(
        policy_name: str,
    ) -> Dict[str, bool]:
        profiles = {
            "observe_only": {
                "investigate_cases": False,
                "isolate_cases": False,
                "trace_cases": False,
                "notify_contacts": False,
                "test_contacts": False,
            },
            "isolate_only": {
                "investigate_cases": True,
                "isolate_cases": True,
                "trace_cases": False,
                "notify_contacts": False,
                "test_contacts": False,
            },
            "trace_only": {
                "investigate_cases": True,
                "isolate_cases": False,
                "trace_cases": True,
                "notify_contacts": True,
                "test_contacts": False,
            },
            "trace_then_isolate": {
                "investigate_cases": True,
                "isolate_cases": True,
                "trace_cases": True,
                "notify_contacts": True,
                "test_contacts": False,
            },
            "trace_and_test": {
                "investigate_cases": True,
                "isolate_cases": True,
                "trace_cases": True,
                "notify_contacts": True,
                "test_contacts": True,
            },
            "network_crawl": {
                "investigate_cases": True,
                "isolate_cases": False,
                "trace_cases": True,
                "notify_contacts": True,
                "test_contacts": False,
            },
            "network_crawl_isolate": {
                "investigate_cases": True,
                "isolate_cases": True,
                "trace_cases": True,
                "notify_contacts": True,
                "test_contacts": False,
            },
            "clique_crawl_containment": {
                "investigate_cases": True,
                "isolate_cases": True,
                "trace_cases": True,
                "notify_contacts": True,
                "test_contacts": True,
            },
        }

        if policy_name not in profiles:
            raise ValueError(
                f"No Phase 2 workflow profile exists for "
                f"policy '{policy_name}'"
            )

        return dict(profiles[policy_name])

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
                f"No work-unit cost configured for "
                f"task type '{task_type}'"
            )

        if task_type not in self.task_priorities:
            raise ValueError(
                f"No priority configured for "
                f"task type '{task_type}'"
            )

        created_day = int(created_day)

        # Child tasks are deliberately unavailable until the next daily
        # scheduling cycle.
        if earliest_start_day is None:
            earliest_start_day = created_day + 1

        deadline_offset = self.task_deadline_days.get(
            task_type
        )
        deadline_day = (
            None
            if deadline_offset is None
            else created_day + int(deadline_offset)
        )

        task_params = dict(params or {})

        if task_type == "isolation_support":
            task_params = {
                "reduction":
                    self.default_iso_reduction,
                "duration":
                    self.default_iso_duration,
                "contact_types":
                    self.default_iso_contact_types,
                **task_params,
            }

        elif task_type == "contact_interview":
            task_params = {
                **self.default_trace_params,
                **task_params,
            }

        elif task_type == "testing_referral":
            task_params = {
                **self.default_test_params,
                **task_params,
            }

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

        task_or_existing, created = (
            self.state.task_queue.add(task)
        )

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
                    "earliest_start_day":
                        task.earliest_start_day,
                    "deadline_day":
                        task.deadline_day,
                    "reason": reason,
                },
            )

        return task_or_existing, created

    def _ingest_new_reports(
        self,
        *,
        day: int,
    ) -> int:
        if not self.workflow_profile[
            "investigate_cases"
        ]:
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

            if not self.workflow_profile[
                "notify_contacts"
            ]:
                continue

            _, was_created = self.enqueue_task(
                task_type="contact_notification",
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
        expired = self.state.task_queue.expire_overdue(
            day
        )

        for task in expired:
            self.model.record_event(
                day=day,
                event_type="task_expired",
                node_id=int(task.target),
                source_node_id=task.source_case,
                task_id=task.task_id,
                metadata={
                    "task_type": task.task_type,
                    "reason":
                        task.expiration_reason,
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

    def step(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
        scheduled_surv_actions: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, np.ndarray]:
        day = int(t)

        tokens_expired = self.process_expirations(
            day
        )

        pending_tasks_start = len(
            self.state.task_queue.pending_tasks()
        )
        pending_units_start = (
            self.state.task_queue.pending_work_units()
        )

        batch = self.observe(
            t=day,
            epi_state=epi_state,
            scheduled_actions=(
                scheduled_surv_actions
            ),
        )

        self.state.process_batch(batch)

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

        eligible_tasks = (
            self.state.task_queue.eligible_tasks(
                day
            )
        )
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
                work_units_used += float(
                    result.work_units_used
                )
                tokens_scheduled += int(
                    result.tokens_scheduled
                )
                child_tasks_created += int(
                    result.child_tasks_created
                )

                self.model.record_event(
                    day=day,
                    event_type="task_completed",
                    node_id=int(task.target),
                    source_node_id=task.source_case,
                    task_id=task.task_id,
                    metadata={
                        "task_type":
                            task.task_type,
                        "work_units_used":
                            result.work_units_used,
                        "queue_age":
                            task.age(day),
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
                        "task_type":
                            task.task_type,
                        "reason":
                            result.message,
                    },
                )

        self._log_day(
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
    ) -> None:
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

        pending_tasks_end = (
            self.state.task_queue.pending_tasks()
        )
        pending_units_end = (
            self.state.task_queue.pending_work_units()
        )
        queue_ages = (
            self.state.task_queue.queue_ages(day)
        )

        if queue_ages:
            mean_age = float(np.mean(queue_ages))
            median_age = float(
                np.median(queue_ages)
            )
            max_age = int(max(queue_ages))
        else:
            mean_age = 0.0
            median_age = 0.0
            max_age = 0

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

        active_isolations = int(
            np.count_nonzero(
                self.state.isolated_until > day
            )
        )

        capacity_available = float(
            schedule.capacity_available
        )
        capacity_committed = float(
            schedule.capacity_committed
        )
        capacity_used = float(work_units_used)
        unused_capacity = max(
            0.0,
            capacity_available - capacity_used,
        )

        backlog_ratio = (
            float(pending_units_end)
            / capacity_available
            if capacity_available > 0
            else (
                np.inf
                if pending_units_end > 0
                else 0.0
            )
        )

        row = {
            "run_number": int(
                self.model.replicate_ind
            ),
            "t": int(day),
            "policy_name": self.policy_name,

            "cases_reported": int(
                reported.size
            ),
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
            "contacts_elicited": int(
                trace_targets.size
            ),
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
            "tests_completed": int(
                tests_completed.size
            ),
            "positive_tests": int(
                np.count_nonzero(
                    positive_tests
                )
            ),
            "isolation_orders": int(
                completed_counts.get(
                    "isolation_support",
                    0,
                )
            ),
            "active_isolations":
                active_isolations,

            "capacity_available":
                capacity_available,
            "capacity_committed":
                capacity_committed,
            "capacity_used":
                capacity_used,
            "unused_capacity":
                unused_capacity,

            "pending_tasks_start": int(
                pending_tasks_start
            ),
            "pending_tasks_end": int(
                len(pending_tasks_end)
            ),
            "pending_work_units_start": float(
                pending_units_start
            ),
            "pending_work_units_end": float(
                pending_units_end
            ),
            "backlog_to_capacity_ratio": float(
                backlog_ratio
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
            ),
            "tasks_expired": int(
                len(expired_tasks)
            ),

            "mean_queue_age": mean_age,
            "median_queue_age": median_age,
            "max_queue_age": max_age,

            "tokens_scheduled": int(
                tokens_scheduled
            ),
            "tokens_expired": int(
                tokens_expired
            ),

            # Phase 3 will replace this with
            # explicit phase management.
            "policy_phase": (
                0
                if self.policy_name
                == "observe_only"
                else 1
            ),
            "escalation_trigger": None,
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

        self._results_rows.append(row)
        self._lhd_daily_log.append(row)

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

        reduction = float(
            np.clip(reduction, 0.0, 1.0)
        )
        factor = max(
            self.min_factor,
            1.0 - reduction,
        )

        # Do not multiply an already-active isolation again.
        eligible = nodes[
            self.state.isolated_until[nodes] <= t
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

        expires_at = (
            int(t + duration)
            if duration > 0
            else int(t + 1)
        )

        self.state.isolated_until[eligible] = (
            expires_at
        )

        tokens_added = 0

        if duration > 0:
            token = MultiplierToken(
                expires_at=expires_at,
                nodes=eligible.copy(),
                contact_types=tuple(
                    contact_types
                ),
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
                    "contact_types": list(
                        contact_types
                    ),
                },
            )

        return int(eligible.size), tokens_added

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
                if not self.state.is_trace_pending(
                    int(node)
                )
                and not self.state.has_been_traced(
                    int(node)
                )
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
                if not self.state.is_test_pending(
                    int(node)
                )
                and not self.state.known_case[
                    int(node)
                ]
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

            if token.action == "isolate":
                nodes = np.asarray(
                    token.nodes,
                    dtype=np.int32,
                )
                expired = nodes[
                    self.state.isolated_until[nodes]
                    <= t
                ]
                self.state.isolated_until[
                    expired
                ] = -1

                for node in expired:
                    self.model.record_event(
                        day=t,
                        event_type="isolation_ended",
                        node_id=int(node),
                    )

        return int(len(due))

    def _schedule_token(
        self,
        token: MultiplierToken,
    ) -> None:
        self._expiry_tokens_by_day[
            int(token.expires_at)
        ].append(token)

    def lhd_daily_log_to_df(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self._lhd_daily_log
        )

    def results_to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._results_rows
        )

    def action_log_to_df(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self._action_log
        )

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
        self.state.reset_for_run(
            run_number=int(
                self.model.replicate_ind
            )
        )