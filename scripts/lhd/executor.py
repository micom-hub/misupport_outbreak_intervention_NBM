from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from scripts.lhd.tasks import LHDTask


@dataclass
class TaskExecutionResult:
    success: bool
    work_units_used: float
    message: str = ""
    child_tasks_created: int = 0
    tokens_scheduled: int = 0


class TaskExecutor:
    """
    Execute one selected operational task.

    Child tasks are added to the persistent queue, but are not eligible for
    selection until a later daily scheduling cycle.
    """

    def execute(
        self,
        *,
        lhd,
        day: int,
        task: LHDTask,
    ) -> TaskExecutionResult:
        task_type = task.task_type
        target = int(task.target)

        if task_type == "case_investigation":
            return self._case_investigation(
                lhd=lhd,
                day=day,
                task=task,
                target=target,
            )

        if task_type == "isolation_support":
            return self._isolation_support(
                lhd=lhd,
                day=day,
                task=task,
                target=target,
            )

        if task_type == "contact_interview":
            return self._contact_interview(
                lhd=lhd,
                day=day,
                task=task,
                target=target,
            )

        if task_type == "contact_notification":
            return self._contact_notification(
                lhd=lhd,
                day=day,
                task=task,
                target=target,
            )

        if task_type == "testing_referral":
            return self._testing_referral(
                lhd=lhd,
                day=day,
                task=task,
                target=target,
            )

        return TaskExecutionResult(
            success=False,
            work_units_used=0.0,
            message=f"Unknown task type: {task_type}",
        )

    def _case_investigation(
        self,
        *,
        lhd,
        day: int,
        task: LHDTask,
        target: int,
    ) -> TaskExecutionResult:
        children = 0

        lhd.record_task_semantic_event(
            day=day,
            event_type="case_investigation_completed",
            task=task,
            node_id=target,
        )

        if lhd.workflow_profile["isolate_cases"]:
            _, created = lhd.enqueue_task(
                task_type="isolation_support",
                target=target,
                source_case=target,
                created_day=day,
                reason="reported case investigated",
                source_algorithm="workflow",
            )
            children += int(created)

        if lhd.workflow_profile["trace_cases"]:
            _, created = lhd.enqueue_task(
                task_type="contact_interview",
                target=target,
                source_case=target,
                created_day=day,
                reason="reported case investigated",
                source_algorithm="workflow",
            )
            children += int(created)

        return TaskExecutionResult(
            success=True,
            work_units_used=float(task.work_cost),
            child_tasks_created=children,
        )

    def _isolation_support(
        self,
        *,
        lhd,
        day: int,
        task: LHDTask,
        target: int,
    ) -> TaskExecutionResult:
        applied, tokens = lhd._apply_isolation(
            t=day,
            nodes=np.asarray(
                [target],
                dtype=np.int32,
            ),
            params=task.params,
        )

        if applied <= 0:
            return TaskExecutionResult(
                success=False,
                work_units_used=0.0,
                message="Isolation was not applied",
            )

        lhd.record_task_semantic_event(
            day=day,
            event_type="isolation_support_completed",
            task=task,
            node_id=target,
        )

        return TaskExecutionResult(
            success=True,
            work_units_used=float(task.work_cost),
            tokens_scheduled=int(tokens),
        )

    def _contact_interview(
        self,
        *,
        lhd,
        day: int,
        task: LHDTask,
        target: int,
    ) -> TaskExecutionResult:
        ordered, _ = lhd._order_trace(
            t=day,
            cases=np.asarray(
                [target],
                dtype=np.int32,
            ),
            params=task.params,
        )

        if ordered <= 0:
            return TaskExecutionResult(
                success=False,
                work_units_used=0.0,
                message=(
                    "Trace was already pending or previously completed"
                ),
            )

        lhd.record_task_semantic_event(
            day=day,
            event_type="contact_interview_completed",
            task=task,
            node_id=target,
        )

        return TaskExecutionResult(
            success=True,
            work_units_used=float(task.work_cost),
        )

    def _contact_notification(
        self,
        *,
        lhd,
        day: int,
        task: LHDTask,
        target: int,
    ) -> TaskExecutionResult:
        lhd.record_task_semantic_event(
            day=day,
            event_type="contact_notified",
            task=task,
            node_id=target,
            source_node_id=task.source_case,
        )

        children = 0

        if lhd.workflow_profile["test_contacts"]:
            _, created = lhd.enqueue_task(
                task_type="testing_referral",
                target=target,
                source_case=task.source_case,
                created_day=day,
                reason="notified traced contact",
                source_algorithm="workflow",
            )
            children += int(created)

        return TaskExecutionResult(
            success=True,
            work_units_used=float(task.work_cost),
            child_tasks_created=children,
        )

    def _testing_referral(
        self,
        *,
        lhd,
        day: int,
        task: LHDTask,
        target: int,
    ) -> TaskExecutionResult:
        ordered, _ = lhd._order_test(
            t=day,
            nodes=np.asarray(
                [target],
                dtype=np.int32,
            ),
            params=task.params,
        )

        if ordered <= 0:
            return TaskExecutionResult(
                success=False,
                work_units_used=0.0,
                message=(
                    "Test was already pending or target is a known case"
                ),
            )

        lhd.record_task_semantic_event(
            day=day,
            event_type="testing_referral_completed",
            task=task,
            node_id=target,
            source_node_id=task.source_case,
        )

        return TaskExecutionResult(
            success=True,
            work_units_used=float(task.work_cost),
        )