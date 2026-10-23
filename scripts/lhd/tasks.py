from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ACTIVE_TASK_STATUSES = {
    "pending",
    "selected",
    "in_progress",
}


@dataclass
class LHDTask:
    task_id: str
    task_type: str
    target_kind: str
    target: int

    created_day: int
    earliest_start_day: int
    deadline_day: Optional[int]

    base_priority: float
    work_cost: float

    source_case: Optional[int] = None
    source_algorithm: str = ""
    reason: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    status: str = "pending"
    selected_day: Optional[int] = None
    started_day: Optional[int] = None
    completed_day: Optional[int] = None
    expired_day: Optional[int] = None
    cancelled_day: Optional[int] = None

    expiration_reason: Optional[str] = None
    cancellation_reason: Optional[str] = None

    def dedup_key(self) -> Tuple[Any, ...]:
        """
        Return a task-specific key used to prevent duplicate active work.
        """
        if self.task_type in {
            "case_investigation",
            "isolation_support",
            "contact_interview",
        }:
            return (
                self.task_type,
                self.target_kind,
                int(self.target),
            )

        if self.task_type in {
            "contact_notification",
            "automated_notification",
        }:
            return (
                self.task_type,
                self.target_kind,
                int(self.target),
                (
                    None
                    if self.source_case is None
                    else int(self.source_case)
                ),
            )

        if self.task_type == "testing_referral":
            return (
                self.task_type,
                self.target_kind,
                int(self.target),
            )

        return (
            self.task_type,
            self.target_kind,
            int(self.target),
            (
                None
                if self.source_case is None
                else int(self.source_case)
            ),
        )

    def is_active(self) -> bool:
        return self.status in ACTIVE_TASK_STATUSES

    def is_eligible(self, day: int) -> bool:
        return (
            self.status == "pending"
            and int(day) >= int(self.earliest_start_day)
            and (
                self.deadline_day is None
                or int(day) <= int(self.deadline_day)
            )
        )

    def age(self, day: int) -> int:
        return max(0, int(day) - int(self.created_day))


class TaskQueue:
    """
    Persistent task registry and active-work queue.

    Completed, expired, and cancelled tasks remain in the registry for
    auditing, while the active-key index prevents duplicate pending work.
    """

    def __init__(self, run_number: int = 0):
        self.run_number = int(run_number)
        self._counter = 0
        self._tasks: Dict[str, LHDTask] = {}
        self._active_key_to_id: Dict[Tuple[Any, ...], str] = {}

    def reset(self, run_number: int) -> None:
        self.run_number = int(run_number)
        self._counter = 0
        self._tasks.clear()
        self._active_key_to_id.clear()

    def next_task_id(self) -> str:
        task_id = (
            f"run-{self.run_number:04d}-"
            f"task-{self._counter:08d}"
        )
        self._counter += 1
        return task_id

    def add(self, task: LHDTask) -> Tuple[LHDTask, bool]:
        """
        Add a task unless equivalent active work already exists.

        Returns:
            (task_or_existing_task, was_created)
        """
        key = task.dedup_key()
        existing_id = self._active_key_to_id.get(key)

        if existing_id is not None:
            existing = self._tasks[existing_id]

            if existing.is_active():
                # Allow later proposals to increase urgency.
                existing.base_priority = max(
                    float(existing.base_priority),
                    float(task.base_priority),
                )

                if task.deadline_day is not None:
                    if existing.deadline_day is None:
                        existing.deadline_day = int(
                            task.deadline_day
                        )
                    else:
                        existing.deadline_day = min(
                            int(existing.deadline_day),
                            int(task.deadline_day),
                        )

                return existing, False

            self._active_key_to_id.pop(key, None)

        self._tasks[task.task_id] = task
        self._active_key_to_id[key] = task.task_id
        return task, True

    def get(self, task_id: str) -> LHDTask:
        return self._tasks[task_id]

    def all_tasks(self) -> List[LHDTask]:
        return list(self._tasks.values())

    def pending_tasks(self) -> List[LHDTask]:
        return [
            task
            for task in self._tasks.values()
            if task.status == "pending"
        ]

    def eligible_tasks(self, day: int) -> List[LHDTask]:
        return [
            task
            for task in self._tasks.values()
            if task.is_eligible(day)
        ]

    def mark_selected(
        self,
        task_id: str,
        day: int,
    ) -> None:
        task = self.get(task_id)
        task.status = "selected"
        task.selected_day = int(day)

    def mark_started(
        self,
        task_id: str,
        day: int,
    ) -> None:
        task = self.get(task_id)
        task.status = "in_progress"
        task.started_day = int(day)

    def mark_completed(
        self,
        task_id: str,
        day: int,
    ) -> None:
        task = self.get(task_id)
        task.status = "completed"
        task.completed_day = int(day)
        self._release_active_key(task)

    def mark_expired(
        self,
        task_id: str,
        day: int,
        reason: str,
    ) -> None:
        task = self.get(task_id)
        task.status = "expired"
        task.expired_day = int(day)
        task.expiration_reason = str(reason)
        self._release_active_key(task)

    def mark_cancelled(
        self,
        task_id: str,
        day: int,
        reason: str,
    ) -> None:
        task = self.get(task_id)
        task.status = "cancelled"
        task.cancelled_day = int(day)
        task.cancellation_reason = str(reason)
        self._release_active_key(task)

    def expire_overdue(self, day: int) -> List[LHDTask]:
        expired = []

        for task in self.pending_tasks():
            if (
                task.deadline_day is not None
                and int(day) > int(task.deadline_day)
            ):
                self.mark_expired(
                    task.task_id,
                    day,
                    "deadline_passed",
                )
                expired.append(task)

        return expired

    def pending_work_units(self) -> float:
        return float(
            sum(
                float(task.work_cost)
                for task in self.pending_tasks()
            )
        )

    def pending_count_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}

        for task in self.pending_tasks():
            counts[task.task_type] = (
                counts.get(task.task_type, 0) + 1
            )

        return counts

    def pending_units_by_type(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}

        for task in self.pending_tasks():
            totals[task.task_type] = (
                totals.get(task.task_type, 0.0)
                + float(task.work_cost)
            )

        return totals

    def queue_ages(self, day: int) -> List[int]:
        return [
            task.age(day)
            for task in self.pending_tasks()
        ]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []

        for task in self.all_tasks():
            rows.append(
                {
                    "run_number": self.run_number,
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "target_kind": task.target_kind,
                    "target": int(task.target),
                    "source_case": task.source_case,
                    "created_day": int(task.created_day),
                    "earliest_start_day": int(
                        task.earliest_start_day
                    ),
                    "deadline_day": task.deadline_day,
                    "base_priority": float(
                        task.base_priority
                    ),
                    "work_cost": float(task.work_cost),
                    "status": task.status,
                    "selected_day": task.selected_day,
                    "started_day": task.started_day,
                    "completed_day": task.completed_day,
                    "expired_day": task.expired_day,
                    "cancelled_day": task.cancelled_day,
                    "expiration_reason":
                        task.expiration_reason,
                    "cancellation_reason":
                        task.cancellation_reason,
                    "source_algorithm":
                        task.source_algorithm,
                    "reason": task.reason,
                    "params": task.params,
                }
            )

        return pd.DataFrame(rows)

    def _release_active_key(
        self,
        task: LHDTask,
    ) -> None:
        key = task.dedup_key()

        if self._active_key_to_id.get(key) == task.task_id:
            self._active_key_to_id.pop(key, None)