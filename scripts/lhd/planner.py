from __future__ import annotations

from dataclasses import dataclass
from typing import List

from scripts.lhd.tasks import LHDTask


@dataclass
class SchedulePlan:
    selected: List[LHDTask]
    capacity_available: float
    capacity_committed: float

    @property
    def unused_capacity(self) -> float:
        return max(
            0.0,
            float(self.capacity_available)
            - float(self.capacity_committed),
        )


class WorkScheduler:
    """
    Daily scheduler for persistent LHD tasks.

    Tasks are ranked by:
      1. effective priority per work unit;
      2. effective priority;
      3. age;
      4. lower work cost;
      5. earlier creation day;
      6. stable task ID.

    Effective priority increases as a task waits.
    """

    name = "work_scheduler"

    def __init__(
        self,
        *,
        aging_priority_per_day: float = 0.25,
    ):
        self.aging_priority_per_day = float(
            aging_priority_per_day
        )

    def effective_priority(
        self,
        task: LHDTask,
        day: int,
    ) -> float:
        return (
            float(task.base_priority)
            + self.aging_priority_per_day
            * float(task.age(day))
        )

    def select(
        self,
        tasks: List[LHDTask],
        capacity: float,
        day: int,
    ) -> SchedulePlan:
        capacity = max(0.0, float(capacity))

        eligible = [
            task
            for task in tasks
            if task.is_eligible(day)
        ]

        def ranking_key(task: LHDTask):
            priority = self.effective_priority(
                task,
                day,
            )
            cost = max(float(task.work_cost), 1e-9)

            return (
                priority / cost,
                priority,
                task.age(day),
                -cost,
                -int(task.created_day),
                task.task_id,
            )

        eligible.sort(
            key=ranking_key,
            reverse=True,
        )

        selected: List[LHDTask] = []
        committed = 0.0

        for task in eligible:
            cost = max(0.0, float(task.work_cost))

            if committed + cost <= capacity + 1e-9:
                selected.append(task)
                committed += cost

        return SchedulePlan(
            selected=selected,
            capacity_available=capacity,
            capacity_committed=committed,
        )