from __future__ import annotations

from scripts.lhd.planner import WorkScheduler
from scripts.lhd.tasks import LHDTask, TaskQueue


def make_task(
    task_id,
    *,
    task_type="case_investigation",
    target=1,
    source_case=None,
    created_day=0,
    earliest_start_day=0,
    deadline_day=10,
    priority=1.0,
    cost=1.0,
):
    return LHDTask(
        task_id=task_id,
        task_type=task_type,
        target_kind="node",
        target=target,
        source_case=source_case,
        created_day=created_day,
        earliest_start_day=earliest_start_day,
        deadline_day=deadline_day,
        base_priority=priority,
        work_cost=cost,
    )


def test_equivalent_active_tasks_are_deduplicated():
    queue = TaskQueue(run_number=0)

    first, first_created = queue.add(
        make_task("task-1")
    )
    second, second_created = queue.add(
        make_task(
            "task-2",
            priority=2.0,
        )
    )

    assert first_created is True
    assert second_created is False
    assert first.task_id == second.task_id
    assert len(queue.all_tasks()) == 1
    assert first.base_priority == 2.0


def test_unselected_tasks_remain_pending():
    queue = TaskQueue(run_number=0)
    tasks = [
        queue.add(
            make_task(
                f"task-{index}",
                target=index,
                cost=2.0,
            )
        )[0]
        for index in range(3)
    ]

    scheduler = WorkScheduler(
        aging_priority_per_day=0.25
    )
    plan = scheduler.select(
        tasks,
        capacity=2.0,
        day=0,
    )

    assert len(plan.selected) == 1
    assert len(queue.pending_tasks()) == 3


def test_selected_cost_never_exceeds_capacity():
    scheduler = WorkScheduler(
        aging_priority_per_day=0.25
    )

    tasks = [
        make_task(
            f"task-{index}",
            target=index,
            cost=2.0,
        )
        for index in range(4)
    ]

    plan = scheduler.select(
        tasks,
        capacity=5.0,
        day=0,
    )

    assert plan.capacity_committed <= 5.0
    assert sum(
        task.work_cost
        for task in plan.selected
    ) <= 5.0


def test_priority_aging_increases_effective_priority():
    scheduler = WorkScheduler(
        aging_priority_per_day=0.5
    )
    task = make_task(
        "task-1",
        created_day=0,
        priority=1.0,
    )

    assert (
        scheduler.effective_priority(
            task,
            day=4,
        )
        == 3.0
    )


def test_overdue_tasks_expire():
    queue = TaskQueue(run_number=0)
    task, _ = queue.add(
        make_task(
            "task-1",
            deadline_day=2,
        )
    )

    expired = queue.expire_overdue(day=3)

    assert expired == [task]
    assert task.status == "expired"
    assert task.expired_day == 3
    assert task.expiration_reason == "deadline_passed"


def test_completed_and_cancelled_tasks_release_key():
    queue = TaskQueue(run_number=0)

    first, _ = queue.add(
        make_task("task-1")
    )
    queue.mark_completed(
        first.task_id,
        day=1,
    )

    replacement, created = queue.add(
        make_task("task-2")
    )

    assert created is True
    assert replacement.task_id == "task-2"

    queue.mark_cancelled(
        replacement.task_id,
        day=2,
        reason="test",
    )

    third, created = queue.add(
        make_task("task-3")
    )

    assert created is True
    assert third.task_id == "task-3"


def test_queue_counts_and_work_units_reconcile():
    queue = TaskQueue(run_number=0)

    queue.add(
        make_task(
            "task-1",
            task_type="case_investigation",
            target=1,
            cost=4.0,
        )
    )
    queue.add(
        make_task(
            "task-2",
            task_type="testing_referral",
            target=2,
            cost=1.0,
        )
    )

    assert queue.pending_work_units() == 5.0
    assert queue.pending_count_by_type() == {
        "case_investigation": 1,
        "testing_referral": 1,
    }
    assert queue.pending_units_by_type() == {
        "case_investigation": 4.0,
        "testing_referral": 1.0,
    }


def test_zero_capacity_selects_nothing():
    scheduler = WorkScheduler()
    tasks = [
        make_task("task-1", cost=1.0)
    ]

    plan = scheduler.select(
        tasks,
        capacity=0.0,
        day=0,
    )

    assert plan.selected == []
    assert plan.capacity_committed == 0.0


def test_task_more_expensive_than_capacity_remains_unselected():
    scheduler = WorkScheduler()
    task = make_task(
        "task-1",
        cost=10.0,
    )

    plan = scheduler.select(
        [task],
        capacity=5.0,
        day=0,
    )

    assert plan.selected == []
    assert task.status == "pending"