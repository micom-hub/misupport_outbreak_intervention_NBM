from __future__ import annotations

import numpy as np
import pytest


def run_policy(
    toy_model_factory,
    policy_name,
    *,
    transmission_probability=0.0,
    lhd_overrides=None,
    simulation_duration=8,
):
    return toy_model_factory.model(
        policy_name=policy_name,
        transmission_probability=(
            transmission_probability
        ),
        lhd_overrides=lhd_overrides,
        sim_overrides={
            "simulation_duration":
                simulation_duration,
        },
        simulate=True,
    )


def test_observe_only_has_no_tasks_or_interventions(
    toy_model_factory,
):
    model = run_policy(
        toy_model_factory,
        "observe_only",
    )

    tasks = model.lhd_tasks_to_df()
    daily = model.lhd_daily_to_df()
    events = model.event_log_to_df()

    assert tasks.empty
    assert daily["capacity_used"].sum() == 0
    assert not (
        events["event_type"]
        == "isolation_started"
    ).any()


def test_case_isolation_only_has_no_tracing_tasks(
    toy_model_factory,
):
    model = run_policy(
        toy_model_factory,
        "case_isolation_only",
    )

    task_types = set(
        model.lhd_tasks_to_df()[
            "task_type"
        ]
    )

    assert task_types <= {
        "case_investigation",
        "isolation_support",
    }


def test_trace_centered_response_creates_full_workflow(
    toy_model_factory,
):
    model = run_policy(
        toy_model_factory,
        "trace_centered_response",
    )

    task_types = set(
        model.lhd_tasks_to_df()[
            "task_type"
        ]
    )

    assert {
        "case_investigation",
        "isolation_support",
        "contact_interview",
        "contact_notification",
        "testing_referral",
    } <= task_types


def test_immediate_mitigation_orders_on_first_report(
    toy_model_factory,
):
    model = run_policy(
        toy_model_factory,
        "immediate_scalable_mitigation",
    )

    events = model.event_log_to_df()

    first_report = events.loc[
        events["event_type"]
        == "report_received",
        "day",
    ].min()

    first_order = events.loc[
        events["event_type"]
        == "escalation_ordered",
        "day",
    ].min()

    assert first_order == first_report


def test_adaptive_escalation_requires_trigger(
    toy_model_factory,
):
    no_trigger_model = run_policy(
        toy_model_factory,
        "adaptive_escalation",
        lhd_overrides={
            "escalation_trigger_threshold":
                1000.0,
        },
    )
    trigger_model = run_policy(
        toy_model_factory,
        "adaptive_escalation",
        lhd_overrides={
            "escalation_trigger_threshold": 1.0,
            "escalation_consecutive_days": 1,
        },
    )

    no_trigger_events = (
        no_trigger_model.event_log_to_df()
    )
    trigger_events = (
        trigger_model.event_log_to_df()
    )

    assert not (
        no_trigger_events["event_type"]
        == "escalation_started"
    ).any()
    assert (
        trigger_events["event_type"]
        == "escalation_started"
    ).any()


def test_escalation_starts_after_order_day(
    toy_model_factory,
):
    model = run_policy(
        toy_model_factory,
        "adaptive_escalation",
        lhd_overrides={
            "escalation_trigger_threshold": 1.0,
            "escalation_consecutive_days": 1,
            "escalation_implementation_delay_days":
                0,
        },
    )

    events = model.event_log_to_df()

    order_day = events.loc[
        events["event_type"]
        == "escalation_ordered",
        "day",
    ].min()
    start_day = events.loc[
        events["event_type"]
        == "escalation_started",
        "day",
    ].min()

    assert start_day > order_day


def test_phase2_stop_prevents_new_tracing(
    toy_model_factory,
):
    model = toy_model_factory.model(
        policy_name="adaptive_escalation",
        simulate=False,
    )
    model._initialize_replicate(0)

    model.lhd.policy_phase = 2

    assert (
        model.lhd.tracing_enabled_for_new_case(
            int(model.imported_ids[0])
        )
        is False
    )


def test_phase2_targeted_restricts_interview_to_household(
    toy_model_factory,
):
    model = toy_model_factory.model(
        policy_name="adaptive_escalation",
        lhd_overrides={
            "phase2_trace_mode": "targeted",
        },
        simulate=False,
    )
    model._initialize_replicate(0)
    model.lhd.policy_phase = 2

    task, created = model.lhd.enqueue_task(
        task_type="contact_interview",
        target=int(model.imported_ids[0]),
        source_case=int(model.imported_ids[0]),
        created_day=1,
    )

    assert created is True
    assert task.params["contact_types"] == ["hh"]


def test_escalation_cancels_tracing_backlog(
    toy_model_factory,
):
    model = toy_model_factory.model(
        policy_name="adaptive_escalation",
        simulate=False,
    )
    model._initialize_replicate(0)

    task, _ = model.lhd.enqueue_task(
        task_type="contact_interview",
        target=int(model.imported_ids[0]),
        source_case=int(model.imported_ids[0]),
        created_day=1,
    )

    cancelled = (
        model.lhd._cancel_tracing_backlog(
            day=2
        )
    )

    assert cancelled == 1
    assert task.status == "cancelled"
    assert (
        task.cancellation_reason
        == "policy_escalation"
    )


def test_perfect_information_uses_no_capacity(
    toy_model_factory,
):
    model = run_policy(
        toy_model_factory,
        "perfect_information",
        transmission_probability=1.0,
    )

    assert model.lhd_tasks_to_df().empty
    assert (
        model.lhd_daily_to_df()[
            "capacity_used"
        ].sum()
        == 0
    )


def test_perfect_information_acts_before_transmission(
    toy_model_factory,
):
    model = run_policy(
        toy_model_factory,
        "perfect_information",
        transmission_probability=1.0,
        simulation_duration=2,
    )

    events = model.event_log_to_df()

    identifications = events[
        events["event_type"]
        == "perfect_information_identification"
    ]

    assert not identifications.empty
    assert identifications["day"].min() == 1


def test_perfect_information_prevents_local_spread(
    toy_model_factory,
):
    model = run_policy(
        toy_model_factory,
        "perfect_information",
        transmission_probability=1.0,
        lhd_overrides={
            "perfect_information_reduction": 1.0,
        },
    )

    generations = (
        model.all_infection_generation[0]
    )

    assert np.count_nonzero(generations > 0) == 0