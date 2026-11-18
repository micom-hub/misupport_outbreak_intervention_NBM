from __future__ import annotations

import numpy as np


def run_trace_workflow(
    toy_model_factory,
    *,
    capacity=100.0,
):
    return toy_model_factory.model(
        policy_name="trace_centered_response",
        transmission_probability=0.0,
        lhd_overrides={
            "lhd_daily_capacity": capacity,
            "p_detect_imported_daily": 1.0,
            "p_detect_symptomatic_daily": 1.0,
            "passive_report_delay_days": 0,
            "trace_recall_prob": 1.0,
            "trace_delay_days": 1,
            "test_turnaround_days": 1,
            "test_report_delay_days": 0,
        },
        sim_overrides={
            "simulation_duration": 7,
        },
        simulate=True,
    )


def test_report_creates_case_investigation(
    toy_model_factory,
):
    model = run_trace_workflow(
        toy_model_factory
    )
    tasks = model.lhd_tasks_to_df()

    assert (
        tasks["task_type"]
        == "case_investigation"
    ).any()


def test_case_investigation_creates_child_tasks(
    toy_model_factory,
):
    model = run_trace_workflow(
        toy_model_factory
    )
    tasks = model.lhd_tasks_to_df()

    assert (
        tasks["task_type"]
        == "isolation_support"
    ).any()
    assert (
        tasks["task_type"]
        == "contact_interview"
    ).any()


def test_isolation_applies_and_expires_multipliers(
    toy_model_factory,
):
    model = toy_model_factory.model(
        policy_name="case_isolation_only",
        transmission_probability=0.0,
        simulate=False,
    )
    model._initialize_replicate(0)

    node = int(model.imported_ids[0])

    applied, tokens = model.lhd._apply_isolation(
        t=1,
        nodes=np.asarray([node], dtype=np.int32),
        params={
            "reduction": 0.5,
            "duration": 2,
            "contact_types": ["wp"],
        },
    )

    assert applied == 1
    assert tokens == 1
    assert model.out_multiplier["wp"][node] == 0.5
    assert model.in_multiplier["wp"][node] == 0.5

    model.lhd.process_expirations(day=3)

    assert np.isclose(
        model.out_multiplier["wp"][node],
        1.0,
    )
    assert np.isclose(
        model.in_multiplier["wp"][node],
        1.0,
    )
    assert model.lhd.state.isolated_until[node] == -1


def test_contact_interview_creates_delayed_trace_order(
    toy_model_factory,
):
    model = run_trace_workflow(
        toy_model_factory
    )
    tasks = model.lhd_tasks_to_df()

    interview = tasks.loc[
        tasks["task_type"]
        == "contact_interview"
    ].iloc[0]

    assert interview["status"] == "completed"

    events = model.event_log_to_df()

    elicited = events[
        events["event_type"]
        == "contact_elicited"
    ]

    assert not elicited.empty
    assert (
        elicited["day"].min()
        > interview["completed_day"]
    )


def test_completed_trace_creates_contact_notifications(
    toy_model_factory,
):
    model = run_trace_workflow(
        toy_model_factory
    )
    tasks = model.lhd_tasks_to_df()

    assert (
        tasks["task_type"]
        == "contact_notification"
    ).any()


def test_contact_notification_creates_testing_referral(
    toy_model_factory,
):
    model = run_trace_workflow(
        toy_model_factory
    )
    tasks = model.lhd_tasks_to_df()

    assert (
        tasks["task_type"]
        == "testing_referral"
    ).any()


def test_testing_referral_creates_test_order(
    toy_model_factory,
):
    model = run_trace_workflow(
        toy_model_factory
    )
    events = model.event_log_to_df()

    assert (
        events["event_type"]
        == "test_ordered"
    ).any()


def test_backlog_persists_under_limited_capacity(
    toy_model_factory,
):
    model = run_trace_workflow(
        toy_model_factory,
        capacity=1.0,
    )
    daily = model.lhd_daily_to_df()

    assert (
        daily["pending_tasks_end"] > 0
    ).any()
    assert (
        daily["pending_work_units_end"] > 0
    ).any()


def test_daily_capacity_accounting_reconciles(
    toy_model_factory,
):
    model = run_trace_workflow(
        toy_model_factory,
        capacity=5.0,
    )
    daily = model.lhd_daily_to_df()

    assert (
        daily["capacity_committed"]
        <= daily["capacity_available"] + 1e-9
    ).all()
    assert (
        daily["capacity_used"]
        <= daily["capacity_committed"] + 1e-9
    ).all()

    expected_unused = (
        daily["capacity_available"]
        - daily["capacity_used"]
    ).clip(lower=0)

    np.testing.assert_allclose(
        daily["unused_capacity"],
        expected_unused,
    )


def test_pending_tests_and_traces_clear_after_completion(
    toy_model_factory,
):
    model = run_trace_workflow(
        toy_model_factory
    )

    assert model.lhd.state.pending_tests == {}
    assert model.lhd.state.pending_traces == {}