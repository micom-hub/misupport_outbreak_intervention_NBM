from __future__ import annotations

import numpy as np

from scripts.lhd.surveillance import SurveillanceModel


def make_surveillance(
    toy_model_factory,
    **overrides,
):
    parameters = {
        "neighbor_map":
            toy_model_factory.graphdata.neighbor_map,
        "ct_to_id":
            toy_model_factory.graphdata.ct_to_id,
        "seed": 100,
        "N": toy_model_factory.graphdata.N,
        "ages": toy_model_factory.graphdata.ages,
        "is_vax": np.zeros(6, dtype=bool),
        "p_detect_symptomatic_daily": 1.0,
        "p_detect_asymptomatic_daily": 0.0,
        "p_detect_imported_daily": 0.0,
        "imported_detection_start_day": 1,
        "passive_report_delay_days": 0,
    }
    parameters.update(overrides)

    return SurveillanceModel(**parameters)


def epi_state(
    *,
    infectious=(),
    exposed=(),
    symptomatic=(),
    imported=(),
):
    return {
        "inf_ids": np.asarray(
            infectious,
            dtype=np.int32,
        ),
        "pre_ids": np.asarray(
            exposed,
            dtype=np.int32,
        ),
        "symptomatic_ids": np.asarray(
            symptomatic,
            dtype=np.int32,
        ),
        "imported_ids": np.asarray(
            imported,
            dtype=np.int32,
        ),
        "new_inf_ids": np.empty(
            0,
            dtype=np.int32,
        ),
    }


def test_symptomatic_cases_receive_daily_detection(
    toy_model_factory,
):
    surveillance = make_surveillance(
        toy_model_factory
    )

    batch = surveillance.step(
        t=1,
        epi_state=epi_state(
            infectious=[1],
            symptomatic=[1],
        ),
    )

    assert batch["detected_cases"].tolist() == [1]
    assert batch["reported_cases"].tolist() == [1]
    assert (
        batch["detection_sources"].tolist()
        == ["passive_symptomatic"]
    )


def test_asymptomatic_detection_uses_configured_hazard(
    toy_model_factory,
):
    surveillance = make_surveillance(
        toy_model_factory,
        p_detect_symptomatic_daily=0.0,
        p_detect_asymptomatic_daily=1.0,
    )

    batch = surveillance.step(
        t=1,
        epi_state=epi_state(
            infectious=[2],
            symptomatic=[],
        ),
    )

    assert batch["detected_cases"].tolist() == [2]
    assert (
        batch["detection_sources"].tolist()
        == ["passive_asymptomatic"]
    )


def test_imported_ascertainment_begins_on_start_day(
    toy_model_factory,
):
    surveillance = make_surveillance(
        toy_model_factory,
        p_detect_symptomatic_daily=0.0,
        p_detect_asymptomatic_daily=0.0,
        p_detect_imported_daily=1.0,
        imported_detection_start_day=2,
    )

    day_one = surveillance.step(
        t=1,
        epi_state=epi_state(
            infectious=[0],
            imported=[0],
        ),
    )
    day_two = surveillance.step(
        t=2,
        epi_state=epi_state(
            infectious=[0],
            imported=[0],
        ),
    )

    assert day_one["detected_cases"].size == 0
    assert day_two["detected_cases"].tolist() == [0]


def test_case_is_detected_and_reported_only_once(
    toy_model_factory,
):
    surveillance = make_surveillance(
        toy_model_factory
    )

    first = surveillance.step(
        t=1,
        epi_state=epi_state(
            infectious=[1],
            symptomatic=[1],
        ),
    )
    second = surveillance.step(
        t=2,
        epi_state=epi_state(
            infectious=[1],
            symptomatic=[1],
        ),
    )

    assert first["detected_cases"].tolist() == [1]
    assert first["reported_cases"].tolist() == [1]
    assert second["detected_cases"].size == 0
    assert second["reported_cases"].size == 0


def test_report_delay_is_respected(
    toy_model_factory,
):
    surveillance = make_surveillance(
        toy_model_factory,
        passive_report_delay_days=2,
    )

    first = surveillance.step(
        t=1,
        epi_state=epi_state(
            infectious=[1],
            symptomatic=[1],
        ),
    )
    second = surveillance.step(
        t=2,
        epi_state=epi_state(
            infectious=[1],
            symptomatic=[1],
        ),
    )
    third = surveillance.step(
        t=3,
        epi_state=epi_state(
            infectious=[1],
            symptomatic=[1],
        ),
    )

    assert first["detected_cases"].tolist() == [1]
    assert first["reported_cases"].size == 0
    assert second["reported_cases"].size == 0
    assert third["reported_cases"].tolist() == [1]


def test_positive_active_test_creates_detection_and_report(
    toy_model_factory,
):
    surveillance = make_surveillance(
        toy_model_factory,
        p_detect_symptomatic_daily=0.0,
        p_detect_asymptomatic_daily=0.0,
    )

    surveillance.order_test(
        t=0,
        nodes=np.asarray([2], dtype=np.int32),
        params={
            "delay_days": 1,
            "sensitivity_exposed": 1.0,
            "sensitivity_infectious": 1.0,
            "specificity": 1.0,
            "report_delay_days": 0,
        },
    )

    batch = surveillance.step(
        t=1,
        epi_state=epi_state(
            exposed=[2],
        ),
    )

    assert batch["tests_completed"].tolist() == [2]
    assert batch["test_results_positive"].tolist() == [True]
    assert batch["detected_cases"].tolist() == [2]
    assert batch["reported_cases"].tolist() == [2]


def test_negative_test_does_not_create_report(
    toy_model_factory,
):
    surveillance = make_surveillance(
        toy_model_factory,
        p_detect_symptomatic_daily=0.0,
        p_detect_asymptomatic_daily=0.0,
    )

    surveillance.order_test(
        t=0,
        nodes=np.asarray([3], dtype=np.int32),
        params={
            "delay_days": 1,
            "sensitivity_exposed": 0.0,
            "sensitivity_infectious": 0.0,
            "specificity": 1.0,
            "report_delay_days": 0,
        },
    )

    batch = surveillance.step(
        t=1,
        epi_state=epi_state(),
    )

    assert batch["tests_completed"].tolist() == [3]
    assert batch["test_results_positive"].tolist() == [False]
    assert batch["detected_cases"].size == 0
    assert batch["reported_cases"].size == 0


def test_test_sensitivity_depends_on_stage(
    toy_model_factory,
):
    surveillance = make_surveillance(
        toy_model_factory,
        p_detect_symptomatic_daily=0.0,
        p_detect_asymptomatic_daily=0.0,
    )

    surveillance.order_test(
        t=0,
        nodes=np.asarray(
            [1, 2],
            dtype=np.int32,
        ),
        params={
            "delay_days": 1,
            "sensitivity_exposed": 0.0,
            "sensitivity_infectious": 1.0,
            "specificity": 1.0,
            "report_delay_days": 0,
        },
    )

    batch = surveillance.step(
        t=1,
        epi_state=epi_state(
            exposed=[1],
            infectious=[2],
        ),
    )

    result_by_node = dict(
        zip(
            batch["tests_completed"].tolist(),
            batch["test_results_positive"].tolist(),
        )
    )

    assert result_by_node[1] is False
    assert result_by_node[2] is True


def test_trace_delay_recall_and_edge_validity(
    toy_model_factory,
):
    surveillance = make_surveillance(
        toy_model_factory,
        p_detect_symptomatic_daily=0.0,
        p_detect_asymptomatic_daily=0.0,
    )

    surveillance.order_trace(
        t=0,
        cases=np.asarray([0], dtype=np.int32),
        params={
            "delay_days": 2,
            "recall_prob": 1.0,
            "max_per_case": 25,
            "contact_types": [
                "hh",
                "cas",
            ],
        },
    )

    day_one = surveillance.step(
        t=1,
        epi_state=epi_state(),
    )
    day_two = surveillance.step(
        t=2,
        epi_state=epi_state(),
    )

    assert day_one["trace_src"].size == 0

    returned = set(
        zip(
            day_two["trace_src"].tolist(),
            day_two["trace_tgt"].tolist(),
            day_two["trace_ct"].tolist(),
        )
    )

    expected = {
        (
            0,
            target,
            toy_model_factory.graphdata.ct_to_id[
                contact_type
            ],
        )
        for target, _weight, contact_type
        in toy_model_factory.graphdata.neighbor_map[0]
        if contact_type in {"hh", "cas"}
    }

    assert returned == expected
    assert day_two["traces_completed"].tolist() == [0]