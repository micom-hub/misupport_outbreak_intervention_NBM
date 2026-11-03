from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd


EXPECTED_POLICIES = {
    "observe_only",
    "case_isolation_only",
    "trace_centered_response",
    "immediate_scalable_mitigation",
    "adaptive_escalation",
    "perfect_information",
}

TRACING_TASK_TYPES = {
    "contact_interview",
    "contact_notification",
    "testing_referral",
    "automated_notification",
}

ESCALATION_EVENT_TYPES = {
    "response_phase_started",
    "escalation_ordered",
    "escalation_started",
    "escalation_ended",
}

EXPECTED_TASK_TYPES_BY_POLICY = {
    "observe_only": set(),
    "case_isolation_only": {
        "case_investigation",
        "isolation_support",
    },
    "trace_centered_response": {
        "case_investigation",
        "isolation_support",
        "contact_interview",
        "contact_notification",
        "testing_referral",
    },
    "immediate_scalable_mitigation": {
        "case_investigation",
        "isolation_support",
        "automated_notification",
    },
    "adaptive_escalation": {
        "case_investigation",
        "isolation_support",
        "contact_interview",
        "contact_notification",
        "testing_referral",
        "automated_notification",
    },
    "perfect_information": set(),
}


def _safe_model_frame(model: Any, method_name: str) -> pd.DataFrame:
    method = getattr(model, method_name, None)

    if method is None:
        return pd.DataFrame()

    try:
        frame = method()
    except Exception:
        return pd.DataFrame()

    if not isinstance(frame, pd.DataFrame):
        try:
            frame = pd.DataFrame(frame)
        except Exception:
            return pd.DataFrame()

    return frame.copy()


def _model_policy_name(model: Any) -> str:
    try:
        return str(model.config.lhd.policy_name)
    except Exception:
        return str(
            getattr(
                getattr(model, "lhd", None),
                "policy_name",
                "unknown",
            )
        )


def _normalize_models(
    models: Union[
        Any,
        Iterable[Any],
        Mapping[str, Any],
    ],
) -> List[Tuple[str, Any]]:
    """
    Normalize one model, a sequence of models, or a name-to-model mapping.
    """
    if isinstance(models, Mapping):
        return [
            (str(name), model)
            for name, model in models.items()
        ]

    if isinstance(models, (list, tuple)):
        normalized = []

        for index, model in enumerate(models):
            policy = _model_policy_name(model)
            normalized.append(
                (f"{policy}__model_{index}", model)
            )

        return normalized

    policy = _model_policy_name(models)
    return [(policy, models)]


def _add_model_metadata(
    frame: pd.DataFrame,
    *,
    model_label: str,
    policy_name: str,
) -> pd.DataFrame:
    frame = frame.copy()

    if "audit_model_label" not in frame.columns:
        frame.insert(
            0,
            "audit_model_label",
            model_label,
        )

    if "policy_name" not in frame.columns:
        frame.insert(
            1,
            "policy_name",
            policy_name,
        )

    return frame


def _first_event_day(
    events: pd.DataFrame,
    event_type: str,
) -> Any:
    if events.empty or "event_type" not in events.columns:
        return pd.NA

    subset = events[
        events["event_type"] == event_type
    ]

    if subset.empty:
        return pd.NA

    return int(subset["day"].min())


def _last_event_day(
    events: pd.DataFrame,
    event_type: str,
) -> Any:
    if events.empty or "event_type" not in events.columns:
        return pd.NA

    subset = events[
        events["event_type"] == event_type
    ]

    if subset.empty:
        return pd.NA

    return int(subset["day"].max())


def _safe_sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0

    return float(
        pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(0).sum()
    )


def _safe_max(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    return float(values.max())


def _safe_count(
    frame: pd.DataFrame,
    column: str,
    value: Any,
) -> int:
    if frame.empty or column not in frame.columns:
        return 0

    return int((frame[column] == value).sum())


def _safe_bool_sum(
    frame: pd.DataFrame,
    column: str,
) -> int:
    if frame.empty or column not in frame.columns:
        return 0

    return int(
        frame[column]
        .fillna(False)
        .astype(bool)
        .sum()
    )


def _task_types(frame: pd.DataFrame) -> set:
    if frame.empty or "task_type" not in frame.columns:
        return set()

    return set(
        frame["task_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def _run_numbers_for_model(model: Any) -> range:
    return range(int(getattr(model, "n_replicates", 0)))


def audit_phase3_models(
    models: Union[
        Any,
        Iterable[Any],
        Mapping[str, Any],
    ],
    *,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Audit Phase 3 policy behavior across one or more simulated models.

    Best use:
        models = run_variants(...)
        audit = audit_phase3_models(models)
        print_phase3_audit(audit)

    Returns a dictionary containing:
      - passed
      - errors
      - warnings
      - diagnostic DataFrames
    """
    normalized_models = _normalize_models(models)

    errors: List[str] = []
    warnings: List[str] = []

    overview_rows = []
    run_rows = []
    phase_rows = []
    escalation_rows = []
    task_rows = []
    capacity_rows = []
    perfect_info_rows = []
    pairing_rows = []
    trigger_rows = []

    all_events = []
    all_daily_lhd = []
    all_daily_epi = []
    all_tasks = []
    all_factual = []

    seen_policies = []

    for model_label, model in normalized_models:
        policy_name = _model_policy_name(model)
        seen_policies.append(policy_name)

        events = _safe_model_frame(
            model,
            "event_log_to_df",
        )
        daily_lhd = _safe_model_frame(
            model,
            "lhd_daily_to_df",
        )
        daily_epi = _safe_model_frame(
            model,
            "daily_epi_to_df",
        )
        tasks = _safe_model_frame(
            model,
            "lhd_tasks_to_df",
        )
        factual = _safe_model_frame(
            model,
            "factual_run_summary_to_df",
        )
        transmissions = _safe_model_frame(
            model,
            "transmissions_to_df",
        )

        events = _add_model_metadata(
            events,
            model_label=model_label,
            policy_name=policy_name,
        )
        daily_lhd = _add_model_metadata(
            daily_lhd,
            model_label=model_label,
            policy_name=policy_name,
        )
        daily_epi = _add_model_metadata(
            daily_epi,
            model_label=model_label,
            policy_name=policy_name,
        )
        tasks = _add_model_metadata(
            tasks,
            model_label=model_label,
            policy_name=policy_name,
        )
        factual = _add_model_metadata(
            factual,
            model_label=model_label,
            policy_name=policy_name,
        )
        transmissions = _add_model_metadata(
            transmissions,
            model_label=model_label,
            policy_name=policy_name,
        )

        if not events.empty:
            all_events.append(events)

        if not daily_lhd.empty:
            all_daily_lhd.append(daily_lhd)

        if not daily_epi.empty:
            all_daily_epi.append(daily_epi)

        if not tasks.empty:
            all_tasks.append(tasks)

        if not factual.empty:
            all_factual.append(factual)

        if policy_name not in EXPECTED_POLICIES:
            warnings.append(
                f"{model_label}: policy '{policy_name}' is not in the "
                "simplified Phase 3 policy set."
            )

        observed_task_types = _task_types(tasks)
        expected_task_types = EXPECTED_TASK_TYPES_BY_POLICY.get(
            policy_name,
            set(),
        )
        unexpected_task_types = (
            observed_task_types - expected_task_types
        )

        if unexpected_task_types:
            errors.append(
                f"{model_label}: policy '{policy_name}' produced unexpected "
                f"task types: {sorted(unexpected_task_types)}"
            )

        if daily_lhd.empty:
            errors.append(
                f"{model_label}: no daily LHD output was produced."
            )

        if daily_epi.empty:
            errors.append(
                f"{model_label}: no daily epidemiological output was produced."
            )

        n_replicates = int(
            getattr(model, "n_replicates", 0)
        )

        overview_rows.append(
            {
                "audit_model_label": model_label,
                "policy_name": policy_name,
                "n_replicates": n_replicates,
                "n_events": int(len(events)),
                "n_daily_lhd_rows": int(len(daily_lhd)),
                "n_daily_epi_rows": int(len(daily_epi)),
                "n_tasks": int(len(tasks)),
                "n_transmissions": int(len(transmissions)),
                "observed_task_types": ", ".join(
                    sorted(observed_task_types)
                ),
                "unexpected_task_types": ", ".join(
                    sorted(unexpected_task_types)
                ),
                "escalation_orders": _safe_count(
                    events,
                    "event_type",
                    "escalation_ordered",
                ),
                "escalation_starts": _safe_count(
                    events,
                    "event_type",
                    "escalation_started",
                ),
                "perfect_information_identifications": _safe_count(
                    events,
                    "event_type",
                    "perfect_information_identification",
                ),
                "total_work_units": _safe_sum(
                    daily_lhd,
                    "capacity_used",
                ),
                "peak_pending_work_units": _safe_max(
                    daily_lhd,
                    "pending_work_units_end",
                ),
            }
        )

        # --------------------------------------------------------------
        # Policy-specific invariants
        # --------------------------------------------------------------

        if policy_name == "observe_only":
            if not tasks.empty:
                errors.append(
                    f"{model_label}: observe_only produced {len(tasks)} tasks."
                )

            if _safe_sum(daily_lhd, "capacity_used") > 1e-9:
                errors.append(
                    f"{model_label}: observe_only used operational capacity."
                )

            if _safe_count(
                events,
                "event_type",
                "isolation_started",
            ) > 0:
                errors.append(
                    f"{model_label}: observe_only started isolation."
                )

        if policy_name == "case_isolation_only":
            forbidden = (
                observed_task_types & TRACING_TASK_TYPES
            )

            if forbidden:
                errors.append(
                    f"{model_label}: case_isolation_only produced tracing "
                    f"tasks: {sorted(forbidden)}"
                )

        if policy_name == "trace_centered_response":
            if _safe_count(
                events,
                "event_type",
                "escalation_started",
            ) > 0:
                errors.append(
                    f"{model_label}: trace_centered_response escalated."
                )

            if "automated_notification" in observed_task_types:
                errors.append(
                    f"{model_label}: trace_centered_response used automated "
                    "notification without escalation."
                )

        if policy_name == "immediate_scalable_mitigation":
            if not events.empty:
                reports = events[
                    events["event_type"] == "report_received"
                ]
                orders = events[
                    events["event_type"] == "escalation_ordered"
                ]

                for run_number in _run_numbers_for_model(model):
                    run_reports = reports[
                        reports["run_number"] == run_number
                    ]
                    run_orders = orders[
                        orders["run_number"] == run_number
                    ]

                    if not run_reports.empty:
                        first_report = int(
                            run_reports["day"].min()
                        )

                        if run_orders.empty:
                            errors.append(
                                f"{model_label}, run {run_number}: immediate "
                                "mitigation received a report but never ordered "
                                "escalation."
                            )
                        else:
                            first_order = int(
                                run_orders["day"].min()
                            )

                            if first_order != first_report:
                                errors.append(
                                    f"{model_label}, run {run_number}: immediate "
                                    f"mitigation ordered escalation on day "
                                    f"{first_order}, not first-report day "
                                    f"{first_report}."
                                )

        if policy_name == "adaptive_escalation":
            if not daily_lhd.empty:
                if (
                    "escalation_trigger_value"
                    not in daily_lhd.columns
                ):
                    errors.append(
                        f"{model_label}: adaptive policy has no trigger values."
                    )

        if policy_name == "perfect_information":
            if not tasks.empty:
                errors.append(
                    f"{model_label}: perfect_information produced ordinary "
                    f"tasks ({len(tasks)} rows)."
                )

            if _safe_sum(daily_lhd, "capacity_used") > 1e-9:
                errors.append(
                    f"{model_label}: perfect_information used constrained "
                    "operational capacity."
                )

            if _safe_count(
                events,
                "event_type",
                "perfect_information_identification",
            ) == 0:
                warnings.append(
                    f"{model_label}: perfect_information produced no truth "
                    "identification events."
                )

        # --------------------------------------------------------------
        # Daily capacity and phase accounting
        # --------------------------------------------------------------

        if not daily_lhd.empty:
            for column in [
                "capacity_available",
                "capacity_committed",
                "capacity_used",
                "unused_capacity",
                "pending_work_units_end",
                "policy_phase",
            ]:
                if column not in daily_lhd.columns:
                    errors.append(
                        f"{model_label}: daily LHD output is missing "
                        f"'{column}'."
                    )

            required_capacity_columns = {
                "capacity_available",
                "capacity_committed",
                "capacity_used",
                "unused_capacity",
            }

            if required_capacity_columns.issubset(
                daily_lhd.columns
            ):
                available = pd.to_numeric(
                    daily_lhd["capacity_available"],
                    errors="coerce",
                )
                committed = pd.to_numeric(
                    daily_lhd["capacity_committed"],
                    errors="coerce",
                )
                used = pd.to_numeric(
                    daily_lhd["capacity_used"],
                    errors="coerce",
                )
                unused = pd.to_numeric(
                    daily_lhd["unused_capacity"],
                    errors="coerce",
                )

                if (committed > available + 1e-8).any():
                    errors.append(
                        f"{model_label}: committed capacity exceeds available "
                        "capacity."
                    )

                if (used > committed + 1e-8).any():
                    errors.append(
                        f"{model_label}: used capacity exceeds committed "
                        "capacity."
                    )

                expected_unused = (
                    available - used
                ).clip(lower=0)

                if not np.allclose(
                    unused,
                    expected_unused,
                    atol=1e-7,
                    equal_nan=True,
                ):
                    errors.append(
                        f"{model_label}: unused capacity does not reconcile."
                    )

            if "policy_phase" in daily_lhd.columns:
                invalid_phases = (
                    set(
                        pd.to_numeric(
                            daily_lhd["policy_phase"],
                            errors="coerce",
                        )
                        .dropna()
                        .astype(int)
                        .unique()
                    )
                    - {0, 1, 2}
                )

                if invalid_phases:
                    errors.append(
                        f"{model_label}: invalid policy phases "
                        f"{sorted(invalid_phases)}."
                    )

            for run_number, run_daily in daily_lhd.groupby(
                "run_number"
            ):
                run_daily = run_daily.sort_values("t")

                phase_values = pd.to_numeric(
                    run_daily["policy_phase"],
                    errors="coerce",
                ).dropna().to_numpy(dtype=int)

                if (
                    phase_values.size > 1
                    and np.any(
                        np.diff(phase_values) < 0
                    )
                ):
                    errors.append(
                        f"{model_label}, run {run_number}: policy phase "
                        "decreased."
                    )

                capacity_rows.append(
                    {
                        "audit_model_label": model_label,
                        "policy_name": policy_name,
                        "run_number": int(run_number),
                        "days_recorded": int(len(run_daily)),
                        "total_capacity_available": _safe_sum(
                            run_daily,
                            "capacity_available",
                        ),
                        "total_capacity_committed": _safe_sum(
                            run_daily,
                            "capacity_committed",
                        ),
                        "total_capacity_used": _safe_sum(
                            run_daily,
                            "capacity_used",
                        ),
                        "peak_capacity_used": _safe_max(
                            run_daily,
                            "capacity_used",
                        ),
                        "peak_pending_tasks": _safe_max(
                            run_daily,
                            "pending_tasks_end",
                        ),
                        "peak_pending_work_units": _safe_max(
                            run_daily,
                            "pending_work_units_end",
                        ),
                        "peak_backlog_ratio": _safe_max(
                            run_daily,
                            "backlog_to_capacity_ratio",
                        ),
                        "maximum_queue_age": _safe_max(
                            run_daily,
                            "max_queue_age",
                        ),
                        "days_at_full_capacity": int(
                            (
                                pd.to_numeric(
                                    run_daily["capacity_used"],
                                    errors="coerce",
                                )
                                >= pd.to_numeric(
                                    run_daily["capacity_available"],
                                    errors="coerce",
                                ) - 1e-9
                            ).sum()
                        ),
                        "final_pending_tasks": int(
                            run_daily[
                                "pending_tasks_end"
                            ].iloc[-1]
                        ),
                        "final_pending_work_units": float(
                            run_daily[
                                "pending_work_units_end"
                            ].iloc[-1]
                        ),
                    }
                )

                phase_rows.append(
                    {
                        "audit_model_label": model_label,
                        "policy_name": policy_name,
                        "run_number": int(run_number),
                        "first_phase_0_day": (
                            int(
                                run_daily.loc[
                                    run_daily["policy_phase"] == 0,
                                    "t",
                                ].min()
                            )
                            if (
                                run_daily["policy_phase"] == 0
                            ).any()
                            else pd.NA
                        ),
                        "first_phase_1_day": (
                            int(
                                run_daily.loc[
                                    run_daily["policy_phase"] == 1,
                                    "t",
                                ].min()
                            )
                            if (
                                run_daily["policy_phase"] == 1
                            ).any()
                            else pd.NA
                        ),
                        "first_phase_2_day": (
                            int(
                                run_daily.loc[
                                    run_daily["policy_phase"] == 2,
                                    "t",
                                ].min()
                            )
                            if (
                                run_daily["policy_phase"] == 2
                            ).any()
                            else pd.NA
                        ),
                        "final_phase": int(
                            run_daily["policy_phase"].iloc[-1]
                        ),
                        "phase_sequence": " → ".join(
                            map(
                                str,
                                run_daily[
                                    "policy_phase"
                                ]
                                .drop_duplicates()
                                .astype(int)
                                .tolist(),
                            )
                        ),
                    }
                )

                if policy_name == "adaptive_escalation":
                    trigger_metric = str(
                        model.config.lhd
                        .escalation_trigger_metric
                    )
                    threshold = float(
                        model.config.lhd
                        .escalation_trigger_threshold
                    )

                    if (
                        "escalation_trigger_value"
                        in run_daily.columns
                    ):
                        trigger_values = pd.to_numeric(
                            run_daily[
                                "escalation_trigger_value"
                            ],
                            errors="coerce",
                        )

                        for row_index, row in run_daily.iterrows():
                            trigger_rows.append(
                                {
                                    "audit_model_label": model_label,
                                    "policy_name": policy_name,
                                    "run_number": int(run_number),
                                    "day": int(row["t"]),
                                    "trigger_metric": trigger_metric,
                                    "threshold": threshold,
                                    "trigger_value": trigger_values.loc[
                                        row_index
                                    ],
                                    "triggered_today": bool(
                                        row.get(
                                            "escalation_triggered_today",
                                            False,
                                        )
                                    ),
                                    "policy_phase": int(
                                        row["policy_phase"]
                                    ),
                                }
                            )

        # --------------------------------------------------------------
        # Escalation event ordering
        # --------------------------------------------------------------

        if not events.empty:
            escalation_events = events[
                events["event_type"].isin(
                    ESCALATION_EVENT_TYPES
                )
            ].copy()

            for run_number in _run_numbers_for_model(model):
                run_events = events[
                    events["run_number"] == run_number
                ].sort_values(
                    ["day", "event_id"]
                )

                first_report = _first_event_day(
                    run_events,
                    "report_received",
                )
                phase1_day = _first_event_day(
                    run_events,
                    "response_phase_started",
                )
                order_day = _first_event_day(
                    run_events,
                    "escalation_ordered",
                )
                start_day = _first_event_day(
                    run_events,
                    "escalation_started",
                )
                end_day = _first_event_day(
                    run_events,
                    "escalation_ended",
                )

                if (
                    not pd.isna(order_day)
                    and not pd.isna(start_day)
                    and int(start_day) <= int(order_day)
                ):
                    errors.append(
                        f"{model_label}, run {run_number}: escalation started "
                        f"on day {start_day}, not after order day {order_day}."
                    )

                if (
                    not pd.isna(end_day)
                    and not pd.isna(start_day)
                    and int(end_day) <= int(start_day)
                ):
                    errors.append(
                        f"{model_label}, run {run_number}: escalation ended "
                        "before or on its start day."
                    )

                escalation_rows.append(
                    {
                        "audit_model_label": model_label,
                        "policy_name": policy_name,
                        "run_number": int(run_number),
                        "first_report_day": first_report,
                        "phase_1_start_day": phase1_day,
                        "escalation_order_day": order_day,
                        "escalation_start_day": start_day,
                        "escalation_end_day": end_day,
                        "order_to_start_delay": (
                            int(start_day) - int(order_day)
                            if (
                                not pd.isna(start_day)
                                and not pd.isna(order_day)
                            )
                            else pd.NA
                        ),
                        "n_escalation_orders": int(
                            (
                                run_events["event_type"]
                                == "escalation_ordered"
                            ).sum()
                        ),
                        "n_escalation_starts": int(
                            (
                                run_events["event_type"]
                                == "escalation_started"
                            ).sum()
                        ),
                    }
                )

                if (
                    (
                        run_events["event_type"]
                        == "escalation_ordered"
                    ).sum()
                    > 1
                ):
                    errors.append(
                        f"{model_label}, run {run_number}: escalation was "
                        "ordered more than once."
                    )

                if (
                    (
                        run_events["event_type"]
                        == "escalation_started"
                    ).sum()
                    > 1
                ):
                    errors.append(
                        f"{model_label}, run {run_number}: escalation started "
                        "more than once."
                    )

        # --------------------------------------------------------------
        # Task behavior by replicate
        # --------------------------------------------------------------

        for run_number in _run_numbers_for_model(model):
            run_tasks = (
                tasks[
                    tasks["run_number"] == run_number
                ].copy()
                if not tasks.empty
                else pd.DataFrame()
            )

            task_rows.append(
                {
                    "audit_model_label": model_label,
                    "policy_name": policy_name,
                    "run_number": int(run_number),
                    "tasks_total": int(len(run_tasks)),
                    "completed": _safe_count(
                        run_tasks,
                        "status",
                        "completed",
                    ),
                    "cancelled": _safe_count(
                        run_tasks,
                        "status",
                        "cancelled",
                    ),
                    "expired": _safe_count(
                        run_tasks,
                        "status",
                        "expired",
                    ),
                    "pending": _safe_count(
                        run_tasks,
                        "status",
                        "pending",
                    ),
                    "case_investigations": _safe_count(
                        run_tasks,
                        "task_type",
                        "case_investigation",
                    ),
                    "isolation_support": _safe_count(
                        run_tasks,
                        "task_type",
                        "isolation_support",
                    ),
                    "contact_interviews": _safe_count(
                        run_tasks,
                        "task_type",
                        "contact_interview",
                    ),
                    "contact_notifications": _safe_count(
                        run_tasks,
                        "task_type",
                        "contact_notification",
                    ),
                    "testing_referrals": _safe_count(
                        run_tasks,
                        "task_type",
                        "testing_referral",
                    ),
                    "automated_notifications": _safe_count(
                        run_tasks,
                        "task_type",
                        "automated_notification",
                    ),
                    "cancelled_for_escalation": (
                        int(
                            (
                                run_tasks[
                                    "cancellation_reason"
                                ]
                                == "policy_escalation"
                            ).sum()
                        )
                        if (
                            not run_tasks.empty
                            and "cancellation_reason"
                            in run_tasks.columns
                        )
                        else 0
                    ),
                }
            )

        # --------------------------------------------------------------
        # Factual run summary
        # --------------------------------------------------------------

        for run_number in _run_numbers_for_model(model):
            run_factual = (
                factual[
                    factual["run_number"] == run_number
                ]
                if not factual.empty
                else pd.DataFrame()
            )

            if not run_factual.empty:
                factual_row = run_factual.iloc[0]

                total_local_infections = int(
                    factual_row.get(
                        "total_local_infections",
                        0,
                    )
                )
                total_work_units = float(
                    factual_row.get(
                        "total_work_units",
                        0.0,
                    )
                )
                first_detection_day = factual_row.get(
                    "first_detection_day",
                    pd.NA,
                )
                first_report_day = factual_row.get(
                    "first_report_day",
                    pd.NA,
                )
                escalation_effective_day = factual_row.get(
                    "escalation_effective_day",
                    pd.NA,
                )
            else:
                generation = np.asarray(
                    model.all_infection_generation[
                        run_number
                    ]
                )
                total_local_infections = int(
                    np.count_nonzero(
                        generation > 0
                    )
                )
                total_work_units = 0.0
                first_detection_day = pd.NA
                first_report_day = pd.NA
                escalation_effective_day = pd.NA

            run_rows.append(
                {
                    "audit_model_label": model_label,
                    "policy_name": policy_name,
                    "run_number": int(run_number),
                    "total_local_infections":
                        total_local_infections,
                    "total_work_units":
                        total_work_units,
                    "first_detection_day":
                        first_detection_day,
                    "first_report_day":
                        first_report_day,
                    "escalation_effective_day":
                        escalation_effective_day,
                    "end_day": int(
                        model.all_end_days[
                            run_number
                        ]
                    ),
                    "stochastic_dieout": bool(
                        model.all_stochastic_dieout[
                            run_number
                        ]
                    ),
                    "imported_cases": int(
                        np.asarray(
                            model.all_imported_ids[
                                run_number
                            ]
                        ).size
                    ),
                }
            )

        # --------------------------------------------------------------
        # Perfect-information diagnostics
        # --------------------------------------------------------------

        if policy_name == "perfect_information":
            for run_number in _run_numbers_for_model(model):
                run_events = events[
                    events["run_number"] == run_number
                ]

                generation = np.asarray(
                    model.all_infection_generation[
                        run_number
                    ]
                )

                local_infections = int(
                    np.count_nonzero(generation > 0)
                )
                truth_identifications = int(
                    (
                        run_events["event_type"]
                        == "perfect_information_identification"
                    ).sum()
                )
                day_zero_exposures = int(
                    (
                        (
                            run_events["event_type"]
                            == "exposure"
                        )
                        & (run_events["day"] == 0)
                    ).sum()
                )
                post_day_zero_exposures = int(
                    (
                        (
                            run_events["event_type"]
                            == "exposure"
                        )
                        & (run_events["day"] > 0)
                    ).sum()
                )

                perfect_info_rows.append(
                    {
                        "audit_model_label": model_label,
                        "policy_name": policy_name,
                        "run_number": int(run_number),
                        "truth_identifications":
                            truth_identifications,
                        "local_infections":
                            local_infections,
                        "day_zero_exposures":
                            day_zero_exposures,
                        "post_day_zero_exposures":
                            post_day_zero_exposures,
                        "ordinary_tasks":
                            int(
                                (
                                    tasks["run_number"]
                                    == run_number
                                ).sum()
                            )
                            if not tasks.empty
                            else 0,
                        "capacity_used":
                            _safe_sum(
                                daily_lhd[
                                    daily_lhd[
                                        "run_number"
                                    ] == run_number
                                ],
                                "capacity_used",
                            )
                            if not daily_lhd.empty
                            else 0.0,
                    }
                )

                reduction = float(
                    model.config.lhd
                    .perfect_information_reduction
                )

                if (
                    reduction >= 1.0
                    and local_infections > 0
                ):
                    warnings.append(
                        f"{model_label}, run {run_number}: perfect information "
                        f"with reduction={reduction} produced "
                        f"{local_infections} local infections. Because the "
                        "current multiplier floor is nonzero, inspect whether "
                        "this is a rare residual transmission or an ordering "
                        "defect."
                    )

        # --------------------------------------------------------------
        # Pairing signatures
        # --------------------------------------------------------------

        for run_number in _run_numbers_for_model(model):
            imported = tuple(
                np.asarray(
                    model.all_imported_ids[
                        run_number
                    ],
                    dtype=np.int32,
                ).tolist()
            )

            vaccination = np.asarray(
                model.all_vax_status[
                    run_number
                ],
                dtype=bool,
            )

            initial_states = model.all_states_over_time[
                run_number
            ][0]

            pairing_rows.append(
                {
                    "audit_model_label": model_label,
                    "policy_name": policy_name,
                    "run_number": int(run_number),
                    "replicate_seed": int(
                        model._seed_metadata.get(
                            "replicate_seed",
                            -1,
                        )
                    )
                    if (
                        int(model.replicate_ind)
                        == run_number
                    )
                    else pd.NA,
                    "imported_ids": imported,
                    "vaccinated_count": int(
                        vaccination.sum()
                    ),
                    "initial_susceptible_count": int(
                        len(initial_states[0])
                    ),
                    "initial_exposed_count": int(
                        len(initial_states[1])
                    ),
                    "initial_infectious_count": int(
                        len(initial_states[2])
                    ),
                    "initial_recovered_count": int(
                        len(initial_states[3])
                    ),
                }
            )

    # ------------------------------------------------------------------
    # Cross-policy pairing checks
    # ------------------------------------------------------------------

    pairing_df = pd.DataFrame(pairing_rows)

    if not pairing_df.empty:
        for run_number, group in pairing_df.groupby(
            "run_number"
        ):
            imported_signatures = group[
                "imported_ids"
            ].apply(tuple)

            if imported_signatures.nunique() > 1:
                errors.append(
                    f"Run {run_number}: imported-case identities differ "
                    "across policy models."
                )

            if (
                group["vaccinated_count"].nunique()
                > 1
            ):
                errors.append(
                    f"Run {run_number}: vaccination counts differ across "
                    "policy models."
                )

            for column in [
                "initial_susceptible_count",
                "initial_exposed_count",
                "initial_infectious_count",
                "initial_recovered_count",
            ]:
                if group[column].nunique() > 1:
                    errors.append(
                        f"Run {run_number}: {column} differs across policies."
                    )

    # ------------------------------------------------------------------
    # Assemble diagnostics
    # ------------------------------------------------------------------

    overview_df = pd.DataFrame(overview_rows)
    run_summary_df = pd.DataFrame(run_rows)
    phase_summary_df = pd.DataFrame(phase_rows)
    escalation_summary_df = pd.DataFrame(
        escalation_rows
    )
    task_summary_df = pd.DataFrame(task_rows)
    capacity_summary_df = pd.DataFrame(
        capacity_rows
    )
    perfect_info_df = pd.DataFrame(
        perfect_info_rows
    )
    trigger_diagnostics_df = pd.DataFrame(
        trigger_rows
    )

    combined_events = (
        pd.concat(
            all_events,
            ignore_index=True,
            sort=False,
        )
        if all_events
        else pd.DataFrame()
    )
    combined_daily_lhd = (
        pd.concat(
            all_daily_lhd,
            ignore_index=True,
            sort=False,
        )
        if all_daily_lhd
        else pd.DataFrame()
    )
    combined_daily_epi = (
        pd.concat(
            all_daily_epi,
            ignore_index=True,
            sort=False,
        )
        if all_daily_epi
        else pd.DataFrame()
    )
    combined_tasks = (
        pd.concat(
            all_tasks,
            ignore_index=True,
            sort=False,
        )
        if all_tasks
        else pd.DataFrame()
    )
    combined_factual = (
        pd.concat(
            all_factual,
            ignore_index=True,
            sort=False,
        )
        if all_factual
        else pd.DataFrame()
    )

    policy_outcome_comparison = pd.DataFrame()

    if not run_summary_df.empty:
        policy_outcome_comparison = (
            run_summary_df.groupby(
                "policy_name",
                dropna=False,
            )
            .agg(
                n_runs=("run_number", "size"),
                mean_local_infections=(
                    "total_local_infections",
                    "mean",
                ),
                median_local_infections=(
                    "total_local_infections",
                    "median",
                ),
                min_local_infections=(
                    "total_local_infections",
                    "min",
                ),
                max_local_infections=(
                    "total_local_infections",
                    "max",
                ),
                mean_work_units=(
                    "total_work_units",
                    "mean",
                ),
                dieout_fraction=(
                    "stochastic_dieout",
                    "mean",
                ),
            )
            .reset_index()
            .sort_values("policy_name")
        )

    paired_outcomes = pd.DataFrame()

    if (
        not run_summary_df.empty
        and (
            run_summary_df["policy_name"]
            == "observe_only"
        ).any()
    ):
        baseline = run_summary_df[
            run_summary_df["policy_name"]
            == "observe_only"
        ][
            [
                "run_number",
                "total_local_infections",
            ]
        ].rename(
            columns={
                "total_local_infections":
                    "baseline_local_infections"
            }
        )

        paired_outcomes = (
            run_summary_df[
                run_summary_df["policy_name"]
                != "observe_only"
            ]
            .merge(
                baseline,
                on="run_number",
                how="inner",
                validate="many_to_one",
            )
        )

        paired_outcomes[
            "infections_averted"
        ] = (
            paired_outcomes[
                "baseline_local_infections"
            ]
            - paired_outcomes[
                "total_local_infections"
            ]
        )

        paired_outcomes = paired_outcomes[
            [
                "audit_model_label",
                "policy_name",
                "run_number",
                "baseline_local_infections",
                "total_local_infections",
                "infections_averted",
                "total_work_units",
                "escalation_effective_day",
            ]
        ].sort_values(
            ["run_number", "policy_name"]
        )

    missing_expected_policies = (
        EXPECTED_POLICIES - set(seen_policies)
    )

    if missing_expected_policies:
        warnings.append(
            "The audit did not receive all Phase 3 policies. Missing: "
            f"{sorted(missing_expected_policies)}"
        )

    result = {
        "passed": len(errors) == 0,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "policies_seen": sorted(set(seen_policies)),
        "overview": overview_df,
        "run_summary": run_summary_df,
        "policy_outcome_comparison":
            policy_outcome_comparison,
        "paired_outcomes": paired_outcomes,
        "phase_summary": phase_summary_df,
        "escalation_summary":
            escalation_summary_df,
        "trigger_diagnostics":
            trigger_diagnostics_df,
        "task_summary": task_summary_df,
        "capacity_summary":
            capacity_summary_df,
        "perfect_information_summary":
            perfect_info_df,
        "pairing_summary": pairing_df,
        "events": combined_events,
        "daily_lhd": combined_daily_lhd,
        "daily_epi": combined_daily_epi,
        "tasks": combined_tasks,
        "factual_summary": combined_factual,
    }

    if raise_on_error and errors:
        message = "\n".join(
            f"- {error}"
            for error in errors
        )
        raise AssertionError(
            "Phase 3 audit failed:\n"
            f"{message}"
        )

    return result


def _print_dataframe(
    name: str,
    frame: pd.DataFrame,
    *,
    max_rows: int = 100,
) -> None:
    print(f"\n=== {name} ===")

    if frame is None or frame.empty:
        print("<empty>")
        return

    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        80,
    ):
        print(
            frame.head(max_rows).to_string(
                index=False
            )
        )

    if len(frame) > max_rows:
        print(
            f"... showing {max_rows} of "
            f"{len(frame)} rows"
        )


def print_phase3_audit(
    result: Dict[str, Any],
    *,
    max_rows: int = 100,
    include_raw_samples: bool = False,
) -> None:
    """
    Print the main Phase 3 diagnostic tables to the console.
    """
    print("\n========================================")
    print("PHASE 3 MULTI-POLICY AUDIT")
    print("========================================")
    print("Passed:", result["passed"])
    print("Errors:", result["n_errors"])
    print("Warnings:", result["n_warnings"])
    print(
        "Policies:",
        ", ".join(result["policies_seen"]),
    )

    if result["errors"]:
        print("\nERRORS")
        for error in result["errors"]:
            print(" -", error)

    if result["warnings"]:
        print("\nWARNINGS")
        for warning in result["warnings"]:
            print(" -", warning)

    _print_dataframe(
        "POLICY OVERVIEW",
        result["overview"],
        max_rows=max_rows,
    )

    _print_dataframe(
        "RUN-LEVEL FACTUAL SUMMARY",
        result["run_summary"],
        max_rows=max_rows,
    )

    _print_dataframe(
        "POLICY OUTCOME COMPARISON",
        result["policy_outcome_comparison"],
        max_rows=max_rows,
    )

    _print_dataframe(
        "PAIRED OUTCOMES VS OBSERVE_ONLY",
        result["paired_outcomes"],
        max_rows=max_rows,
    )

    _print_dataframe(
        "POLICY PHASE SUMMARY",
        result["phase_summary"],
        max_rows=max_rows,
    )

    _print_dataframe(
        "ESCALATION SUMMARY",
        result["escalation_summary"],
        max_rows=max_rows,
    )

    _print_dataframe(
        "ADAPTIVE TRIGGER DIAGNOSTICS",
        result["trigger_diagnostics"],
        max_rows=max_rows,
    )

    _print_dataframe(
        "TASK SUMMARY",
        result["task_summary"],
        max_rows=max_rows,
    )

    _print_dataframe(
        "CAPACITY AND BACKLOG SUMMARY",
        result["capacity_summary"],
        max_rows=max_rows,
    )

    _print_dataframe(
        "PERFECT-INFORMATION SUMMARY",
        result["perfect_information_summary"],
        max_rows=max_rows,
    )

    _print_dataframe(
        "CROSS-POLICY PAIRING SUMMARY",
        result["pairing_summary"],
        max_rows=max_rows,
    )

    if include_raw_samples:
        event_columns = [
            column
            for column in [
                "audit_model_label",
                "policy_name",
                "event_id",
                "run_number",
                "day",
                "event_type",
                "node_id",
                "source_node_id",
                "policy_phase",
                "result",
                "value",
                "metadata_json",
            ]
            if column in result["events"].columns
        ]

        _print_dataframe(
            "RAW EVENT SAMPLE",
            result["events"][event_columns]
            if event_columns
            else result["events"],
            max_rows=max_rows,
        )

        _print_dataframe(
            "RAW DAILY LHD SAMPLE",
            result["daily_lhd"],
            max_rows=max_rows,
        )

        _print_dataframe(
            "RAW TASK SAMPLE",
            result["tasks"],
            max_rows=max_rows,
        )