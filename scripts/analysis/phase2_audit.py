from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def audit_phase2_model(
    model,
    *,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    tasks = model.lhd_tasks_to_df()
    daily = model.lhd_daily_to_df()
    events = model.event_log_to_df()

    if daily.empty:
        errors.append(
            "No daily LHD output was produced."
        )

    if not tasks.empty:
        if tasks["task_id"].duplicated().any():
            errors.append(
                "Task IDs are not unique."
            )

        valid_statuses = {
            "pending",
            "selected",
            "in_progress",
            "completed",
            "expired",
            "cancelled",
        }
        invalid_statuses = set(
            tasks["status"].dropna().unique()
        ) - valid_statuses

        if invalid_statuses:
            errors.append(
                "Invalid task statuses: "
                f"{sorted(invalid_statuses)}"
            )

        completed = tasks[
            tasks["status"] == "completed"
        ]

        if completed["completed_day"].isna().any():
            errors.append(
                "At least one completed task has no completion day."
            )

        if (
            completed["completed_day"]
            < completed["created_day"]
        ).any():
            errors.append(
                "At least one task completed before creation."
            )

        expired = tasks[
            tasks["status"] == "expired"
        ]

        if expired["expired_day"].isna().any():
            errors.append(
                "At least one expired task has no expiration day."
            )

        if (
            tasks["earliest_start_day"]
            < tasks["created_day"]
        ).any():
            errors.append(
                "A task is eligible before its creation day."
            )

        if (tasks["work_cost"] < 0).any():
            errors.append(
                "A task has negative work cost."
            )

        # No active duplicate keys in final state.
        active = tasks[
            tasks["status"].isin(
                [
                    "pending",
                    "selected",
                    "in_progress",
                ]
            )
        ].copy()

        if not active.empty:
            active["dedup_audit_key"] = (
                active["task_type"].astype(str)
                + "|"
                + active["target"].astype(str)
                + "|"
                + active["source_case"]
                .fillna(-1)
                .astype(str)
            )

            duplicated = active[
                "dedup_audit_key"
            ].duplicated()

            if duplicated.any():
                errors.append(
                    "Duplicate active tasks remain in the queue."
                )

    if not daily.empty:
        numeric_nonnegative = [
            "capacity_available",
            "capacity_committed",
            "capacity_used",
            "unused_capacity",
            "pending_tasks_start",
            "pending_tasks_end",
            "pending_work_units_start",
            "pending_work_units_end",
            "tasks_selected",
            "tasks_completed",
            "tasks_expired",
            "tasks_cancelled",
        ]

        for column in numeric_nonnegative:
            if (
                column in daily.columns
                and (daily[column] < -1e-9).any()
            ):
                errors.append(
                    f"Daily column '{column}' contains "
                    "negative values."
                )

        if (
            daily["capacity_committed"]
            > daily["capacity_available"] + 1e-9
        ).any():
            errors.append(
                "Committed capacity exceeds available capacity."
            )

        if (
            daily["capacity_used"]
            > daily["capacity_committed"] + 1e-9
        ).any():
            errors.append(
                "Used capacity exceeds committed capacity."
            )

        expected_unused = (
            daily["capacity_available"]
            - daily["capacity_used"]
        ).clip(lower=0)

        if not np.allclose(
            expected_unused,
            daily["unused_capacity"],
            atol=1e-7,
            equal_nan=True,
        ):
            errors.append(
                "Unused capacity does not reconcile."
            )

        if (
            daily["tasks_completed"]
            + daily["tasks_cancelled"]
            > daily["tasks_selected"]
        ).any():
            errors.append(
                "Completed plus cancelled tasks exceed selected tasks."
            )

        # Backlog continuity:
        # prior-day pending end should generally reconcile with next-day
        # pending start because child tasks are already counted at prior end.
        for run_number, group in daily.groupby(
            "run_number"
        ):
            group = group.sort_values("t")

            prior_end = group[
                "pending_tasks_end"
            ].to_numpy()[:-1]
            next_start = group[
                "pending_tasks_start"
            ].to_numpy()[1:]

            if not np.array_equal(
                prior_end,
                next_start,
            ):
                warnings.append(
                    f"Run {run_number}: pending task count does "
                    "not perfectly carry between adjacent days. "
                    "Inspect task expiration or stopping behavior."
                )

        observe_rows = daily[
            daily["policy_name"] == "observe_only"
        ]

        if (
            not observe_rows.empty
            and (
                observe_rows["tasks_selected"] > 0
            ).any()
        ):
            errors.append(
                "observe_only selected operational tasks."
            )

    task_event_counts = {}

    if not events.empty:
        task_events = events[
            events["event_type"].str.startswith(
                "task_",
                na=False,
            )
        ]
        task_event_counts = (
            task_events["event_type"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    result = {
        "passed": len(errors) == 0,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "n_tasks": int(len(tasks)),
        "n_daily_lhd_rows": int(len(daily)),
        "task_status_counts": (
            tasks["status"]
            .value_counts()
            .to_dict()
            if not tasks.empty
            else {}
        ),
        "task_type_counts": (
            tasks["task_type"]
            .value_counts()
            .to_dict()
            if not tasks.empty
            else {}
        ),
        "task_event_counts":
            task_event_counts,
    }

    if raise_on_error and errors:
        formatted = "\n".join(
            f"- {error}"
            for error in errors
        )
        raise AssertionError(
            "Phase 2 audit failed:\n"
            f"{formatted}"
        )

    return result


def print_phase2_audit(
    result: Dict[str, Any],
) -> None:
    print("\n=== PHASE 2 AUDIT ===")
    print("Passed:", result["passed"])
    print("Tasks:", result["n_tasks"])
    print(
        "Daily LHD rows:",
        result["n_daily_lhd_rows"],
    )
    print("Errors:", result["n_errors"])
    print("Warnings:", result["n_warnings"])

    print("\nTask statuses:")
    for key, value in result[
        "task_status_counts"
    ].items():
        print(f"  {key}: {value}")

    print("\nTask types:")
    for key, value in result[
        "task_type_counts"
    ].items():
        print(f"  {key}: {value}")

    print("\nTask events:")
    for key, value in result[
        "task_event_counts"
    ].items():
        print(f"  {key}: {value}")

    if result["errors"]:
        print("\nErrors:")
        for error in result["errors"]:
            print(" -", error)

    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(" -", warning)