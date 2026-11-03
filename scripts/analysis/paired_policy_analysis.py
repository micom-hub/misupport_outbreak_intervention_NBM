from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def paired_policy_differences(
    factual_summary: pd.DataFrame,
    *,
    baseline_policy: str = "observe_only",
    metrics: Optional[Iterable[str]] = None,
    policy_col: str = "variant_name",
) -> pd.DataFrame:
    """
    Calculate paired policy differences using model_index and run_number.

    Positive infections_averted means the comparison policy produced fewer
    local infections than the baseline.
    """
    required = {
        "model_index",
        "run_number",
        policy_col,
    }

    missing = required - set(
        factual_summary.columns
    )
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if metrics is None:
        metrics = [
            "total_local_infections",
            "peak_local_prevalence_count",
            "peak_local_incidence_count",
            "total_work_units",
            "peak_pending_work_units",
        ]

    metrics = list(metrics)

    baseline = factual_summary[
        factual_summary[policy_col]
        == baseline_policy
    ][
        [
            "model_index",
            "run_number",
            *metrics,
        ]
    ].copy()

    if baseline.empty:
        raise ValueError(
            f"Baseline policy '{baseline_policy}' "
            "was not found."
        )

    baseline = baseline.rename(
        columns={
            metric: f"{metric}_baseline"
            for metric in metrics
        }
    )

    comparisons = factual_summary[
        factual_summary[policy_col]
        != baseline_policy
    ].copy()

    paired = comparisons.merge(
        baseline,
        on=["model_index", "run_number"],
        how="inner",
        validate="many_to_one",
    )

    for metric in metrics:
        paired[
            f"{metric}_difference"
        ] = (
            paired[metric]
            - paired[f"{metric}_baseline"]
        )

    if (
        "total_local_infections"
        in metrics
    ):
        paired["infections_averted"] = (
            paired[
                "total_local_infections_baseline"
            ]
            - paired[
                "total_local_infections"
            ]
        )

    return paired