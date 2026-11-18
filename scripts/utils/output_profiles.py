# scripts/utils/output_profiles.py
# Define "output profiles" for model runs so that you don't need to type "save x, y, z, ...." in each function call
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Set

import pandas as pd


OUTPUT_PROFILES: Mapping[str, Set[str]] = {
    "none": set(),
    "core": {
        "factual_summary",
        "daily_epi",
        "daily_lhd",
    },
    "audit": {
        "factual_summary",
        "daily_epi",
        "daily_lhd",
        "events",
        "tasks",
    },
    "full": {
        "factual_summary",
        "daily_epi",
        "daily_lhd",
        "events",
        "tasks",
        "incidence",
        "prevalence",
        "transmissions",
        "sampled_edges",
    },
}


OUTPUT_FILENAMES = {
    "factual_summary": "factual_run_summary.parquet",
    "daily_epi": "daily_epi.parquet",
    "daily_lhd": "daily_lhd.parquet",
    "events": "events.parquet",
    "tasks": "tasks.parquet",
    "incidence": "incidence.parquet",
    "prevalence": "prevalence.parquet",
    "transmissions": "transmissions.parquet",
    "sampled_edges": "SampledEdgelist.parquet",
}


AGGREGATED_FILENAMES = {
    "factual_summary":
        "aggregated_factual_summary.parquet",
    "daily_epi":
        "aggregated_daily_epi.parquet",
    "daily_lhd":
        "aggregated_daily_lhd.parquet",
    "events":
        "aggregated_events.parquet",
    "tasks":
        "aggregated_tasks.parquet",
    "incidence":
        "aggregated_incidence.parquet",
    "prevalence":
        "aggregated_prevalence.parquet",
    "transmissions":
        "aggregated_transmissions.parquet",
}


def resolve_outputs(
    profile: str,
    extra_outputs: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Resolve a named output profile and any explicitly requested outputs."""
    profile = str(profile).strip().lower()

    if profile not in OUTPUT_PROFILES:
        raise ValueError(
            f"Unknown output profile '{profile}'. "
            f"Choose from {sorted(OUTPUT_PROFILES)}."
        )

    outputs = set(OUTPUT_PROFILES[profile])

    if extra_outputs is not None:
        outputs.update(
            str(output).strip()
            for output in extra_outputs
        )

    unknown = outputs - set(OUTPUT_FILENAMES)

    if unknown:
        raise ValueError(
            "Unknown requested outputs: "
            f"{sorted(unknown)}"
        )

    return outputs


def atomic_write_parquet(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """Write a Parquet file atomically."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    dataframe.to_parquet(
        temporary_path,
        index=False,
    )
    os.replace(
        temporary_path,
        path,
    )


def write_json(
    data,
    path: Path,
) -> None:
    """Write JSON while supporting NumPy-like scalar values."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w") as file:
        json.dump(
            data,
            file,
            indent=2,
            default=str,
        )