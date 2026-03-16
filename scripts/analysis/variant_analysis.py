# scripts/analysis/variant_analysis.py
#Common helper functions for analytics, used in visualization pipelines
from __future__ import annotations
from dataclasses import dataclass
from typing import  Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def ensure_variant_column(
    df: pd.DataFrame,
    variant_col: str = "variant",
    fallback_col: str = "model_index",
) -> pd.DataFrame:
    """
    Ensure df has a `variant_col`. If missing, create it from `fallback_col`.
    This keeps plotting functions robust even before you standardize metadata.
    """
    if variant_col in df.columns:
        return df
    if fallback_col not in df.columns:
        raise ValueError(f"Missing '{variant_col}' and fallback '{fallback_col}' in df.columns")
    out = df.copy()
    out[variant_col] = out[fallback_col]
    return out


def time_columns(df: pd.DataFrame, prefix: str = "t_") -> list[str]:
    """
    Return time columns sorted by integer suffix: t_0, t_1, ...
    """
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    def key(c: str) -> int:
        try:
            return int(c[len(prefix):])
        except Exception:
            return 10**18
    return sorted(cols, key=key)


@dataclass(frozen=True)
class TrajectoryQuantiles:
    """
    Quantiles for a single variant. q values correspond to `qs` in the order supplied.
    """
    variant: object
    t: np.ndarray              # shape (T,)
    qs: Tuple[float, ...]      # e.g. (0.05,0.25,0.5,0.75,0.95)
    values: np.ndarray         # shape (len(qs), T)


def trajectory_quantiles_by_variant(
    ts_df: pd.DataFrame,
    *,
    variant_col: str = "variant",
    time_prefix: str = "t_",
    qs: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
    variants: Optional[Sequence[object]] = None,
) -> list[TrajectoryQuantiles]:
    """
    Compute quantile trajectories per variant from a wide timeseries dataframe.
    Returns a list of TrajectoryQuantiles (one per variant).
    """
    ts_df = ensure_variant_column(ts_df, variant_col=variant_col)

    tcols = time_columns(ts_df, prefix=time_prefix)
    if not tcols:
        raise ValueError(f"No time columns found with prefix '{time_prefix}'")

    qs = tuple(float(q) for q in qs)
    t = np.array([int(c[len(time_prefix):]) for c in tcols], dtype=np.int32)

    if variants is None:
        groups = ts_df.groupby(variant_col, sort=False)
    else:
        wanted = set(variants)
        groups = ((v, g) for v, g in ts_df.groupby(variant_col, sort=False) if v in wanted)

    out: list[TrajectoryQuantiles] = []
    for v, g in groups:
        arr = g[tcols].to_numpy(dtype=np.float32, copy=False)  # shape (n_runs, T)
        if arr.shape[0] == 0:
            continue
        qvals = np.quantile(arr, qs, axis=0)  # shape (len(qs), T)
        out.append(TrajectoryQuantiles(variant=v, t=t, qs=qs, values=qvals.astype(np.float32, copy=False)))
    return out


def paired_difference_quantiles(
    ts_df: pd.DataFrame,
    variant_a: object,
    variant_b: object,
    *,
    variant_col: str = "variant",
    pair_on: str = "run_number",
    time_prefix: str = "t_",
    qs: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> TrajectoryQuantiles:
    """
    Pair runs by `pair_on` (default run_number) and compute (A - B) trajectory,
    then quantiles across paired runs.
    """
    ts_df = ensure_variant_column(ts_df, variant_col=variant_col)

    tcols = time_columns(ts_df, prefix=time_prefix)
    if not tcols:
        raise ValueError(f"No time columns found with prefix '{time_prefix}'")

    need_cols = {variant_col, pair_on, *tcols}
    missing = [c for c in need_cols if c not in ts_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    A = ts_df.loc[ts_df[variant_col] == variant_a, [pair_on] + tcols].copy()
    B = ts_df.loc[ts_df[variant_col] == variant_b, [pair_on] + tcols].copy()
    if A.empty or B.empty:
        raise ValueError(f"One or both variants not found in data: {variant_a}, {variant_b}")

    # inner-join to ensure pairing is well-defined
    M = A.merge(B, on=pair_on, how="inner", suffixes=("_a", "_b"))
    if M.empty:
        raise ValueError(f"No paired runs found between variants on '{pair_on}'")

    a_cols = [c + "_a" for c in tcols]
    b_cols = [c + "_b" for c in tcols]
    diff = M[a_cols].to_numpy(dtype=np.float32, copy=False) - M[b_cols].to_numpy(dtype=np.float32, copy=False)

    qs = tuple(float(q) for q in qs)
    qvals = np.quantile(diff, qs, axis=0)

    t = np.array([int(c[len(time_prefix):]) for c in tcols], dtype=np.int32)
    label = f"{variant_a} - {variant_b}"
    return TrajectoryQuantiles(variant=label, t=t, qs=qs, values=qvals.astype(np.float32, copy=False))