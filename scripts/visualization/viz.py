# scripts/analysis/variant_plots.py
from __future__ import annotations
from math import ceil
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.analysis.variant_analysis import (
    ensure_variant_column,
    trajectory_quantiles_by_variant,
    paired_difference_quantiles
)

#Plot distribution of summary metrics by variant
def plot_metric_boxplot(
    summary_df: pd.DataFrame,
    metric: str,
    *,
    variant_col: str = "variant",
    figsize: Tuple[float, float] = (10, 5),
    showfliers: bool = False,
    order: Optional[Sequence[object]] = None,
    ax=None,
):
    """
    Boxplot of `metric` by `variant_col`.
    Returns (fig, ax).
    """
    df = ensure_variant_column(summary_df, variant_col=variant_col)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in dataframe columns")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if order is None:
        variants = list(df[variant_col].dropna().unique())
    else:
        variants = list(order)

    data = [df.loc[df[variant_col] == v, metric].dropna().to_numpy() for v in variants]
    ax.boxplot(data, labels=[str(v) for v in variants], showfliers=showfliers)

    ax.set_title(f"{metric} by variant")
    ax.set_xlabel("variant")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax

#Plot trajectory by variant
def plot_trajectories_by_variant(
    ts_df: pd.DataFrame,
    *,
    variant_col: str = "variant",
    variants: Optional[Sequence[object]] = None,
    time_prefix: str = "t_",
    qs: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5.0, 3.5),
):
    """
    One panel per variant. Median = red line; 50% ribbon = dark blue; 90% ribbon = light blue.
    Returns (fig, axes).
    """
    qlist = trajectory_quantiles_by_variant(
        ts_df, variant_col=variant_col, time_prefix=time_prefix, qs=qs, variants=variants
    )
    if not qlist:
        raise ValueError("No variants to plot (check variants filter / variant column).")

    # find indices of key quantiles (must exist in qs)
    qs = tuple(float(q) for q in qs)
    def qidx(q): 
        if q not in qs:
            raise ValueError(f"Required quantile {q} missing from qs={qs}")
        return qs.index(q)

    i05, i25, i50, i75, i95 = qidx(0.05), qidx(0.25), qidx(0.50), qidx(0.75), qidx(0.95)

    n = len(qlist)
    ncols = max(1, int(ncols))
    nrows = ceil(n / ncols)
    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, tq in zip(axes, qlist):
        t = tq.t
        qv = tq.values

        ax.fill_between(t, qv[i05], qv[i95], color="#9ecae1", alpha=0.6, linewidth=0)  # light blue (90%)
        ax.fill_between(t, qv[i25], qv[i75], color="#08519c", alpha=0.5, linewidth=0)  # dark blue (50%)
        ax.plot(t, qv[i50], color="red", linewidth=2)

        ax.set_title(str(tq.variant))
        ax.grid(alpha=0.25)

    # blank unused axes
    for ax in axes[len(qlist):]:
        ax.axis("off")

    fig.suptitle("Outbreak trajectory by variant (median + 50%/90% bands)", y=1.02)
    fig.tight_layout()
    return fig, axes

#Calculate the pairwise difference between two variants
def plot_paired_difference_trajectory(
    ts_df: pd.DataFrame,
    variant_a: object,
    variant_b: object,
    *,
    variant_col: str = "variant",
    pair_on: str = "run_number",
    time_prefix: str = "t_",
    qs: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
    figsize: Tuple[float, float] = (10, 4),
    ax=None,
):
    """
    Plot paired (A - B) difference trajectory over time with median and 50%/90% ribbons.
    Returns (fig, ax).
    """
    tq = paired_difference_quantiles(
        ts_df,
        variant_a,
        variant_b,
        variant_col=variant_col,
        pair_on=pair_on,
        time_prefix=time_prefix,
        qs=qs,
    )

    qs = tuple(float(q) for q in qs)
    def qidx(q): 
        if q not in qs:
            raise ValueError(f"Required quantile {q} missing from qs={qs}")
        return qs.index(q)

    i05, i25, i50, i75, i95 = qidx(0.05), qidx(0.25), qidx(0.50), qidx(0.75), qidx(0.95)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    t = tq.t
    qv = tq.values

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.fill_between(t, qv[i05], qv[i95], color="#9ecae1", alpha=0.6, linewidth=0)
    ax.fill_between(t, qv[i25], qv[i75], color="#08519c", alpha=0.5, linewidth=0)
    ax.plot(t, qv[i50], color="red", linewidth=2)

    ax.set_title(f"Paired difference trajectory: {variant_a} - {variant_b}")
    ax.set_xlabel("time")
    ax.set_ylabel("difference")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax