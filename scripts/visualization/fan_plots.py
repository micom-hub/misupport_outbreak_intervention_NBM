from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union, Literal, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TimeseriesKind = Literal["incidence", "prevalence"]


# ----------------------------
# File discovery + column utils
# ----------------------------

def _find_file(run_dir: Union[str, Path], filename: str) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    direct = run_dir / filename
    if direct.exists():
        return direct
    hits = list(run_dir.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} under {run_dir}")
    return hits[0]


def _detect_policy_col(df: pd.DataFrame, policy_col: Optional[str] = None) -> str:
    if policy_col and policy_col in df.columns:
        return policy_col
    for cand in ("variant_name", "policy_name", "variant", "policy"):
        if cand in df.columns:
            return cand
    raise ValueError(
        "Could not infer policy column. Expected one of: "
        "variant_name, policy_name, variant, policy. Pass policy_col explicitly."
    )


def _time_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith("t_")]
    def key(c: str) -> int:
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return 10**18
    return sorted(cols, key=key)


def _labels(kind: TimeseriesKind) -> Tuple[str, str]:
    if kind == "incidence":
        return ("Day", "Incidence (new exposures / N)")
    return ("Day", "Prevalence (I / N)")


def _as_list(x: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(x, str):
        return [x]
    return [str(v) for v in x]


# ----------------------------
# Main plotting function
# ----------------------------

def plot_policy_fan(
    run_dir: Union[str, Path],
    *,
    kind: TimeseriesKind,
    policy: Union[str, Sequence[str]],
    policy_col: Optional[str] = None,
    t_max: Optional[int] = None,
    replicate_agg: Literal["none", "mean", "median"] = "none",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Fan plot (median + 50% band + 90% band) for incidence or prevalence trajectories.

    Args:
      run_dir: model run directory containing aggregated_{kind}.parquet
      kind: "incidence" or "prevalence"
      policy: policy name or list of policies
      policy_col: optional override for policy column (defaults to variant_name/policy_name/etc)
      t_max: optional max day to plot (inclusive). If None, plots all available t_* columns.
      replicate_agg:
        - "none": quantiles computed over all rows (all model_index x run_number)
        - "mean"/"median": first aggregate within each (model_index, policy), then compute quantiles across model_index
          (this avoids overweighting stochastic replicates)
      figsize: optional (width, height)
      out_path: optional save path
      show: if True, show the plot window (otherwise closes after saving)

    Style:
      median: red line
      50% band (25–75): dark blue fill
      90% band (5–95): light blue fill
    """
    kind = str(kind).strip().lower()  # type: ignore[assignment]
    if kind not in ("incidence", "prevalence"):
        raise ValueError("kind must be 'incidence' or 'prevalence'")

    policies = _as_list(policy)

    parquet_name = f"aggregated_{kind}.parquet"
    path = _find_file(run_dir, parquet_name)
    df = pd.read_parquet(path)

    pol_col = _detect_policy_col(df, policy_col)
    tcols = _time_cols(df)
    if not tcols:
        raise ValueError(f"No t_* columns found in {path}")

    # Optional truncate time
    if t_max is not None:
        tcols = [c for c in tcols if int(c.split("_", 1)[1]) <= int(t_max)]
        if not tcols:
            raise ValueError(f"t_max={t_max} removed all time columns.")

    # Filter policies
    present = set(df[pol_col].astype(str).unique().tolist())
    missing = [p for p in policies if p not in present]
    if missing:
        raise ValueError(f"Policies not found in {parquet_name}: {missing}. Present: {sorted(present)[:30]} ...")

    df = df[df[pol_col].astype(str).isin(policies)].copy()

    # Optionally aggregate replicates within each model_index/policy
    if replicate_agg != "none":
        if "model_index" not in df.columns:
            raise ValueError("replicate_agg != 'none' requires 'model_index' column in the parquet.")
        grp = df.groupby(["model_index", pol_col], as_index=False)
        if replicate_agg == "median":
            df = grp[list(tcols)].median()
        else:
            df = grp[list(tcols)].mean()

    # Set up faceting
    n = len(policies)
    if figsize is None:
        figsize = (max(6.0, 5.0 * n), 4.8)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True, sharex=True)
    if n == 1:
        axes = [axes]

    # Colors (requested)
    c_median = "#e31a1c"   # red
    c_50 = "#1f78b4"       # dark blue
    c_90 = "#a6cee3"       # light blue

    # Precompute global y-limits for shared y-axis
    global_lo = np.inf
    global_hi = -np.inf

    quantiles_by_policy = {}

    x = np.array([int(c.split("_", 1)[1]) for c in tcols], dtype=int)

    for p in policies:
        sub = df[df[pol_col].astype(str) == p]
        arr = sub[tcols].to_numpy(dtype=float, copy=False)
        if arr.size == 0:
            continue

        # robust to any NaNs
        q05, q25, q50, q75, q95 = np.nanquantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)

        quantiles_by_policy[p] = (q05, q25, q50, q75, q95)
        global_lo = float(min(global_lo, np.nanmin(q05)))
        global_hi = float(max(global_hi, np.nanmax(q95)))

    # enforce non-negative scale (incidence/prevalence are fractions)
    if not np.isfinite(global_lo):
        global_lo = 0.0
    if not np.isfinite(global_hi):
        global_hi = 1.0
    global_lo = min(0.0, global_lo)
    pad = 0.05 * (global_hi - global_lo) if global_hi > global_lo else 0.05
    y0 = max(0.0, global_lo - pad)
    y1 = global_hi + pad

    xlabel, ylabel = _labels(kind)  # type: ignore[arg-type]

    for ax, p in zip(axes, policies):
        if p not in quantiles_by_policy:
            ax.set_title(str(p))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        q05, q25, q50, q75, q95 = quantiles_by_policy[p]

        # 90% band (light blue)
        ax.fill_between(x, q05, q95, color=c_90, alpha=0.6, linewidth=0)

        # 50% band (dark blue)
        ax.fill_between(x, q25, q75, color=c_50, alpha=0.55, linewidth=0)

        # median line (red)
        ax.plot(x, q50, color=c_median, linewidth=2.2)

        ax.set_title(str(p))
        ax.set_ylim(y0, y1)
        ax.grid(True, alpha=0.25)

    # shared labels
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    if title is None:
        title = f"{kind.capitalize()} fan plot"
    fig.suptitle(title, y=1.02)

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


if __name__ == "__main__":
    plot_policy_fan(
    "model_runs/SecondSensitivityAnalysis",
    kind="incidence",
    policy=["observe_only", "trace_then_isolate"],
    out_path="model_runs/SecondSensitivityAnalysis/results/fan_incidence_observe_only.png",
)

plot_policy_fan(
    "model_runs/SecondSensitivityAnalysis",
    kind="prevalence",
    policy=["observe_only", "trace_then_isolate"],
    replicate_agg="median",  # optional: avoid overweighting stochastic reps
        out_path="model_runs/SecondSensitivityAnalysis/results/fan_prevalence_faceted.png",
)