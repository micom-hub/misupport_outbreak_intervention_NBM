from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union, Literal, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TimeseriesKind = Literal["incidence", "prevalence"]


# ---------------------------
# File discovery + loading
# ---------------------------

def _find_file(run_dir: Union[str, Path], filename: str) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    direct = run_dir / filename
    if direct.exists():
        return direct

    # fallback: search recursively (in case you reorganize outputs)
    hits = list(run_dir.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} under {run_dir}")
    return hits[0]


def load_aggregated_timeseries(run_dir: Union[str, Path], kind: TimeseriesKind) -> Tuple[pd.DataFrame, Path]:
    fname = f"aggregated_{kind}.parquet"
    path = _find_file(run_dir, fname)
    return pd.read_parquet(path), path


def load_aggregated_summary(run_dir: Union[str, Path]) -> Tuple[pd.DataFrame, Path]:
    path = _find_file(run_dir, "aggregated_summary.parquet")
    return pd.read_parquet(path), path


def _detect_variant_col(df: pd.DataFrame, variant_col: Optional[str] = None) -> str:
    if variant_col and variant_col in df.columns:
        return variant_col
    for cand in ("variant_name", "policy_name", "variant", "policy"):
        if cand in df.columns:
            return cand
    raise ValueError("Could not infer variant/policy column. Pass variant_col explicitly.")


def _time_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith("t_")]
    def key(c: str) -> int:
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return 10**18
    return sorted(cols, key=key)


def _labels_for_kind(kind: TimeseriesKind) -> Dict[str, str]:
    if kind == "incidence":
        return {
            "x": "Day",
            "y": "Incidence (new exposures / N)",
            "title": "Incidence trajectories",
        }
    return {
        "x": "Day",
        "y": "Prevalence (I / N)",
        "title": "Prevalence trajectories",
    }


def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _savefig(fig: plt.Figure, out_path: Optional[Union[str, Path]]) -> None:
    if out_path is None:
        return
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")


# ---------------------------
# Timeseries summaries
# ---------------------------

def summarize_timeseries(
    df: pd.DataFrame,
    *,
    variant_col: str,
    tcols: Optional[Sequence[str]] = None,
    center: Literal["median", "mean"] = "median",
    band: Optional[Tuple[float, float]] = (0.25, 0.75),
) -> pd.DataFrame:
    """
    Returns a tidy summary table with columns:
      variant, time, center, lo, hi

    Summary is across all rows in df (i.e., across model_index and run_number).
    """
    if tcols is None:
        tcols = _time_cols(df)
    if not tcols:
        raise ValueError("No t_* columns found.")

    long = df.melt(
        id_vars=[variant_col],
        value_vars=list(tcols),
        var_name="_t",
        value_name="value",
    )
    long["time"] = long["_t"].str.split("_", n=1, expand=True)[1].astype(int)
    long.drop(columns=["_t"], inplace=True)

    g = long.groupby([variant_col, "time"])["value"]
    if center == "median":
        cen = g.median()
    else:
        cen = g.mean()

    out = cen.rename("center").reset_index()

    if band is not None:
        qlo, qhi = band
        lo = g.quantile(qlo).rename("lo").reset_index(drop=True)
        hi = g.quantile(qhi).rename("hi").reset_index(drop=True)
        out["lo"] = lo
        out["hi"] = hi
    else:
        out["lo"] = np.nan
        out["hi"] = np.nan

    out = out.rename(columns={variant_col: "variant"})
    return out


def summarize_timeseries_paired_delta_vs_baseline(
    df: pd.DataFrame,
    *,
    variant_col: str,
    baseline_variant: str,
    tcols: Optional[Sequence[str]] = None,
    replicate_agg: Literal["median", "mean"] = "median",
    across_models_center: Literal["median", "mean"] = "median",
    band: Optional[Tuple[float, float]] = (0.25, 0.75),
) -> pd.DataFrame:
    """
    Computes paired (by model_index) delta trajectories: variant - baseline_variant.

    Steps:
      1) aggregate stochastic replicates within each (model_index, variant)
      2) compute per-model_index delta vs baseline
      3) summarize delta across model_index (median/IQR or mean)

    Returns tidy table:
      variant, time, center, lo, hi
    """
    if "model_index" not in df.columns:
        raise ValueError("paired delta requires model_index column.")

    if tcols is None:
        tcols = _time_cols(df)
    if not tcols:
        raise ValueError("No t_* columns found.")

    # 1) aggregate replicates within (model_index, variant)
    grp = df.groupby(["model_index", variant_col], as_index=False)
    if replicate_agg == "median":
        agg = grp[list(tcols)].median()
    else:
        agg = grp[list(tcols)].mean()

    # 2) pivot baseline + each variant and compute deltas per model_index
    base = agg[agg[variant_col] == baseline_variant].set_index("model_index")[list(tcols)]
    if base.empty:
        raise ValueError(f"Baseline variant '{baseline_variant}' not found.")

    deltas = []
    for v in sorted(agg[variant_col].unique().tolist()):
        if v == baseline_variant:
            continue
        cur = agg[agg[variant_col] == v].set_index("model_index")[list(tcols)]
        joined = cur.join(base, how="inner", lsuffix="_v", rsuffix="_b")
        if joined.empty:
            continue
        dv = joined[[c + "_v" for c in tcols]].to_numpy(float) - joined[[c + "_b" for c in tcols]].to_numpy(float)
        dv_df = pd.DataFrame(dv, columns=list(tcols))
        dv_df.insert(0, "variant", v)
        deltas.append(dv_df)

    if not deltas:
        raise ValueError("No paired deltas computed (check model_index overlap across variants).")

    big = pd.concat(deltas, ignore_index=True)

    # 3) summarize across model_index (rows) for each variant/time
    long = big.melt(id_vars=["variant"], value_vars=list(tcols), var_name="_t", value_name="value")
    long["time"] = long["_t"].str.split("_", n=1, expand=True)[1].astype(int)
    long.drop(columns=["_t"], inplace=True)

    g = long.groupby(["variant", "time"])["value"]
    cen = g.median() if across_models_center == "median" else g.mean()
    out = cen.rename("center").reset_index()

    if band is not None:
        qlo, qhi = band
        out["lo"] = g.quantile(qlo).to_numpy()
        out["hi"] = g.quantile(qhi).to_numpy()
    else:
        out["lo"] = np.nan
        out["hi"] = np.nan

    return out


# ---------------------------
# Plotting: timeseries
# ---------------------------

def plot_timeseries_by_variant(
    run_dir: Union[str, Path],
    *,
    kind: TimeseriesKind,
    variant_col: Optional[str] = None,
    center: Literal["median", "mean"] = "median",
    band: Optional[Tuple[float, float]] = (0.25, 0.75),
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    df, path = load_aggregated_timeseries(run_dir, kind)
    vcol = _detect_variant_col(df, variant_col)
    tcols = _time_cols(df)

    summ = summarize_timeseries(df, variant_col=vcol, tcols=tcols, center=center, band=band)

    labels = _labels_for_kind(kind)
    fig, ax = plt.subplots(figsize=(11, 5))

    for v in sorted(summ["variant"].unique().tolist()):
        s = summ[summ["variant"] == v].sort_values("time")
        ax.plot(s["time"], s["center"], label=str(v), linewidth=2)
        if band is not None and s["lo"].notna().any():
            ax.fill_between(s["time"], s["lo"], s["hi"], alpha=0.2)

    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["y"])
    ax.set_title(title or f"{labels['title']} (source: {path.name})")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_timeseries_paired_delta_vs_baseline(
    run_dir: Union[str, Path],
    *,
    kind: TimeseriesKind,
    baseline_variant: str,
    variant_col: Optional[str] = None,
    replicate_agg: Literal["median", "mean"] = "median",
    across_models_center: Literal["median", "mean"] = "median",
    band: Optional[Tuple[float, float]] = (0.25, 0.75),
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    df, path = load_aggregated_timeseries(run_dir, kind)
    vcol = _detect_variant_col(df, variant_col)
    tcols = _time_cols(df)

    summ = summarize_timeseries_paired_delta_vs_baseline(
        df,
        variant_col=vcol,
        baseline_variant=baseline_variant,
        tcols=tcols,
        replicate_agg=replicate_agg,
        across_models_center=across_models_center,
        band=band,
    )

    labels = _labels_for_kind(kind)
    fig, ax = plt.subplots(figsize=(11, 5))

    for v in sorted(summ["variant"].unique().tolist()):
        s = summ[summ["variant"] == v].sort_values("time")
        ax.plot(s["time"], s["center"], label=f"{v} - {baseline_variant}", linewidth=2)
        if band is not None and s["lo"].notna().any():
            ax.fill_between(s["time"], s["lo"], s["hi"], alpha=0.2)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(f"Δ {labels['y']} (paired by model_index)")
    ax.set_title(title or f"Paired delta vs baseline (source: {path.name})")
    ax.legend(ncol=1, fontsize=9)
    ax.grid(True, alpha=0.25)

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------
# Plotting: summary metrics
# ---------------------------

def plot_summary_metrics_boxplots(
    run_dir: Union[str, Path],
    *,
    metrics: Sequence[str],
    variant_col: Optional[str] = None,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    df, path = load_aggregated_summary(run_dir)
    vcol = _detect_variant_col(df, variant_col)

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Missing summary metric columns: {missing}. Available: {list(df.columns)}")

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)
    axes = axes[0]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        # simple boxplot using matplotlib (no seaborn dependency)
        groups = [g[metric].dropna().to_numpy() for _, g in df.groupby(vcol, sort=True)]
        labels = [str(k) for k, _ in df.groupby(vcol, sort=True)]

        ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_title(metric)
        ax.set_xlabel("Variant")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(title or f"Summary metrics (source: {path.name})")
    fig.tight_layout()

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_summary_metric_paired_differences(
    run_dir: Union[str, Path],
    *,
    metric: str,
    baseline_variant: str,
    variant_col: Optional[str] = None,
    replicate_agg: Literal["median", "mean"] = "median",
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Paired differences (by model_index): metric(variant) - metric(baseline).
    Aggregates stochastic replicates per model_index, variant before differencing.
    """
    df, path = load_aggregated_summary(run_dir)
    vcol = _detect_variant_col(df, variant_col)

    if "model_index" not in df.columns:
        raise ValueError("paired differences require model_index column.")
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")

    grp = df.groupby(["model_index", vcol], as_index=False)
    agg = grp[metric].median() if replicate_agg == "median" else grp[metric].mean()

    base = agg[agg[vcol] == baseline_variant].set_index("model_index")[[metric]]
    if base.empty:
        raise ValueError(f"Baseline variant '{baseline_variant}' not found.")

    deltas = []
    for v in sorted(agg[vcol].unique().tolist()):
        if v == baseline_variant:
            continue
        cur = agg[agg[vcol] == v].set_index("model_index")[[metric]]
        joined = cur.join(base, how="inner", lsuffix="_v", rsuffix="_b")
        if joined.empty:
            continue
        d = joined[f"{metric}_v"] - joined[f"{metric}_b"]
        deltas.append(pd.DataFrame({"variant": v, "delta": d.to_numpy()}))

    if not deltas:
        raise ValueError("No paired deltas computed (check model_index overlap across variants).")

    dd = pd.concat(deltas, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    groups = [g["delta"].to_numpy() for _, g in dd.groupby("variant", sort=True)]
    labels = [str(k) for k, _ in dd.groupby("variant", sort=True)]

    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Variant")
    ax.set_ylabel(f"Δ {metric} (variant - {baseline_variant})")
    ax.set_title(title or f"Paired Δ {metric} vs baseline (source: {path.name})")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------
# Convenience runner
# ---------------------------

def make_run_visualizations(
    run_dir: Union[str, Path],
    *,
    summary_metrics: Sequence[str] = ("peakPrev", "peakTime", "outbreakSize"),
    baseline_variant: Optional[str] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """
    Creates a standard set of plots under <run_dir>/results/trajectory_plots by default.
    Returns a dict of named output paths.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if out_dir is None:
        out_dir = run_dir / "results" / "trajectory_plots"
    out_dir = _ensure_dir(out_dir)

    outputs: Dict[str, Path] = {}

    # summary
    p = out_dir / "summary_boxplots.png"
    plot_summary_metrics_boxplots(run_dir, metrics=summary_metrics, out_path=p)
    outputs["summary_boxplots"] = p

    # incidence/prevalence trajectories
    for kind in ("incidence", "prevalence"):
        p = out_dir / f"{kind}_trajectories.png"
        plot_timeseries_by_variant(run_dir, kind=kind, out_path=p)
        outputs[f"{kind}_trajectories"] = p

        if baseline_variant is not None:
            p2 = out_dir / f"{kind}_paired_delta_vs_{baseline_variant}.png"
            plot_timeseries_paired_delta_vs_baseline(run_dir, kind=kind, baseline_variant=baseline_variant, out_path=p2)
            outputs[f"{kind}_paired_delta"] = p2

    # paired summary deltas
    if baseline_variant is not None:
        for m in summary_metrics:
            p = out_dir / f"paired_delta_{m}_vs_{baseline_variant}.png"
            plot_summary_metric_paired_differences(run_dir, metric=m, baseline_variant=baseline_variant, out_path=p)
            outputs[f"paired_delta_{m}"] = p

    return outputs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default=None, help="baseline variant/policy name for paired delta plots")
    ap.add_argument("--metrics", nargs="*", default=["peakPrev", "peakTime", "outbreakSize"])
    args = ap.parse_args()

    outs = make_run_visualizations(args.run_dir, summary_metrics=args.metrics, baseline_variant=args.baseline)
    print("[run_trajectories] wrote:")
    for k, v in outs.items():
        print(" ", k, "->", v)