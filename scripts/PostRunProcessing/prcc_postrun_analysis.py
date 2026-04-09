from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union, Literal, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


Method = Literal["uncorrected", "ztest", "bonferroni", "bhfdr"]


# -------------------------
# Config + loading
# -------------------------

@dataclass(frozen=True)
class PrccPaths:
    run_dir: Path
    prcc_long: Path
    out_dir: Path

    @classmethod
    def from_run_dir(cls, run_dir: Union[str, Path]) -> "PrccPaths":
        run_dir = Path(run_dir).expanduser().resolve()
        prcc_long = run_dir / "results" / "prcc_results_long.parquet"
        out_dir = run_dir / "results" / "prcc_analysis"
        return cls(run_dir=run_dir, prcc_long=prcc_long, out_dir=out_dir)


def load_prcc_long(run_dir: Union[str, Path], *, parquet_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    paths = PrccPaths.from_run_dir(run_dir)
    p = Path(parquet_path).expanduser().resolve() if parquet_path else paths.prcc_long
    if not p.exists():
        raise FileNotFoundError(f"PRCC long parquet not found: {p}")
    df = pd.read_parquet(p)
    # normalize types
    if "significant" in df.columns:
        df["significant"] = df["significant"].astype(bool)
    if "timepoint" in df.columns:
        # keep numeric; convert float days to int if they are actually integers
        tp = pd.to_numeric(df["timepoint"], errors="coerce")
        df["timepoint"] = tp
    return df


def subset_prcc(
    df: pd.DataFrame,
    *,
    kind: Optional[str] = None,
    method: Optional[Method] = None,
    policies: Optional[Sequence[str]] = None,
    outputs: Optional[Sequence[str]] = None,
    parameters: Optional[Sequence[str]] = None,
    timepoints: Optional[Sequence[Union[int, float]]] = None,
    significant_only: bool = False,
) -> pd.DataFrame:
    x = df.copy()
    if kind is not None:
        x = x[x["kind"] == kind]
    if method is not None:
        x = x[x["method"] == method]
    if policies is not None:
        x = x[x["policy"].isin(list(policies))]
    if outputs is not None:
        x = x[x["output"].isin(list(outputs))]
    if parameters is not None:
        x = x[x["parameter"].isin(list(parameters))]
    if timepoints is not None:
        x = x[x["timepoint"].isin(list(timepoints))]
    if significant_only:
        x = x[x["significant"] == True]  # noqa: E712
    return x


# -------------------------
# Ranking / tables
# -------------------------

def top_k_parameters(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    k: int = 15,
    require_significant: bool = True,
    score: Literal["abs_prcc", "p_value"] = "abs_prcc",
) -> pd.DataFrame:
    """
    Returns a long table with top-k parameters per group.
    score:
      - abs_prcc: sort by |prcc| desc (common)
      - p_value:  sort by p_value asc
    """
    x = df.copy()
    if require_significant:
        x = x[x["significant"] == True]  # noqa: E712

    x["abs_prcc"] = x["prcc"].abs()

    if score == "abs_prcc":
        x = x.sort_values(list(group_cols) + ["abs_prcc"], ascending=[True]*len(group_cols) + [False])
    else:
        x = x.sort_values(list(group_cols) + ["p_value"], ascending=[True]*len(group_cols) + [True])

    out = (
        x.groupby(list(group_cols), as_index=False, sort=False)
        .head(int(k))
        .reset_index(drop=True)
    )
    return out


# -------------------------
# Plot helpers
# -------------------------

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _order_parameters_by_max_abs(sub: pd.DataFrame) -> list[str]:
    s = sub.groupby("parameter")["prcc"].apply(lambda v: float(np.nanmax(np.abs(v.to_numpy()))))
    return s.sort_values(ascending=False).index.tolist()


def _savefig(path: Optional[Union[str, Path]]) -> None:
    if path is None:
        return
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")


# -------------------------
# Heatmaps
# -------------------------

def plot_timeseries_heatmap(
    df: pd.DataFrame,
    *,
    kind: Literal["incidence", "prevalence"],
    policy: str,
    output: str = "output",
    method: Method = "bhfdr",
    significant_only: bool = False,
    top_n_params: int = 30,
    timepoints: Optional[Sequence[Union[int, float]]] = None,
    cmap: str = "coolwarm",
    vlim: float = 1.0,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> None:
    sub = subset_prcc(
        df,
        kind=kind,
        method=method,
        policies=[policy],
        outputs=[output],
        timepoints=timepoints,
        significant_only=significant_only,
    )
    if sub.empty:
        raise ValueError(f"No rows for kind={kind}, policy={policy}, output={output}, method={method}")

    # reduce parameters for readability
    ordered = _order_parameters_by_max_abs(sub)
    keep = ordered[: int(top_n_params)]
    sub = sub[sub["parameter"].isin(keep)]

    # pivot to (param x time)
    mat = sub.pivot_table(index="parameter", columns="timepoint", values="prcc", aggfunc="mean")
    mat = mat.loc[keep]  # keep order
    mat = mat.sort_index(axis=1)  # time ascending

    plt.figure(figsize=(min(18, 0.22 * mat.shape[1] + 4), min(12, 0.28 * mat.shape[0] + 3)))
    ax = sns.heatmap(
        mat,
        cmap=cmap,
        center=0.0,
        vmin=-float(vlim),
        vmax=float(vlim),
        cbar_kws={"label": "PRCC"},
    )
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Parameter")
    ax.set_title(title or f"{kind} PRCC heatmap | policy={policy} | method={method} | output={output}")
    plt.tight_layout()
    _savefig(out_path)
    plt.close()


def plot_summary_heatmap(
    df: pd.DataFrame,
    *,
    policy: str,
    method: Method = "bhfdr",
    significant_only: bool = False,
    top_n_params: int = 40,
    cmap: str = "coolwarm",
    vlim: float = 1.0,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> None:
    sub = subset_prcc(
        df,
        kind="summary",
        method=method,
        policies=[policy],
        significant_only=significant_only,
    )
    if sub.empty:
        raise ValueError(f"No rows for summary policy={policy} method={method}")

    ordered = _order_parameters_by_max_abs(sub)
    keep = ordered[: int(top_n_params)]
    sub = sub[sub["parameter"].isin(keep)]

    mat = sub.pivot_table(index="parameter", columns="output", values="prcc", aggfunc="mean")
    mat = mat.loc[keep]

    plt.figure(figsize=(min(14, 0.6 * mat.shape[1] + 4), min(14, 0.28 * mat.shape[0] + 3)))
    ax = sns.heatmap(
        mat,
        cmap=cmap,
        center=0.0,
        vmin=-float(vlim),
        vmax=float(vlim),
        cbar_kws={"label": "PRCC"},
        linewidths=0.2,
        linecolor="white",
    )
    ax.set_xlabel("Output metric")
    ax.set_ylabel("Parameter")
    ax.set_title(title or f"summary PRCC heatmap | policy={policy} | method={method}")
    plt.tight_layout()
    _savefig(out_path)
    plt.close()


# -------------------------
# Lollipop plots
# -------------------------

def plot_lollipop(
    df: pd.DataFrame,
    *,
    kind: str,
    policy: str,
    output: str,
    timepoint: Union[int, float] = 0,
    method: Method = "bhfdr",
    k: int = 20,
    significant_only: bool = True,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> None:
    sub = subset_prcc(
        df,
        kind=kind,
        method=method,
        policies=[policy],
        outputs=[output],
        timepoints=[timepoint],
        significant_only=significant_only,
    )
    if sub.empty:
        raise ValueError("No rows for requested lollipop plot.")

    sub = sub.assign(abs_prcc=sub["prcc"].abs()).sort_values("abs_prcc", ascending=False).head(int(k))
    sub = sub.sort_values("prcc")  # for nicer ordering

    y = np.arange(len(sub))
    vals = sub["prcc"].to_numpy()
    labels = sub["parameter"].to_numpy()

    plt.figure(figsize=(10, max(4, 0.35 * len(sub) + 2)))
    plt.hlines(y=y, xmin=0, xmax=vals, color="gray", linewidth=1)
    plt.scatter(vals, y, c=np.where(sub["significant"], "tab:red", "tab:blue"), s=35)
    plt.axvline(0, color="black", linewidth=1)

    plt.yticks(y, labels)
    plt.xlabel("PRCC")
    plt.title(title or f"Top {k} parameters | {kind} | policy={policy} | output={output} | t={timepoint} | {method}")
    plt.tight_layout()
    _savefig(out_path)
    plt.close()


# -------------------------
# Policy comparisons (PRCC)
# -------------------------

def plot_policy_difference_heatmap(
    df: pd.DataFrame,
    *,
    kind: Literal["incidence", "prevalence"],
    output: str,
    policy_a: str,
    policy_b: str,
    method: Method = "bhfdr",
    top_n_params: int = 30,
    cmap: str = "coolwarm",
    vlim: float = 1.0,
    out_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Heatmap of ΔPRCC = PRCC(policy_a) - PRCC(policy_b) across time for top parameters.
    """
    sub = subset_prcc(df, kind=kind, method=method, policies=[policy_a, policy_b], outputs=[output])
    if sub.empty:
        raise ValueError("No rows for policy difference heatmap request.")

    wide = sub.pivot_table(
        index=["parameter", "timepoint"],
        columns="policy",
        values="prcc",
        aggfunc="mean",
    )

    if policy_a not in wide.columns or policy_b not in wide.columns:
        raise ValueError("One of the policies is missing data for this output/kind/method.")

    delta = (wide[policy_a] - wide[policy_b]).reset_index(name="delta_prcc")

    # choose top params by max |delta|
    score = delta.groupby("parameter")["delta_prcc"].apply(lambda v: float(np.nanmax(np.abs(v.to_numpy()))))
    keep = score.sort_values(ascending=False).head(int(top_n_params)).index.tolist()

    mat = delta[delta["parameter"].isin(keep)].pivot_table(
        index="parameter", columns="timepoint", values="delta_prcc", aggfunc="mean"
    )
    mat = mat.loc[keep].sort_index(axis=1)

    plt.figure(figsize=(min(18, 0.22 * mat.shape[1] + 4), min(12, 0.28 * mat.shape[0] + 3)))
    ax = sns.heatmap(
        mat,
        cmap=cmap,
        center=0.0,
        vmin=-float(vlim),
        vmax=float(vlim),
        cbar_kws={"label": f"ΔPRCC ({policy_a} - {policy_b})"},
    )
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Parameter")
    ax.set_title(f"{kind} ΔPRCC heatmap | {output} | {method}")
    plt.tight_layout()
    _savefig(out_path)
    plt.close()


def plot_policy_compare_lollipop(
    df: pd.DataFrame,
    *,
    kind: str,
    output: str,
    timepoint: Union[int, float],
    policy_a: str,
    policy_b: str,
    method: Method = "bhfdr",
    k: int = 20,
    out_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Lollipop comparing PRCC(policy_a) and PRCC(policy_b) at a single timepoint:
    shows top-k parameters by |ΔPRCC|.
    """
    sub = subset_prcc(df, kind=kind, method=method, policies=[policy_a, policy_b], outputs=[output], timepoints=[timepoint])
    wide = sub.pivot_table(index="parameter", columns="policy", values="prcc", aggfunc="mean")

    if policy_a not in wide.columns or policy_b not in wide.columns:
        raise ValueError("Missing one of the policies for this selection.")

    wide["delta"] = wide[policy_a] - wide[policy_b]
    top = wide.reindex(wide["delta"].abs().sort_values(ascending=False).head(int(k)).index).copy()
    top = top.sort_values("delta")  # for plotting order

    y = np.arange(top.shape[0])
    plt.figure(figsize=(11, max(4, 0.35 * len(top) + 2)))

    # baseline points
    plt.scatter(top[policy_b].to_numpy(), y, label=policy_b, s=35, color="tab:blue")
    plt.scatter(top[policy_a].to_numpy(), y, label=policy_a, s=35, color="tab:orange")
    # connectors
    for i, (_, row) in enumerate(top.iterrows()):
        plt.plot([row[policy_b], row[policy_a]], [i, i], color="gray", linewidth=1)

    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y, top.index.tolist())
    plt.xlabel("PRCC")
    plt.title(f"PRCC comparison | {kind} | {output} | t={timepoint} | {method}")
    plt.legend()
    plt.tight_layout()
    _savefig(out_path)
    plt.close()


# -------------------------
# Batch report generator
# -------------------------

def make_prcc_report(
    run_dir: Union[str, Path],
    *,
    parquet_path: Optional[Union[str, Path]] = None,
    method: Method = "bhfdr",
    significant_only: bool = False,
    top_k: int = 15,
    top_n_params_heatmap: int = 30,
    baseline_policy: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Generates a basic set of plots/tables under <run_dir>/results/prcc_analysis/.
    Returns paths of key outputs.
    """
    paths = PrccPaths.from_run_dir(run_dir)
    out_dir = _ensure_dir(paths.out_dir)

    df = load_prcc_long(paths.run_dir, parquet_path=parquet_path)

    # Identify policies, outputs, kinds
    policies = sorted(df["policy"].dropna().unique().tolist())
    kinds = sorted(df["kind"].dropna().unique().tolist())

    written: Dict[str, Path] = {}

    # Top-k tables (by abs PRCC) for all combinations
    tbl = top_k_parameters(
        subset_prcc(df, method=method, significant_only=significant_only),
        group_cols=["kind", "policy", "output", "timepoint"],
        k=top_k,
        require_significant=significant_only,
        score="abs_prcc",
    )
    top_csv = out_dir / f"top_{top_k}_params__method-{method}__sigonly-{int(significant_only)}.csv"
    tbl.to_csv(top_csv, index=False)
    written["top_table_csv"] = top_csv

    # Summary heatmaps (per policy)
    if "summary" in kinds:
        for pol in policies:
            out = out_dir / "heatmaps" / "summary" / f"heatmap__summary__{pol}__{method}.png"
            plot_summary_heatmap(
                df,
                policy=pol,
                method=method,
                significant_only=significant_only,
                top_n_params=top_n_params_heatmap,
                out_path=out,
            )

    # Timeseries heatmaps + example lollipops
    for kind in ("incidence", "prevalence"):
        if kind not in kinds:
            continue
        # most files will use output name "output"
        outputs = sorted(df.loc[df["kind"] == kind, "output"].dropna().unique().tolist())
        for pol in policies:
            for outname in outputs:
                out = out_dir / "heatmaps" / kind / f"heatmap__{kind}__{pol}__{outname}__{method}.png"
                plot_timeseries_heatmap(
                    df,
                    kind=kind,
                    policy=pol,
                    output=outname,
                    method=method,
                    significant_only=significant_only,
                    top_n_params=top_n_params_heatmap,
                    out_path=out,
                )

                # lollipop at a few key timepoints (start/mid/end)
                sub = subset_prcc(df, kind=kind, method=method, policies=[pol], outputs=[outname])
                if sub.empty:
                    continue
                tps = sorted(sub["timepoint"].dropna().unique().tolist())
                if not tps:
                    continue
                for tp in [tps[0], tps[len(tps)//2], tps[-1]]:
                    lop = out_dir / "lollipops" / kind / f"lollipop__{kind}__{pol}__{outname}__t{tp}__{method}.png"
                    try:
                        plot_lollipop(
                            df,
                            kind=kind,
                            policy=pol,
                            output=outname,
                            timepoint=tp,
                            method=method,
                            k=min(20, top_k),
                            significant_only=True,
                            out_path=lop,
                        )
                    except Exception:
                        # if nothing significant at that tp, skip
                        pass

    # Policy comparisons vs baseline (if requested)
    if baseline_policy is not None and baseline_policy in policies:
        for kind in ("incidence", "prevalence"):
            if kind not in kinds:
                continue
            outputs = sorted(df.loc[df["kind"] == kind, "output"].dropna().unique().tolist())
            for pol in policies:
                if pol == baseline_policy:
                    continue
                for outname in outputs:
                    diff_out = out_dir / "comparisons" / kind / f"delta_heatmap__{kind}__{outname}__{pol}-minus-{baseline_policy}__{method}.png"
                    plot_policy_difference_heatmap(
                        df,
                        kind=kind,
                        output=outname,
                        policy_a=pol,
                        policy_b=baseline_policy,
                        method=method,
                        top_n_params=top_n_params_heatmap,
                        out_path=diff_out,
                    )

                    # compare lollipop at mid timepoint
                    sub = subset_prcc(df, kind=kind, method=method, policies=[pol, baseline_policy], outputs=[outname])
                    tps = sorted(sub["timepoint"].dropna().unique().tolist())
                    if tps:
                        tp = tps[len(tps)//2]
                        cmp_out = out_dir / "comparisons" / kind / f"lollipop_compare__{kind}__{outname}__t{tp}__{pol}-vs-{baseline_policy}__{method}.png"
                        plot_policy_compare_lollipop(
                            df,
                            kind=kind,
                            output=outname,
                            timepoint=tp,
                            policy_a=pol,
                            policy_b=baseline_policy,
                            method=method,
                            k=min(25, top_k),
                            out_path=cmp_out,
                        )

    written["out_dir"] = out_dir
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Run directory containing results/prcc_results_long.parquet")
    ap.add_argument("--parquet", default=None, help="Override parquet path (optional)")
    ap.add_argument("--method", default="bhfdr", choices=["uncorrected","ztest","bonferroni","bhfdr"])
    ap.add_argument("--sigonly", action="store_true", help="Only use significant entries")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--topn-heatmap", type=int, default=30)
    ap.add_argument("--baseline-policy", default=None, help="Policy name to use as baseline for Δ plots")
    args = ap.parse_args()

    out = make_prcc_report(
        args.run_dir,
        parquet_path=args.parquet,
        method=args.method,  # type: ignore[arg-type]
        significant_only=args.sigonly,
        top_k=args.topk,
        top_n_params_heatmap=args.topn_heatmap,
        baseline_policy=args.baseline_policy,
    )
    print(f"[prcc_postrun_analysis] Wrote outputs under: {out['out_dir']}")