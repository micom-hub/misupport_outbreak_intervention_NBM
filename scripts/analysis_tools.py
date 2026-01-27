from typing import Dict, List, Optional, Tuple, Any, Callable, Iterable, Union
import itertools
import copy
import re
import warnings
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import ast 
import json

def generate_lhs_variants(    variants_list: List[Dict[str, Any]],    param_ranges: Dict[str, Tuple[float, float]],    n: int,    *,    name_sep: str = "__",    value_formatter: Optional[Callable[[Any], str]] = None,    int_keys: Optional[Iterable[str]] = None,    seed: Optional[int] = None) -> List[Dict[str, Any]]:    
    """    
Take base variants and conduct a parameter sweep across specified parameters

  - Each generated variant includes:
      * 'param_overrides' -> base param overrides merged with sampled sweep values
      * 'sweep_params' -> dict[param_name -> sampled_value] (explicitly stored)
      * 'sweep_index' -> unique integer index for the sample (global across all base variants)
  - Variant names are produced as "<base_name>{name_sep}sample_{global_index}"

Args:
  variants_list: list of base variant dicts (each may have 'param_overrides' and 'name').
  param_ranges: dict mapping param_name -> (min, max). All values are treated as continuous.
  n: number of LHS samples per base variant (integer >= 1).
  name_sep: separator used when building generated variant names (default "__").
  value_formatter: optional callable(value)->str for name formatting (not required since names are indexed).
  int_keys: iterable of parameter names that should be cast to int in overrides.
  seed: optional RNG seed for reproducibility.

Returns:
  List[Dict[str, Any]]: flattened list of generated variant dicts. Each dict is a deep copy of
  the base variant with the added/updated keys described above.
"""
    if not isinstance(variants_list, list):
        raise TypeError("variants_list must be a list of variant dicts")
    if not isinstance(param_ranges, dict):
        raise TypeError("param_ranges must be a dict mapping param -> (min, max)")
    n = int(n)
    if n <= 0:
        raise ValueError("n must be >= 1")

    # If no parameters to sweep, return deep copies of base variants but add empty sweep_params
    if not param_ranges:
        out = []
        for base in variants_list:
            bcopy = copy.deepcopy(base)
            bcopy.setdefault("param_overrides", copy.deepcopy(bcopy.get("param_overrides", {}) or {}))
            bcopy["sweep_params"] = {}
            bcopy["sweep_index"] = None
            out.append(bcopy)
        return out

    # preserve param order
    sweep_keys = list(param_ranges.keys())

    total_out = n * max(1, len(variants_list))
    print(f"Generating {total_out} variants ({len(variants_list)} base variant(s) x {n} LHS samples each)")
    # RNG
    rng = np.random.default_rng(seed)

    # Build LHS samples: create base jittered strata and permute per parameter
    base = (np.arange(n) + rng.random(n)) / n  # values in [0,1)
    value_arrays: Dict[str, np.ndarray] = {}
    for k in sweep_keys:
        mn, mx = param_ranges[k]
        mn = float(mn)
        mx = float(mx)
        if mn == mx:
            vals = np.full(n, mn, dtype=float)
        else:
            perm = rng.permutation(n)
            u = base[perm]
            vals = mn + u * (mx - mn)
        value_arrays[k] = vals

    # formatter (optional, not used for uniqueness; kept for compatibility)
    if value_formatter is None:
        def _fmt_val(x):
            try:
                if isinstance(x, (int, np.integer)):
                    return str(int(x))
                f = float(x)
                return f"{f:.6g}"
            except Exception:
                return str(x)
        fmt = _fmt_val
    else:
        fmt = value_formatter

    int_keys = set(int_keys) if int_keys is not None else set()

    out_variants: List[Dict[str, Any]] = []
    global_index = 0
    for base_variant in variants_list:
        if not isinstance(base_variant, dict):
            raise TypeError("Each element of variants_list must be a dict")
        base_copy = copy.deepcopy(base_variant)
        base_name = str(base_copy.get("name", "variant"))
        base_param_overrides = copy.deepcopy(base_copy.get("param_overrides", {}) or {})

        # produce n samples for this base variant
        for i in range(n):
            sweep_params: Dict[str, Any] = {}
            for k in sweep_keys:
                raw_val = value_arrays[k][i]
                if k in int_keys:
                    val = int(round(float(raw_val)))
                else:
                    if isinstance(raw_val, (np.floating, np.float32, np.float64)):
                        val = float(raw_val)
                    elif isinstance(raw_val, (np.integer, np.int32, np.int64)):
                        val = int(raw_val)
                    else:
                        val = raw_val
                sweep_params[k] = val

            new_variant = copy.deepcopy(base_copy)
            # merge sweep params into param_overrides (new overrides override base)
            new_overrides = copy.deepcopy(base_param_overrides)
            new_overrides.update(sweep_params)
            new_variant["param_overrides"] = new_overrides
            new_variant["sweep_params"] = sweep_params
            new_variant["sweep_index"] = int(global_index)
            # name uses unique global index so names are unique across all bases
            new_variant["name"] = f"{base_name}{name_sep}sample_{global_index}"
            out_variants.append(new_variant)
            global_index += 1

    return out_variants


#parsing helper
def _try_parse_value(s: str):
    """Try to interpret string as int, then float, then bool, else return original string."""
    if s is None:
        return None
    s = str(s)
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except Exception:
            pass
    try:
        f = float(s)
        return f
    except Exception:
        return s

#get variant parameter values from name
def _parse_variant_name(variant_name: str, name_sep: str = "__") -> Tuple[str, Dict[str, Any]]:
    """
    Parse variant_name like "Base__param1=1__param2=0.5" into:
      base_name = "Base"
      params = {"param1": 1, "param2": 0.5}
    """
    if variant_name is None:
        return "", {}
    parts = str(variant_name).split(name_sep)
    base = parts[0]
    params: Dict[str, Any] = {}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            params[k] = _try_parse_value(v)
        elif part:
            # bare token -> True
            params[part] = True
    return base, params


#Aggregate variant results from sweeps
def aggregate_variant_results(
    variant_results: List[Dict[str, Any]],
    *,
    name_sep: str = "__",
    aggregate_sweeps: bool = True,
    aggregated_summary: bool = True,
    reduced_results: bool = False,
    numeric_aggs: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Build a per-run DataFrame for all variants and optionally an aggregated summary.

    Args:
        variant_results: list of dicts with keys:
            - 'variant_name' (str) or 'name'
            - 'model' (NetworkModel instance, already simulated)
        name_sep: separator used in variant names for swept parameters (default "__")
        aggregate_sweeps: if True, parsed sweep parameters are exposed as columns in overall_df
        aggregated_summary: if True, return a second DataFrame with aggregated summary stats
        aggregated_action_log: if True, returns a third Dataframe with aggregated action log
        numeric_aggs: list of aggregation functions to compute for numeric columns (default: ['mean','std','median','min','max'])

    Returns:
        (overall_df, combined_summary_or_None)
        - overall_df: pandas DataFrame with one row per model run, augmented with:
            ['variant_name', 'variant_dir', 'base_variant', <sweep_param_cols...>, <epi_outcomes columns...>]
        - combined_summary: if aggregated_summary True, a DataFrame of aggregated statistics grouped by:
            * if aggregate_sweeps and sweep params present: ['base_variant'] + sorted(sweep_param_keys)
            * else: ['variant_name']
          combined_summary contains columns like '<metric>_mean', '<metric>_std', etc., plus 'n_runs'.
          If aggregated_summary is False, this return value is None.

    Notes:
        - If a model or model.epi_outcomes() fails, that variant will be skipped with a warning.
        - The function always returns overall_df (may be empty).
    """
    if numeric_aggs is None:
        numeric_aggs = ["mean", "std", "median", "min", "max"]

    per_run_frames: List[pd.DataFrame] = []
    all_param_keys = set()

    for i, entry in enumerate(variant_results):
        vname = entry.get("variant_name") or entry.get("name") or f"variant_{i}"
        model = entry.get("model", None)

        base_name, params = _parse_variant_name(vname, name_sep=name_sep)
        all_param_keys.update(params.keys())

        if model is None:
            warnings.warn(f"aggregate_variant_results: missing model for variant '{vname}' - skipping")
            continue

        # call epi_outcomes for model
        try:
            epi_df = model.epi_outcomes(reduced = reduced_results)
            if not isinstance(epi_df, pd.DataFrame):
                epi_df = pd.DataFrame(epi_df)
        except Exception as exc:
            warnings.warn(f"aggregate_variant_results: epi_outcomes() failed for '{vname}': {exc}")
            epi_df = pd.DataFrame()

        # augment per-run table with variant metadata and sweep params
        pr = epi_df.copy()
        pr["variant_name"] = vname
        pr["base_variant"] = base_name
        # attach parsed sweep parameters as columns (same value for every run row of this model)
        for k, val in params.items():
            pr[k] = val

        per_run_frames.append(pr)

    # overall per-run DataFrame
    if per_run_frames:
        overall_df = pd.concat(per_run_frames, ignore_index=True)
    else:
        # create an empty overall_df with canonical columns so callers can rely on column existence
        cols = ["variant_name", "variant_dir", "base_variant"] + sorted(all_param_keys)
        overall_df = pd.DataFrame(columns=cols)

    # Optionally build aggregated summary
    combined_summary: Optional[pd.DataFrame] = None
    if aggregated_summary:
        # determine grouping columns
        sweep_keys = sorted(list(all_param_keys)) if aggregate_sweeps else []
        if aggregate_sweeps and sweep_keys:
            group_by_cols = ["base_variant"] + sweep_keys
        else:
            # no sweep parameters detected or aggregate_sweeps False -> group per variant_name
            group_by_cols = ["variant_name"]

        # Ensure the grouping columns exist in overall_df
        for g in group_by_cols:
            if g not in overall_df.columns:
                overall_df[g] = np.nan

        # numeric columns to aggregate (exclude group_by columns)
        numeric_cols = overall_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_from_aggregation = {"run_number"}
        exclude_set = set(group_by_cols) | exclude_from_aggregation
        numeric_cols_to_agg = [c for c in numeric_cols if c not in exclude_set]

        if overall_df.empty or not numeric_cols_to_agg:
            # Return an empty aggregated DataFrame with group_by columns + a 'n_runs' column
            combined_summary = pd.DataFrame(columns=group_by_cols + ["n_runs"])
        else:
            # perform groupby aggregation
            grouped = overall_df.groupby(group_by_cols)[numeric_cols_to_agg].agg(numeric_aggs)

            # flatten multiindex columns -> "metric_agg"
            grouped.columns = [f"{metric}_{agg}" for metric, agg in grouped.columns.to_flat_index()]

            # add n_runs (group size)
            counts = overall_df.groupby(group_by_cols).size().rename("n_runs")
            grouped = grouped.join(counts)

            combined_summary = grouped.reset_index()

            #Organize overall_df
            order = [
            'base_variant', 'variant_name', 'run_number',
            'time', 'action_id', 'action_type', 'kind',
            'nodes_count', 'hours_used', 'duration',
            'reversible_tokens', 'nonreversible_tokens'
        ]

        cols = [c for c in order if c in overall_df.columns] + [c for c in overall_df.columns if c not in order]
        overall_df = overall_df[cols]


    return overall_df, combined_summary

#Build time series plots
def plot_epi_series(
    df_timeseries: pd.DataFrame,  
    type: str = "prevalence",
    spaghetti: bool = False,
    frac: bool = True,
    fixed_axis: bool = True,
    outbreak_threshold: Optional[float] = None
    ):
    """
    Args:
    df_timeseries: DataFrame with columns ['variant_name','base_variant','run_number',
                    'prevalence_count_series','prevalence_frac_series',
                    'cumulative_infections_count_series','cumulative_infections_frac_series', ...]
    type: "prevalence" or "incidence"
    spaghetti: if True plot spaghetti lines per run; if False plot fan (quantiles) + median
    frac: if True plot fraction series, else plot counts
    fixed_axis: if True, match y-axis across facets (top row or entire grid as appropriate)
    outbreak_threshold: optional threshold to define outbreak runs (count if >1, fraction if <=1).
                        If provided creates two-row layout: top = all runs, bottom = outbreak runs.
    Returns:
    (fig, axes) matplotlib Figure and axes array

    """
    # Visual defaults (kept consistent)
    spaghetti_color = "#00274C"
    spaghetti_alpha = 0.22
    mean_line_color = "#00274C"
    mean_line_width = 2.2
    median_color = "#DB130D"
    median_width = 2.8
    ci50_color = "#00283B"
    ci90_color = "#0584FA"
    ci95_color = "#8CC6FD"
    band_alpha = 0.8
    figsize_per_facet = (5.2, 4.6)

    t = str(type).lower().strip()
    if t not in {"prevalence", "incidence"}:
        raise ValueError('type must be "prevalence" or "incidence"')

    # required per-run columns
    required_cols = {
        "variant_name", "base_variant", "run_number",
        "prevalence_count_series", "prevalence_frac_series",
        "cumulative_infections_count_series", "cumulative_infections_frac_series"
    }
    missing = required_cols - set(df_timeseries.columns)
    if missing:
        raise ValueError(f"timeseries DataFrame missing required columns: {missing}")

    # robust list parsing helper
    def _ensure_list(x):
        if x is None:
            return []
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return []
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return []
        try:
            # scalar NaN check
            if np.isscalar(x) and pd.isna(x):
                return []
        except Exception:
            pass
        return [x]

    # build long-format records per run/time and capture final cumulative for outbreak detection
    long_records = []
    final_values = {}  # (variant_name, run_number) -> {'final_count', 'final_frac'}

    for _, row in df_timeseries.iterrows():
        vname = row["variant_name"]
        bname = row["base_variant"]
        run_num = int(row.get("run_number", 0))

        prev_count = _ensure_list(row.get("prevalence_count_series", []))
        prev_frac = _ensure_list(row.get("prevalence_frac_series", []))
        cum_count = _ensure_list(row.get("cumulative_infections_count_series", []))
        cum_frac = _ensure_list(row.get("cumulative_infections_frac_series", []))

        # choose series to plot
        if t == "prevalence":
            series = prev_frac if frac else prev_count
            T = max(len(prev_count), len(prev_frac), len(cum_count), len(cum_frac))
            # pad with last value
            def pad(seq, L):
                s = list(seq)
                if len(s) >= L:
                    return s[:L]
                fill = s[-1] if s else 0
                return s + [fill] * (L - len(s))
            series = pad(series, T)
            final_c = int(cum_count[-1]) if len(cum_count) > 0 else 0
            final_f = float(cum_frac[-1]) if len(cum_frac) > 0 else 0.0
        else:  # cumulative incidence plotting (never decreases)
            # use cumulative arrays directly (counts or fractions)
            if frac:
                cum = list(cum_frac)
            else:
                cum = list(cum_count)
            T = len(cum)
            # pad if necessary based on available series lengths
            if T == 0:
                # fallback: try to infer from prevalence length
                T = max(len(prev_count), len(prev_frac))
            def pad(seq, L):
                s = list(seq)
                if len(s) >= L:
                    return s[:L]
                fill = s[-1] if s else 0
                return s + [fill] * (L - len(s))
            cum = pad(cum, T)
            # enforce non-decreasing by cumulative max
            try:
                cum = list(np.maximum.accumulate(np.array(cum, dtype=float)))
            except Exception:
                # fallback simple loop
                cum_out = []
                curmax = -np.inf
                for v in cum:
                    try:
                        vv = float(v)
                    except Exception:
                        vv = 0.0
                    if vv < curmax:
                        vv = curmax
                    else:
                        curmax = vv
                    cum_out.append(vv)
                cum = cum_out
            series = cum
            final_c = int(cum_count[-1]) if len(cum_count) > 0 else int(series[-1]) if series else 0
            final_f = float(cum_frac[-1]) if len(cum_frac) > 0 else float(series[-1]) if series else 0.0

        # append long records
        for ti, val in enumerate(series):
            try:
                v = float(val)
            except Exception:
                v = float("nan")
            long_records.append({
                "base_variant": bname,
                "variant_name": vname,
                "run_number": int(run_num),
                "time": int(ti),
                "value": v
            })

        final_values[(vname, run_num)] = {"final_count": int(final_c), "final_frac": float(final_f)}

    if not long_records:
        raise RuntimeError("No timeseries data to plot")

    df_long = pd.DataFrame(long_records)

    # facets by base_variant; one column per base variant (per request)
    facets = sorted(df_long["base_variant"].dropna().unique().tolist()) if "base_variant" in df_long.columns else [None]
    n_facets = len(facets) if facets else 1

    create_two_rows = outbreak_threshold is not None
    if create_two_rows:
        ncols_eff = n_facets if n_facets > 0 else 1
        nrows_eff = 2
    else:
        ncols_eff = n_facets if n_facets > 0 else 1
        nrows_eff = 1

    fig, axes = plt.subplots(nrows_eff, ncols_eff,
                            figsize=(figsize_per_facet[0] * ncols_eff, figsize_per_facet[1] * nrows_eff),
                            squeeze=False)
    axes = np.array(axes).reshape(nrows_eff, ncols_eff)

    def _plot_for_facet(ax, sub_all_df):
        if sub_all_df.empty:
            return 0.0, 0
        total_runs = sub_all_df.groupby(["variant_name", "run_number"]).ngroups
        if spaghetti:
            for (_, _), grp in sub_all_df.groupby(["variant_name", "run_number"], sort=False):
                ax.plot(grp["time"].to_numpy(), grp["value"].to_numpy(), color=spaghetti_color, alpha=spaghetti_alpha, linewidth=0.7)
            mean_series = sub_all_df.groupby("time")["value"].mean().sort_index()
            if not mean_series.empty:
                ax.plot(mean_series.index.to_numpy(), mean_series.to_numpy(), color=mean_line_color, linewidth=mean_line_width)
            local_max = sub_all_df["value"].max()
            return float(local_max), int(total_runs)
        else:
            qs = sub_all_df.groupby("time")["value"].quantile([0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]).unstack().sort_index()
            if qs.empty:
                return 0.0, int(total_runs)
            x = qs.index.to_numpy()
            ax.fill_between(x, qs[0.025], qs[0.975], color=ci95_color, alpha=band_alpha, linewidth=0)
            ax.fill_between(x, qs[0.05], qs[0.95], color=ci90_color, alpha=band_alpha, linewidth=0)
            ax.fill_between(x, qs[0.25], qs[0.75], color=ci50_color, alpha=band_alpha, linewidth=0)
            ax.plot(x, qs[0.50], color=median_color, linewidth=median_width)
            local_max = float(np.nanmax(qs.to_numpy()))
            return local_max, int(total_runs)

    # compute outbreak pairs
    outbreak_pairs = set()
    if create_two_rows:
        use_count = float(outbreak_threshold) > 1.0
        thr = float(outbreak_threshold)
        for (vname, rn), finals in final_values.items():
            if use_count:
                if finals.get("final_count", 0) > thr:
                    outbreak_pairs.add((vname, int(rn)))
            else:
                if finals.get("final_frac", 0.0) > thr:
                    outbreak_pairs.add((vname, int(rn)))

    top_counts = []
    bot_counts = []
    top_row_max = 0.0
    bottom_row_max = 0.0

    for i, facet in enumerate(facets):
        if create_two_rows:
            row_top, col_top = 0, i
            row_bot, col_bot = 1, i
        else:
            row_top, col_top = 0, i
            row_bot, col_bot = None, None

        ax_top = axes[row_top, col_top]
        sub_all = df_long[df_long["base_variant"] == facet].copy()
        local_max_top, total_runs = _plot_for_facet(ax_top, sub_all)
        top_counts.append(total_runs)
        top_row_max = max(top_row_max, local_max_top)
        ax_top.set_title(str(facet))
        ax_top.set_xlabel("Day")
        if t == "prevalence":
            ylabel = "Prevalence (fraction)" if frac else "Prevalence (count)"
        else:
            ylabel = "Cumulative incidence (fraction)" if frac else "Cumulative incidence (count)"
        ax_top.set_ylabel(ylabel)
        ax_top.grid(True, alpha=0.25)
        ax_top.set_ylim(bottom=0)

        if create_two_rows:
            ax_bot = axes[row_bot, col_bot]
            if outbreak_pairs:
                valid = [(v, r) for (v, r) in outbreak_pairs if v in sub_all["variant_name"].unique()]
                if valid:
                    valid_df = pd.DataFrame(valid, columns=["variant_name", "run_number"])
                    sub_bot = sub_all.merge(valid_df, on=["variant_name", "run_number"], how="inner")
                else:
                    sub_bot = pd.DataFrame(columns=sub_all.columns)
            else:
                sub_bot = pd.DataFrame(columns=sub_all.columns)

            if not sub_bot.empty:
                local_max_bot, outbreak_count = _plot_for_facet(ax_bot, sub_bot)
            else:
                local_max_bot, outbreak_count = 0.0, 0
            bot_counts.append(outbreak_count)
            bottom_row_max = max(bottom_row_max, local_max_bot)
            ax_bot.set_xlabel("Day")
            ax_bot.set_ylabel(ylabel)
            ax_bot.grid(True, alpha=0.25)
            ax_bot.set_ylim(bottom=0)

    # Turn off any unused axes (unlikely since columns == facets)
    if not create_two_rows:
        total_slots = nrows_eff * ncols_eff
        for slot in range(n_facets, total_slots):
            r = slot // ncols_eff
            c = slot % ncols_eff
            axes[r, c].axis("off")

    left_margin = 0.11 if create_two_rows else 0.07
    bottom_margin = 0.16
    top_rect = 0.90
    fig.tight_layout(rect=[left_margin, bottom_margin, 1.0, top_rect])
    fig.subplots_adjust(hspace=0.35) 

    if create_two_rows and fixed_axis:
        if top_row_max > 0:
            ymax_top = top_row_max * 1.05
            for c in range(ncols_eff):
                axes[0, c].set_ylim(0, ymax_top)
        if bottom_row_max > 0:
            ymax_bot = bottom_row_max * 1.05
            for c in range(ncols_eff):
                axes[1, c].set_ylim(0, ymax_bot)
    elif not create_two_rows and fixed_axis:
        all_max = 0.0
        for ax in axes.ravel():
            for ln in ax.get_lines():
                ydata = ln.get_ydata()
                if len(ydata):
                    all_max = max(all_max, np.nanmax(ydata))
        if all_max > 0:
            ymax = all_max * 1.05
            for ax in axes.ravel():
                ax.set_ylim(0, ymax)

    left_x = max(0.005, left_margin - 0.025)
    if create_two_rows:
        top_centers = [axes[0, c].get_position() for c in range(ncols_eff)]
        bottom_centers = [axes[1, c].get_position() for c in range(ncols_eff)]
        top_row_center = float(np.mean([p.y0 + p.height / 2.0 for p in top_centers]))
        bottom_row_center = float(np.mean([p.y0 + p.height / 2.0 for p in bottom_centers]))
        fig.text(left_x, top_row_center, "All Runs", va="center", ha="center", rotation="vertical", fontsize=12, fontweight="bold")
        fig.text(left_x, bottom_row_center, f"Outbreak Runs (threshold {outbreak_threshold})", va="center", ha="center", rotation="vertical", fontsize=11)
        for c in range(ncols_eff):
            p_top = axes[0, c].get_position()
            x_center = p_top.x0 + p_top.width / 2.0
            y_label_pos = p_top.y0 - 0.055
            fig.text(x_center, y_label_pos, f"Number of Runs: {top_counts[c] if c < len(top_counts) else 0}", ha="center", va="top", fontsize=9)
            p_bot = axes[1, c].get_position()
            x_center2 = p_bot.x0 + p_bot.width / 2.0
            y_label_pos2 = p_bot.y0 - 0.055
            fig.text(x_center2, y_label_pos2, f"Number of Runs: {bot_counts[c] if c < len(bot_counts) else 0}", ha="center", va="top", fontsize=9)
    else:
        for i, facet in enumerate(facets):
            r, c = 0, i
            p = axes[r, c].get_position()
            x_center = p.x0 + p.width / 2.0
            y_label_pos = p.y0 - 0.055
            sub_all = df_long if facet is None else df_long[df_long["base_variant"] == facet].copy()
            total_runs = sub_all[["variant_name", "run_number"]].drop_duplicates().shape[0]
            fig.text(x_center, y_label_pos, f"Number of Runs: {total_runs}", ha="center", va="top", fontsize=9)

    if not spaghetti:
        handles = [
            Line2D([0], [0], color=median_color, lw=median_width, label="Median"),
            Patch(facecolor=ci50_color, alpha=band_alpha, label="50% band"),
            Patch(facecolor=ci90_color, alpha=band_alpha, label="90% band"),
            Patch(facecolor=ci95_color, alpha=band_alpha, label="95% band"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False)

    main_title = ("Prevalence over time (fraction)" if t == "prevalence" and frac else
                "Prevalence over time (count)" if t == "prevalence" else
                "Cumulative incidence over time (fraction)" if frac else
                "Cumulative incidence over time (count)")
    fig.suptitle(main_title, x=0.5, y=0.995, ha="center", fontsize=14, fontweight="bold")

    return fig, axes
