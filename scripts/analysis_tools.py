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
import os 

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

    # required per-run columns (keep checks broad but informative)
    req_cols = {
        "variant_name",
        "base_variant",
        "run_number",
        "prevalence_count_series",
        "prevalence_frac_series",
        "cumulative_infections_count_series",
        "cumulative_infections_frac_series",
    }
    missing = req_cols - set(df_timeseries.columns)
    if missing:
        raise ValueError(f"timeseries DataFrame missing required columns: {missing}")

    # robust list parsing helper (handles lists, numpy arrays, JSON or Python-list strings, NaN)
    def _ensure_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, np.ndarray):
            try:
                return x.tolist()
            except Exception:
                return list(np.asarray(x).tolist())
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return []
            # try JSON first, then Python literal
            try:
                parsed = json.loads(s)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                # scalar -> wrap
                return [parsed]
            except Exception:
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        return list(parsed)
                    return [parsed]
                except Exception:
                    return []
        # pandas NA-like values
        try:
            if pd.isna(x):
                return []
        except Exception:
            pass
        # fallback: wrap scalar in list
        return [x]

    # helpers to convert lists to numeric sequences
    def _to_int_list(seq):
        out = []
        for v in seq:
            try:
                if isinstance(v, str):
                    v2 = v.strip()
                    if v2 == "":
                        continue
                    out.append(int(float(v2)))
                else:
                    if pd.isna(v):
                        continue
                    out.append(int(float(v)))
            except Exception:
                # ignore bad entries
                continue
        return out

    def _to_float_list(seq):
        out = []
        for v in seq:
            try:
                if isinstance(v, str):
                    v2 = v.strip()
                    if v2 == "":
                        out.append(float("nan"))
                    else:
                        out.append(float(v2))
                else:
                    if pd.isna(v):
                        out.append(float("nan"))
                    else:
                        out.append(float(v))
            except Exception:
                out.append(float("nan"))
        return out

    # 1) One-pass parse of all runs into a compact run_entries list
    run_entries = []  # each entry: {variant_name, base_variant, run_number, raw_series, T_row, final_count, final_frac}
    for _, row in df_timeseries.iterrows():
        vname = row.get("variant_name")
        bname = row.get("base_variant")
        # robust run_number parsing
        rn_val = row.get("run_number", 0)
        try:
            run_num = int(rn_val)
        except Exception:
            try:
                run_num = int(float(rn_val))
            except Exception:
                run_num = 0

        # parse all series that may be present
        prev_count_raw = _ensure_list(row.get("prevalence_count_series", []))
        prev_frac_raw = _ensure_list(row.get("prevalence_frac_series", []))
        cum_count_raw = _ensure_list(row.get("cumulative_infections_count_series", []))
        cum_frac_raw = _ensure_list(row.get("cumulative_infections_frac_series", []))

        # numeric conversions
        prev_count = _to_int_list(prev_count_raw)
        prev_frac = _to_float_list(prev_frac_raw)
        cum_count = _to_int_list(cum_count_raw)
        cum_frac = _to_float_list(cum_frac_raw)

        # Per-run canonical time length (mimic original behavior)
        T_row = max(len(prev_count), len(prev_frac), len(cum_count), len(cum_frac))

        # choose raw_series according to type/frac
        if t == "prevalence":
            chosen_raw = prev_frac if frac else prev_count
        else:  # cumulative/incidence plotting
            chosen_raw = cum_frac if frac else cum_count

        # convert chosen_raw to numeric float list (do not pad yet)
        chosen_numeric = _to_float_list(chosen_raw)

        # final sizes for outbreak decisions (prefer cumulative arrays if present)
        final_count = cum_count[-1] if len(cum_count) > 0 else None
        final_frac = cum_frac[-1] if len(cum_frac) > 0 else None

        run_entries.append(
            {
                "variant_name": vname,
                "base_variant": bname,
                "run_number": int(run_num),
                "raw_series": chosen_numeric,  # list[float] (may be empty)
                "T_row": int(T_row),
                "final_count": int(final_count) if final_count is not None else None,
                "final_frac": float(final_frac) if final_frac is not None else None,
            }
        )

    if not run_entries:
        raise RuntimeError("No runs parsed from timeseries DataFrame")

    # 2) Compute outbreak boolean per run (robust)
    if outbreak_threshold is not None:
        thr = float(outbreak_threshold)
        use_fraction = thr <= 1.0  # heuristic: <=1 -> interpret as fraction, >1 -> interpret as count
        for e in run_entries:
            if use_fraction:
                final_f = e.get("final_frac")
                e["outbreak"] = bool(final_f is not None and (final_f > thr))
            else:
                final_c = e.get("final_count")
                e["outbreak"] = bool(final_c is not None and (final_c > thr))
    else:
        for e in run_entries:
            e["outbreak"] = False

    # 3) Group runs by facet (base_variant)
    facet_map = {}
    for e in run_entries:
        facet_key = e.get("base_variant", None)
        # Keep None as-is so titles show "None" -> will be cast to str later
        facet_map.setdefault(facet_key, []).append(e)

    facets = sorted(facet_map.keys(), key=lambda x: str(x)) if facet_map else [None]
    n_facets = len(facets) if facets else 1

    create_two_rows = outbreak_threshold is not None
    if create_two_rows:
        ncols_eff = n_facets if n_facets > 0 else 1
        nrows_eff = 2
    else:
        ncols_eff = n_facets if n_facets > 0 else 1
        nrows_eff = 1

    fig, axes = plt.subplots(
        nrows_eff,
        ncols_eff,
        figsize=(figsize_per_facet[0] * ncols_eff, figsize_per_facet[1] * nrows_eff),
        squeeze=False,
    )
    axes = np.array(axes).reshape(nrows_eff, ncols_eff)

    # Helper to pad a run series to length L (pad with last value or 0)
    def _pad_series_to_len(series_list, L):
        s = list(series_list) if series_list else []
        if L <= 0:
            return []
        if len(s) >= L:
            # truncate
            return [float(x) if (x is not None and not pd.isna(x)) else float("nan") for x in s[:L]]
        # pad with last value or 0
        fill = None
        if len(s) > 0:
            try:
                fill = float(s[-1])
            except Exception:
                fill = 0.0
        else:
            fill = 0.0
        out = []
        for x in s:
            try:
                out.append(float(x))
            except Exception:
                out.append(float("nan"))
        out.extend([fill] * (L - len(out)))
        return out

    # Plotting helper using vectorized arrays (much faster than groupby-quantile)
    def _plot_for_facet(ax, entries_for_facet):
        """
        entries_for_facet: list of run-entry dicts for a given facet
        returns: (local_max_value, total_runs)
        """
        if not entries_for_facet:
            return 0.0, 0

        total_runs = len(entries_for_facet)
        T_facet = max(int(e.get("T_row", 0)) for e in entries_for_facet)
        if T_facet <= 0:
            # nothing to plot (no series lengths); still return run count
            return 0.0, int(total_runs)

        # Build numeric 2D array (n_runs x T_facet)
        arr = np.full((total_runs, T_facet), np.nan, dtype=float)
        for i, e in enumerate(entries_for_facet):
            padded = _pad_series_to_len(e.get("raw_series", []), T_facet)
            # ensure length
            if len(padded) != T_facet:
                padded = _pad_series_to_len(padded, T_facet)
            arr[i, :] = np.asarray(padded, dtype=float)

        if spaghetti:
            # spaghetti: plot each run line, then mean
            for i in range(arr.shape[0]):
                ax.plot(np.arange(T_facet), arr[i, :], color=spaghetti_color, alpha=spaghetti_alpha, linewidth=0.7)
            # mean series (ignore NaNs)
            with np.errstate(invalid="ignore"):
                mean_series = np.nanmean(arr, axis=0)
            # plot mean if not all NaNs
            if not np.all(np.isnan(mean_series)):
                ax.plot(np.arange(T_facet), mean_series, color=mean_line_color, linewidth=mean_line_width)
            local_max = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 0.0
            return local_max, int(total_runs)
        else:
            # fan plot: compute quantiles vectorized
            # check for any finite values
            if not np.isfinite(arr).any():
                return 0.0, int(total_runs)

            # percentiles in numeric form (np.nanpercentile expects 0-100)
            q_levels = [2.5, 5.0, 25.0, 50.0, 75.0, 95.0, 97.5]
            try:
                qs = np.nanpercentile(arr, q_levels, axis=0)
                # qs shape -> (len(q_levels), T_facet)
            except Exception:
                # fallback to safe quantile via np.apply_along_axis (slower)
                qs = np.vstack([np.nanpercentile(arr, q, axis=0) for q in q_levels])

            x = np.arange(arr.shape[1])
            # plot bands
            ax.fill_between(x, qs[0], qs[-1], color=ci95_color, alpha=band_alpha, linewidth=0)
            ax.fill_between(x, qs[1], qs[-2], color=ci90_color, alpha=band_alpha, linewidth=0)
            ax.fill_between(x, qs[2], qs[-3], color=ci50_color, alpha=band_alpha, linewidth=0)
            # median
            ax.plot(x, qs[3], color=median_color, linewidth=median_width)
            local_max = float(np.nanmax(qs)) if np.isfinite(np.nanmax(qs)) else 0.0
            return local_max, int(total_runs)

    # Prepare lists to store maxima and counts for axis scaling and annotations
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
        entries_all = facet_map.get(facet, [])
        # top: all runs within facet
        local_max_top, total_runs = _plot_for_facet(ax_top, entries_all)
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
            # bottom: outbreak runs subset within the facet
            ax_bot = axes[row_bot, col_bot]
            outbreak_entries = [e for e in entries_all if bool(e.get("outbreak", False))]
            if outbreak_entries:
                local_max_bot, outbreak_count = _plot_for_facet(ax_bot, outbreak_entries)
            else:
                local_max_bot, outbreak_count = 0.0, 0
            bot_counts.append(outbreak_count)
            bottom_row_max = max(bottom_row_max, local_max_bot)
            ax_bot.set_xlabel("Day")
            ax_bot.set_ylabel(ylabel)
            ax_bot.grid(True, alpha=0.25)
            ax_bot.set_ylim(bottom=0)

    # turn off unused axes (if any)
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

    # fixed axis scaling (preserve original behavior)
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

    # layout and annotations (keep same look)
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
            sub_all = [] if facet not in facet_map else facet_map[facet]
            total_runs = len(sub_all)
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
   
def ensure_list_cell(x):
    """
    Return a Python list for common types stored in timeseries rows:
      - list/tuple -> list
      - numpy.ndarray -> tolist()
      - JSON-string / Python-list-string -> parsed
      - scalar/NaN -> [] or [scalar] depending
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (int, float)) and not np.isnan(x):
        # scalar -> single-element list (rare)
        return [x]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # try JSON first
        try:
            parsed = json.loads(s)
            # ensure lists are lists
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            # If parsed a scalar, wrap
            return [parsed]
        except Exception:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                return [parsed]
            except Exception:
                # fallback: return the raw string in list
                return [s]
    # pandas NA-like values
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    # fallback: wrap in list
    return [x]
    
def load_run_results(run_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load aggregated run outputs saved by clean_run_results:
        - aggregated_run_results.parquet (or .csv) -> overall_df
        - timeseries.parquet (or .csv) -> timeseries_df (list-columns parsed)
        Returns: (overall_df, timeseries_df)
        Raises RuntimeError if required files not found.
        """
        run_dir = os.path.abspath(run_dir)
        agg_parq = os.path.join(run_dir, "aggregated_run_results.parquet")
        agg_csv = os.path.join(run_dir, "aggregated_run_results.csv")
        ts_parq = os.path.join(run_dir, "timeseries.parquet")
        ts_csv = os.path.join(run_dir, "timeseries.csv")

        # Load overall_df
        overall_df = None
        if os.path.exists(agg_parq):
            try:
                overall_df = pd.read_parquet(agg_parq)
            except Exception as e:
                warnings.warn(f"Failed to read {agg_parq}: {e}; trying CSV fallback.")
        if overall_df is None and os.path.exists(agg_csv):
            overall_df = pd.read_csv(agg_csv, dtype=object)

        if overall_df is None:
            raise RuntimeError(f"No aggregated_run_results found in {run_dir}")

        # Load timeseries
        ts_df = None
        if os.path.exists(ts_parq):
            try:
                ts_df = pd.read_parquet(ts_parq)
            except Exception as e:
                warnings.warn(f"Failed to read {ts_parq}: {e}; trying CSV fallback.")
        if ts_df is None and os.path.exists(ts_csv):
            try:
                ts_df = pd.read_csv(ts_csv, dtype=object)
                # parse list columns that were JSON-string-serialized
                list_cols = [
                    "prevalence_count_series",
                    "prevalence_frac_series",
                    "cumulative_infections_count_series",
                    "cumulative_infections_frac_series",
                ]
                for col in list_cols:
                    if col in ts_df.columns:
                        ts_df[col] = ts_df[col].apply(lambda x: ensure_list_cell(x))
            except Exception as e:
                warnings.warn(f"Failed to read/parse {ts_csv}: {e}; creating empty timeseries DataFrame.")
                ts_df = pd.DataFrame()

        if ts_df is None:
            raise RuntimeError(f"No timeseries found in {run_dir}")

        # Ensure expected dtypes for common columns
        if "run_number" in ts_df.columns:
            try:
                ts_df["run_number"] = ts_df["run_number"].astype(int)
            except Exception:
                # leave as-is if conversion fails
                pass

        return overall_df, ts_df


def timeseries_to_metrics(
    ts_df: pd.DataFrame,
    *,
    outbreak_threshold: Optional[float] = 10.0,
    outbreak_threshold_is_fraction: Optional[bool] = None,
    duration_frac_threshold: float = 0.01,
    auc_normalize_by_T: bool = False,
) -> pd.DataFrame:
    """
    Convert the per-run timeseries rows into a per-run metrics DataFrame.

    For each row (variant_name + run_number) we compute:
      - series_length
      - final_size_count, final_size_frac (if available)
      - peak_prevalence_count, peak_prevalence_frac
      - peak_time (first index of peak)
      - auc_prevalence_frac (sum over days; normalized by T if auc_normalize_by_T True)
      - duration_above_frac (days prevalence_frac >= duration_frac_threshold)
      - early_growth_rate (slope of log(series) per day estimated on first contiguous window of positive values up to 7 days)
      - outbreak (boolean) determined by outbreak_threshold and outbreak_threshold_is_fraction:
          * if outbreak_threshold_is_fraction == True -> use final_size_frac
          * if False -> use final_size_count
          * if None -> auto: if outbreak_threshold > 1 use count, else fraction
    Returns DataFrame with one row per run.
    """
    rows = []

    for _, row in ts_df.iterrows():
        vname = row.get("variant_name", None)
        bname = row.get("base_variant", None)
        run_num = int(row.get("run_number", 0)) if row.get("run_number", None) is not None else 0

        prev_count = ensure_list_cell(row.get("prevalence_count_series", []))
        prev_frac = ensure_list_cell(row.get("prevalence_frac_series", []))
        cum_count = ensure_list_cell(row.get("cumulative_infections_count_series", []))
        cum_frac = ensure_list_cell(row.get("cumulative_infections_frac_series", []))

        # Normalize numeric lists to floats/ints where possible
        def to_float_list(seq):
            out = []
            for x in seq:
                try:
                    out.append(float(x))
                except Exception:
                    out.append(float("nan"))
            return out

        def to_int_list(seq):
            out = []
            for x in seq:
                try:
                    out.append(int(x))
                except Exception:
                    try:
                        out.append(int(float(x)))
                    except Exception:
                        out.append(0)
            return out

        prev_count = to_int_list(prev_count) if prev_count else []
        prev_frac = to_float_list(prev_frac) if prev_frac else []
        cum_count = to_int_list(cum_count) if cum_count else []
        cum_frac = to_float_list(cum_frac) if cum_frac else []

        T = max(len(prev_count), len(prev_frac), len(cum_count), len(cum_frac))

        # Final sizes (prefer cumulative arrays if present)
        final_count = int(cum_count[-1]) if len(cum_count) > 0 else (int(prev_count[-1]) if prev_count else None)
        final_frac = float(cum_frac[-1]) if len(cum_frac) > 0 else (float(prev_frac[-1]) if prev_frac else None)

        # Peak prevalence
        peak_prevalence_count = int(max(prev_count)) if prev_count else None
        peak_prevalence_frac = float(max(prev_frac)) if prev_frac else None

        # Peak time (first occurrence)
        def first_index_of_max(lst):
            if not lst:
                return None
            m = max(lst)
            for i, v in enumerate(lst):
                try:
                    if float(v) == float(m):
                        return int(i)
                except Exception:
                    continue
            return None

        peak_time = first_index_of_max(prev_frac if prev_frac else prev_count)

        # AUC (sum of prevalence fraction); normalize by T if requested
        auc = None
        if prev_frac:
            auc = float(np.nansum(prev_frac))
            if auc_normalize_by_T and T > 0:
                auc = auc / float(T)
        elif prev_count and final_count is not None and final_frac is not None and final_frac > 0:
            # if only counts available but we have final_frac, we can approximate fraction series by scaling
            try:
                approx_frac = [c / final_count * final_frac if final_count > 0 else 0.0 for c in prev_count]
                auc = float(np.nansum(approx_frac))
                if auc_normalize_by_T and T > 0:
                    auc = auc / float(T)
            except Exception:
                auc = None

        # Duration above specified fraction threshold
        duration_above = None
        if prev_frac:
            duration_above = int(sum(1 for v in prev_frac if (not math.isnan(v)) and v >= duration_frac_threshold))
        else:
            duration_above = None

        # Early exponential growth estimate (slope on log scale per day); return slope and doubling_time
        def estimate_growth_rate_from_series(series_floats, min_positive=1, max_days=7):
            if not series_floats:
                return (None, None)
            # use first contiguous window starting at first index with series >= min_positive
            arr = np.array(series_floats, dtype=float)
            # find indices where arr > 0 (or >= min_positive if counts)
            pos_mask = arr > 0
            if not np.any(pos_mask):
                return (None, None)
            first_pos = int(np.nonzero(pos_mask)[0][0])
            # build window of up to max_days from first_pos, requiring positive values
            window = arr[first_pos : first_pos + max_days]
            # require at least 3 points for a slope
            if len(window) < 3 or not np.all(window > 0):
                # try to expand window to include zeros converted to tiny epsilon
                window = arr[first_pos : first_pos + max_days]
                if len(window) < 3 or np.sum(window > 0) < 2:
                    return (None, None)
            # fit slope on log(window)
            t = np.arange(len(window))
            y = window
            # guard against non-positive values by clipping to tiny epsilon
            y_clipped = np.clip(y, 1e-9, None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    slope, intercept = np.polyfit(t, np.log(y_clipped), 1)
                except Exception:
                    return (None, None)
            # doubling time: ln(2) / slope if slope > 0
            doubling = None
            try:
                if slope > 0:
                    doubling = math.log(2.0) / slope
                else:
                    doubling = None
            except Exception:
                doubling = None
            return (float(slope), doubling)

        # choose series for growth estimation: prefer counts if not all zeros; else use fractions
        growth_slope = None
        growth_doubling = None
        if prev_count and any(v > 0 for v in prev_count):
            growth_slope, growth_doubling = estimate_growth_rate_from_series([float(x) for x in prev_count])
        elif prev_frac and any((not math.isnan(v)) and v > 0 for v in prev_frac):
            growth_slope, growth_doubling = estimate_growth_rate_from_series([float(x) for x in prev_frac])

        # Determine outbreak flag based on threshold and semantics
        outbreak_flag = None
        if outbreak_threshold is None:
            outbreak_flag = None
        else:
            thr = float(outbreak_threshold)
            if outbreak_threshold_is_fraction is None:
                # heuristic: if threshold > 1 => treat as counts; else fraction
                use_fraction = thr <= 1.0
            else:
                use_fraction = bool(outbreak_threshold_is_fraction)
            if use_fraction:
                # need final_frac
                if final_frac is None:
                    outbreak_flag = None
                else:
                    outbreak_flag = (final_frac > thr)
            else:
                if final_count is None:
                    outbreak_flag = None
                else:
                    outbreak_flag = (final_count > thr)

        # Build metrics row. Also copy any sweep columns present on the timeseries row so merging / plotting are easy
        metrics_row = dict(
            variant_name=vname,
            base_variant=bname,
            run_number=int(run_num),
            series_length=int(T),
            final_size_count=int(final_count) if final_count is not None else None,
            final_size_frac=float(final_frac) if final_frac is not None else None,
            peak_prevalence_count=int(peak_prevalence_count) if peak_prevalence_count is not None else None,
            peak_prevalence_frac=float(peak_prevalence_frac) if peak_prevalence_frac is not None else None,
            peak_time=int(peak_time) if peak_time is not None else None,
            auc_prevalence_frac=float(auc) if auc is not None else None,
            duration_above_frac=int(duration_above) if duration_above is not None else None,
            early_growth_slope=float(growth_slope) if growth_slope is not None else None,
            early_growth_doubling=float(growth_doubling) if growth_doubling is not None else None,
            outbreak=bool(outbreak_flag) if outbreak_flag is not None else None,
        )

        # include any other sweep param columns present in row (non-series, primitives)
        # Avoid adding list-valued series columns again
        for k, v in row.items():
            if k in metrics_row:
                continue
            if k in {
                "prevalence_count_series",
                "prevalence_frac_series",
                "cumulative_infections_count_series",
                "cumulative_infections_frac_series",
            }:
                continue
            # only add primitive types (str/number/bool)
            if isinstance(v, (str, int, float, bool, type(None), np.integer, np.floating)):
                metrics_row[k] = v

        rows.append(metrics_row)

    metrics_df = pd.DataFrame(rows)
    return metrics_df


def plot_metric_box(
    metrics_df: pd.DataFrame,
    metric: str,
    *,
    by: str = "base_variant",
    figsize: Tuple[float, float] = (8, 5),
    show_points: bool = True,
    log_scale: bool = False,
    ax=None,
):
    """
    Boxplot (with optional overlayed jittered points) of a numeric metric grouped by a categorical column.

    Returns (fig, ax).
    """
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not present in metrics_df columns")

    groups = []
    group_labels = []
    if by in metrics_df.columns:
        vals = metrics_df[by].fillna("NA")
        unique_groups = sorted(vals.unique(), key=lambda x: str(x))
        for g in unique_groups:
            gvals = metrics_df.loc[vals == g, metric].dropna().astype(float).values
            groups.append(gvals)
            group_labels.append(str(g))
    else:
        # single group
        gvals = metrics_df[metric].dropna().astype(float).values
        groups = [gvals]
        group_labels = ["all"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()

    # Draw boxplot
    box = ax.boxplot(
        groups,
        labels=group_labels,
        patch_artist=True,
        showmeans=True,
        widths=0.6,
    )
    # Style the boxes
    for patch in box["boxes"]:
        patch.set_facecolor("#cfe2f3")
        patch.set_edgecolor("#00274C")
        patch.set_alpha(0.9)

    # Overlay points
    if show_points:
        rng = np.random.default_rng(12345)
        for i, vals in enumerate(groups):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full_like(vals, i + 1.0) + jitter, vals, color="#00274C", alpha=0.45, s=12)

    ax.set_xlabel(by)
    ax.set_ylabel(metric)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    return fig, ax


def plot_outbreak_probability_by_param(
    metrics_df: pd.DataFrame,
    param: str,
    *,
    bins: Optional[int] = 10,
    param_bins: Optional[Iterable] = None,
    outbreak_col: str = "outbreak",
    min_count_per_bin: int = 2,
    figsize: Tuple[float, float] = (8, 4),
    ax=None,
):
    """
    Plot the outbreak probability (mean(outbreak_col)) as a function of a parameter.
    If the parameter is categorical / low-cardinality, show outbreak probability per category.
    If numeric, bin into `bins` (or use param_bins) and plot mean +/- binomial CI (approx).
    Returns (fig, ax, summary_df) where summary_df contains columns: bin_label, param_mean, p_outbreak, n.
    """
    if param not in metrics_df.columns:
        raise ValueError(f"Parameter column '{param}' not found in metrics_df")

    df = metrics_df[[param, outbreak_col]].dropna(subset=[param]).copy()
    if df.empty:
        raise RuntimeError("No data for plotting")

    numeric = pd.api.types.is_numeric_dtype(df[param])
    if numeric and (param_bins is None):
        unique_count = df[param].nunique()
        if unique_count <= 10:
            # treat as categorical
            df["bin_label"] = df[param].astype(str)
            group_col = "bin_label"
        else:
            # numeric -> bin
            df["bin_label"] = pd.cut(df[param], bins=bins)
            group_col = "bin_label"
    elif numeric and param_bins is not None:
        df["bin_label"] = pd.cut(df[param], bins=param_bins)
        group_col = "bin_label"
    else:
        # categorical / string
        df["bin_label"] = df[param].astype(str)
        group_col = "bin_label"

    summary = df.groupby(group_col).agg(
        n=("bin_label", "size"), p_outbreak=(outbreak_col, lambda s: float(np.nanmean(s.astype(float))))
    )
    # compute mean parameter per bin for ordering if numeric
    try:
        param_mean = df.groupby(group_col)[param].mean()
        summary["param_mean"] = param_mean
    except Exception:
        summary["param_mean"] = np.nan

    summary = summary.reset_index().sort_values("param_mean" if "param_mean" in summary.columns and summary["param_mean"].notna().any() else group_col)
    # drop bins with very small n to reduce noise
    summary_filtered = summary[summary["n"] >= min_count_per_bin].copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()

    ax.plot(summary_filtered["param_mean"].values, summary_filtered["p_outbreak"].values, marker="o", linestyle="-", color="#DB130D")
    ax.set_xlabel(param)
    ax.set_ylabel("Outbreak probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

    return fig, ax, summary.reset_index(drop=True)


def plot_heatmap_metric(
    metrics_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str,
    *,
    x_bins: Optional[int] = 8,
    y_bins: Optional[int] = 8,
    agg: str = "mean",
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (6, 5),
    ax=None,
):
    """
    Create a 2D heatmap of the aggregated metric over bins of param_x and param_y.
    If the parameters are low-cardinality categorical variables, they will be used directly.
    Returns (fig, ax, pivot_df)
    pivot_df is the pivot table used to draw the heatmap (index=y, columns=x).
    """
    for c in (param_x, param_y, metric):
        if c not in metrics_df.columns:
            raise ValueError(f"Column '{c}' not found in metrics_df")

    df = metrics_df[[param_x, param_y, metric]].dropna().copy()

    # bin numeric parameters if specified
    def _bin_col(series, bins):
        if pd.api.types.is_numeric_dtype(series):
            if bins is None:
                return series
            if isinstance(bins, int):
                edges = np.linspace(series.min(), series.max(), bins + 1)
                return pd.cut(series, bins=edges)
            else:
                return pd.cut(series, bins=bins)
        else:
            return series.astype(str)

    df["x_bin"] = _bin_col(df[param_x], x_bins)
    df["y_bin"] = _bin_col(df[param_y], y_bins)

    pivot = df.pivot_table(index="y_bin", columns="x_bin", values=metric, aggfunc=agg)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()

    im = ax.imshow(pivot.values.astype(float), aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{agg}({metric})")
    ax.set_title(f"{agg} {metric} by {param_x} x {param_y}")
    fig.tight_layout()
    return fig, ax, pivot
