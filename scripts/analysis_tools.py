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

#Create straightforward parameter sweeps
#variant for each combination of parameters
def generate_sweep_variants(
    variants_list: List[Dict[str, Any]],
    sweep_spec: Dict[str, Tuple[float, float, int]],
    *,
    interior_points: bool = True,
    include_base: bool = False,
    name_sep: str = "__",
    value_formatter: Optional[Callable[[Any], str]] = None,
    int_keys: Optional[Iterable[str]] = None,
    warn_threshold: int = 2000
) -> List[Dict[str, Any]]:
    """
Builds sweep variants, taking main variants defined above, and returning them with systematically altered parameter values

Args:
    variants_list: list of variant dicts following template above

    sweep_spec: mapping_param_name -> (min_value, max_value, n)
    - if interior_points = True, n is number of points between min and max
        else, n is the total number of points
    -include_base: if True, each base variant is included and unchanged
    - name_sep: string to join name pieces (base + param=value)
    - value_formatter: optional callable to format values
    -int_keys: optional iterable of parameter names to be cast to int
    - warn_threshold: if total generated variants exceeds threshold, emit warning

Returns:
    - a flat list of variant dicts with the same keys as input variant dict
    """
    if not isinstance(variants_list, list):
        raise TypeError("variants_list must be a list of variant dicts")

    if not sweep_spec:
        return copy.deepcopy(variants_list)

    #build grid for each sweep key
    value_arrays: Dict[str, np.ndarray] = {}
    for key, spec in sweep_spec.items():
        if not isinstance(spec, (list, tuple)) and len(spec == 3):
            raise ValueError(f"sweep_spec[{key}] must be tuple/list (min, max, n_inbetween)")
        mn, mx, n_inbetween = spec
        if interior_points:
            num_points = int(n_inbetween) + 2
        else: 
            num_points = int(n_inbetween)
        if num_points <= 0:
            raise ValueError(f"number of points for '{key}' must be >= 1")
        if num_points == 1:
            vals = np.array([float(mn)])
        else:
            vals = np.linspace(float(mn), float(mx), num = num_points)
        value_arrays[key] = vals

    sweep_keys = list(value_arrays.keys())
    grids = [value_arrays[k] for k in sweep_keys]
    combos = list(itertools.product(*grids))
    total_generated_per_base = len(combos)
    total_out = total_generated_per_base * max(1, len(variants_list))
    if total_out > warn_threshold:
        warnings.warn(f"Sweeping will produce {total_out} variants. This may be large.")
    
    #default value formatter
    if value_formatter is None:
        def _fmt_val(x):
            #integers as ints, floats compact
            try:
                if isinstance(x, (int, np.integer)):
                    return str(int(x))
                f = float(x)
                #if near-integer and not requested as int, format as float
                return f"{f:.6g}"
            except Exception:
                return str(x)
        fmt = _fmt_val
    else:
        fmt = value_formatter
    
    int_keys = set(int_keys) if int_keys is not None else set()

    out_variants: List[Dict[str, Any]] = []
    for base in variants_list:
        if not isinstance(base, dict):
            raise TypeError("Each element of variants_list must be a dict")

        base_copy = copy.deepcopy(base)
        base_name = str(base_copy.get("name", "variant"))
        base_param_overrides = copy.deepcopy(base_copy.get("param_overrides", {}) or {})

        if include_base:
            out_variants.append(base_copy)

        for combo in combos:
            new_variant = copy.deepcopy(base_copy)

            #start from base param_overrides if any
            new_param_overrides = copy.deepcopy(base_param_overrides)
            sweep_values_for_naming = {}
            for k, raw_val in zip(sweep_keys, combo):
                if k in int_keys:
                    val = int(round(float(raw_val)))
                else:
                    if isinstance(raw_val,(np.floating, np.float32, np.float64)):
                        val = float(raw_val)
                    elif isinstance(raw_val, (np.integer, np.int32, np.int64)):
                        val = int(raw_val)
                    else:
                        val = raw_val
                new_param_overrides[k] = val
                sweep_values_for_naming[k] = val

            new_variant["param_overrides"] = new_param_overrides

            #build descriptive name
            suffix = name_sep.join([f"{k}={fmt(sweep_values_for_naming[k])}" for k in sweep_keys])
            new_variant["name"] = base_name + name_sep + suffix if suffix else base_name

            out_variants.append(new_variant)

    return out_variants


#create LHS parameter sweeps
#For a model run with a parameter sweep, reaggregate variants by parameter sweep
def generate_lhs_variants(
    variants_list: List[Dict[str, Any]],
    param_ranges: Dict[str, Tuple[float, float]],
    n: int,
    *,
    name_sep: str = "__",
    value_formatter: Optional[Callable[[Any], str]] = None,
    int_keys: Optional[Iterable[str]] = None,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate variant dicts using Latin-Hypercube-style sampling with n x variants replicates

    Args:
      variants_list: list of base variant dicts (same format as generate_sweep_variants input).
      param_ranges: dict mapping param_name -> (min_value, max_value).
      n: number of LHS samples per parameter (same N for all parameters).
      name_sep: separator used when building generated variant names (default "__").
      value_formatter: optional callable(value)->str used for name formatting.
      int_keys: iterable of parameter names that should be cast to int in overrides.
      seed: optional RNG seed for reproducibility.

    Returns:
      List[Dict[str, Any]]: flattened list of generated variant dicts (each has updated 'param_overrides' and 'name').
    """
 # Type validations
    if not isinstance(variants_list, list):
        raise TypeError("variants_list must be a list of variant dicts")
    if not isinstance(param_ranges, dict):
        raise TypeError("param_ranges must be a dict mapping param -> (min, max)")
    n = int(n)
    if n <= 0:
        raise ValueError("n must be >= 1")

    if not param_ranges:
        return copy.deepcopy(variants_list)

# Preserve param order from param_ranges dict
    sweep_keys = list(param_ranges.keys())

    # Print total variants that will be generated
    total_out = n * max(1, len(variants_list))
    print(f"Generating {total_out} variants ({len(variants_list)} base variant(s) x {n} LHS samples each)")

    rng = np.random.default_rng(seed)


    #build a jittered strata baseline [0, 1) then permute to param range
    base = (np.arange(n) + rng.random(n)) / n  # shape (n,)

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

    #format names
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

    #use values to produce variants
    out_variants: List[Dict[str, Any]] = []
    for base_variant in variants_list:
        if not isinstance(base_variant, dict):
            raise TypeError("Each element of variants_list must be a dict")
        base_copy = copy.deepcopy(base_variant)
        base_name = str(base_copy.get("name", "variant"))
        base_param_overrides = copy.deepcopy(base_copy.get("param_overrides", {}) or {})

        for i in range(n):
            new_variant = copy.deepcopy(base_copy)
            new_param_overrides = copy.deepcopy(base_param_overrides)
            sweep_values_for_naming: Dict[str, Any] = {}

            for k in sweep_keys:
                raw_val = value_arrays[k][i]
                if k in int_keys:
                    val = int(round(float(raw_val)))
                else:
                    # convert numpy scalars to python types
                    if isinstance(raw_val, (np.floating, np.float32, np.float64)):
                        val = float(raw_val)
                    elif isinstance(raw_val, (np.integer, np.int32, np.int64)):
                        val = int(raw_val)
                    else:
                        val = raw_val
                new_param_overrides[k] = val
                sweep_values_for_naming[k] = val

            new_variant["param_overrides"] = new_param_overrides

            # build descriptive name using param order from param_ranges
            suffix = name_sep.join([f"{k}={fmt(sweep_values_for_naming[k])}" for k in sweep_keys])
            new_variant["name"] = base_name + name_sep + suffix if suffix else base_name

            out_variants.append(new_variant)

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
            epi_df = model.epi_outcomes()
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

#Extract time series data from models
def disease_over_time(
    variant_results: List[Dict[str, Any]],
    *,
    name_sep: str = "__",
    return_cumulative_incidence: bool = False,
    return_long: bool = False,
    day_prefix: str = "day_",
    pad_with_last: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build prevalence time-series (and optional cumulative-incidence series) per model run. Return in long format or not

    Args:
        variant_results: list of dicts each containing:
            - 'variant_name' (str)
            - 'model' (NetworkModel instance, already simulated)
        name_sep: separator used in variant names (default "__")
        return_cumulative_incidence: if True, return a second DataFrame with cumulative incidence
        return_long: if True, return long (tidy) DataFrame(s) with one row per run per timepoint
        day_prefix: prefix for timestep columns in wide format (default "day_")
        pad_with_last: whether to pad shorter series with last observed value (True) or NaN (False)

    Returns:
        If return_cumulative_incidence is False:
            - wide (or long if return_long=True) prevalence DataFrame
        Else:
            - tuple (prevalence_df, cumulative_incidence_df) in the chosen orientation (wide/long)
    """
    rows = []
    cum_rows = []
    max_t = 0
    parsed_param_keys = set()

    # First pass: determine max length and collect param keys
    for entry in variant_results:
        vname = entry.get("variant_name") or entry.get("name")
        model = entry.get("model", None)
        if model is None:
            continue
        base_name, params = _parse_variant_name(vname, name_sep=name_sep)
        parsed_param_keys.update(params.keys())

        # determine longest saved timeseries length among runs of this model
        for run in range(model.n_runs):
            states = model.all_states_over_time[run]
            if states is None:
                continue
            L = len(states)
            if L > max_t:
                max_t = L

    # prepare day column names
    day_cols = [f"{day_prefix}{t}" for t in range(max_t)]

    # Second pass: build per-run records
    for entry in variant_results:
        vname = entry.get("variant_name") or entry.get("name")
        model = entry.get("model", None)
        if model is None:
            continue

        base_name, params = _parse_variant_name(vname, name_sep=name_sep)

        for run in range(model.n_runs):
            # Meta
            row = {
                "variant_name": vname,
                "base_variant": base_name,
                "run_number": int(run)
            }
            # attach parsed sweep params
            for k in sorted(parsed_param_keys):
                row[k] = params.get(k, None)

            # compute prevalence series
            states_over_time = model.all_states_over_time[run]
            if states_over_time is None or len(states_over_time) == 0:
                prevalence = np.full(max_t, np.nan)
            else:
                N = model.N
                preval_list = [len(state[2]) / float(N) if N > 0 else np.nan for state in states_over_time]
                if len(preval_list) < max_t:
                    if pad_with_last:
                        last_val = preval_list[-1] if preval_list else np.nan
                        preval_list = preval_list + [last_val] * (max_t - len(preval_list))
                    else:
                        preval_list = preval_list + [np.nan] * (max_t - len(preval_list))
                else:
                    preval_list = preval_list[:max_t]
                prevalence = np.array(preval_list, dtype=float)

            # cumulative incidence if requested
            if return_cumulative_incidence:
                exposures = model.all_new_exposures[run]
                # treat exposures[0] empty as imported seeds
                if len(exposures) > 0 and (hasattr(exposures[0], "__len__") and len(exposures[0]) == 0):
                    exposures = list(exposures)
                    exposures[0] = np.array(model.params.get("I0", []), dtype=int)
                ever = np.zeros(model.N, dtype=bool)
                cumul_list = []
                for t in range(len(exposures)):
                    new = np.array(exposures[t], dtype=int) if len(exposures[t]) > 0 else np.array([], dtype=int)
                    if new.size:
                        ever[new] = True
                    cumul_list.append(ever.sum() / float(model.N) if model.N > 0 else np.nan)
                if len(cumul_list) < max_t:
                    if pad_with_last:
                        last_c = cumul_list[-1] if cumul_list else 0.0
                        cumul_list = cumul_list + [last_c] * (max_t - len(cumul_list))
                    else:
                        cumul_list = cumul_list + [np.nan] * (max_t - len(cumul_list))
                else:
                    cumul_list = cumul_list[:max_t]
                cumulative = np.array(cumul_list, dtype=float)
            else:
                cumulative = None

            # attach day columns (wide)
            for idx, col in enumerate(day_cols):
                row[col] = float(prevalence[idx]) if prevalence is not None else np.nan

            rows.append(row)

            if return_cumulative_incidence:
                crow = {
                    "variant_name": vname,
                    "base_variant": base_name,
                    "run_number": int(run)
                }
                for k in sorted(parsed_param_keys):
                    crow[k] = params.get(k, None)
                for idx, col in enumerate(day_cols):
                    crow[col] = float(cumulative[idx]) if cumulative is not None else np.nan
                cum_rows.append(crow)

    # Build wide DataFrames
    meta_cols = ["variant_name", "base_variant", "run_number"]
    param_cols = sorted(parsed_param_keys)
    wide_cols = meta_cols + param_cols + day_cols

    if rows:
        prevalence_df = pd.DataFrame(rows)
        # ensure all expected columns exist (fill missing with NaN)
        for c in wide_cols:
            if c not in prevalence_df.columns:
                prevalence_df[c] = np.nan
        prevalence_df = prevalence_df[wide_cols]
    else:
        prevalence_df = pd.DataFrame(columns=wide_cols)

    if not return_cumulative_incidence:
        # possibly return long
        if return_long:
            id_vars = meta_cols + param_cols
            long = prevalence_df.melt(id_vars=id_vars, value_vars=day_cols, var_name="time", value_name="prevalence")
            # convert time to integer index (strip prefix)
            long["time"] = long["time"].str[len(day_prefix):].astype(int)
            # reorder columns
            long = long[id_vars + ["time", "prevalence"]]
            return long
        else:
            return prevalence_df
    else:
        # build cumulative wide
        if cum_rows:
            cumulative_df = pd.DataFrame(cum_rows)
            for c in wide_cols:
                if c not in cumulative_df.columns:
                    cumulative_df[c] = np.nan
            cumulative_df = cumulative_df[wide_cols]
        else:
            cumulative_df = pd.DataFrame(columns=wide_cols)

        if return_long:
            id_vars = meta_cols + param_cols
            prevalence_long = prevalence_df.melt(id_vars=id_vars, value_vars=day_cols, var_name="time", value_name="prevalence")
            prevalence_long["time"] = prevalence_long["time"].str[len(day_prefix):].astype(int)
            prevalence_long = prevalence_long[id_vars + ["time", "prevalence"]]

            cumulative_long = cumulative_df.melt(id_vars=id_vars, value_vars=day_cols, var_name="time", value_name="cumulative_incidence")
            cumulative_long["time"] = cumulative_long["time"].str[len(day_prefix):].astype(int)
            cumulative_long = cumulative_long[id_vars + ["time", "cumulative_incidence"]]

            return prevalence_long, cumulative_long
        else:
            return prevalence_df, cumulative_df

def plot_epi_series(
    df_long: pd.DataFrame,
    type: str = "prevalence",
    spaghetti: bool = False,
    frac: bool = True,
    population: float | None = None,
    fixed_axis: bool = True,
    main_title: str | None = None,
    outbreak_threshold: Optional[float] = None,
    outbreak_label_fmt: str | None = None,
    outbreak_show_N_in_title: bool = True,
    time_col: str = "time",
    value_col: str | None = None,
    run_col: str = "run_number",
    variant_col: str = "variant_name",
    facet_col: str = "base_variant",
    collapse_func: str = "mean",
    ncols: int = 3,
    spaghetti_color: str = "#00274C",
    spaghetti_alpha: float = 0.22,
    mean_line_color: str = "#00274C",
    mean_line_width: float = 2.2,
    median_color: str = "#DB130D",
    median_width: float = 2.8,
    ci50_color: str = "#00283B",
    ci90_color: str = "#0584FA",
    ci95_color: str = "#8CC6FD",
    band_alpha: float = 0.8,
    figsize_per_facet: tuple[float, float] = (5.2, 3.6),
):
    """
    Plot prevalence/incidence series from a long-format DataFrame.

    Improvements from prior version:
    - Correct grid indexing so facets map to subplot positions robustly.
    - Optional two-row layout when outbreak_threshold is provided: top row = All Runs, bottom row = Outbreak Runs.
    - Row labels ("All Runs", "Outbreak Runs...") placed in the left margin, closer to plots.
    - "Number of Runs: N" label placed under each subplot, avoiding overlap with x-axis label.
    """
    type = type.lower().strip()
    if type not in {"prevalence", "incidence"}:
        raise ValueError('type must be "prevalence" or "incidence"')

    if (not frac) and (population is None):
        raise ValueError("population (scalar) is required when frac=False")

    # infer value_col when not provided
    if value_col is None:
        if type == "prevalence":
            if "prevalence" not in df_long.columns:
                raise ValueError('Expected a "prevalence" column for type="prevalence"')
            value_col = "prevalence"
        else:
            if not frac:
                if "cumulative_incidence" not in df_long.columns and "cumulative_cases" not in df_long.columns:
                    raise ValueError('frac=False requires either "cumulative_incidence" or "cumulative_cases" column')
                value_col = "cumulative_incidence" if "cumulative_incidence" in df_long.columns else "cumulative_cases"
            else:
                if "incidence" in df_long.columns:
                    value_col = "incidence"
                elif "cumulative_incidence" in df_long.columns:
                    value_col = "cumulative_incidence"
                else:
                    raise ValueError('Expected "incidence" or "cumulative_incidence" for type="incidence"')

    # required columns
    for c in (time_col, value_col, run_col, variant_col):
        if c not in df_long.columns:
            raise ValueError(f"Missing required column: {c}")

    # facet list (base variants)
    facets = pd.Series(df_long[facet_col].dropna().unique()).sort_values().tolist() if facet_col in df_long.columns else [None]
    n_facets = len(facets)

    # outbreak_threshold handling -> two-row layout if requested
    create_two_rows = outbreak_threshold is not None
    if create_two_rows:
        ncols_eff = n_facets if n_facets > 0 else 1
        nrows_eff = 2
    else:
        ncols_eff = min(ncols, n_facets) if n_facets > 0 else 1
        nrows_eff = int(np.ceil(n_facets / ncols_eff)) if n_facets > 0 else 1

    # create subplots grid
    fig, axes = plt.subplots(nrows_eff, ncols_eff,
                             figsize=(figsize_per_facet[0] * ncols_eff, figsize_per_facet[1] * nrows_eff),
                             squeeze=False)
    axes = np.array(axes).reshape(nrows_eff, ncols_eff)

    # axis labels
    x_label = "Day" if time_col.lower() in {"day", "t", "time"} else time_col
    is_cum = ("cumulative" in str(value_col).lower())
    if type == "prevalence":
        y_label = "Prevalence" if frac else "Infected (count)"
    else:
        y_label = "Cumulative cases" if not frac else ("Cumulative incidence" if is_cum else "Incidence (new infections / day)")

    # auto main title
    if main_title is None:
        if type == "prevalence":
            main_title = "Prevalence over time" if frac else "Number infected over time"
        else:
            if not frac:
                main_title = "Cumulative cases over time"
            else:
                main_title = "Cumulative incidence over time" if is_cum else "Incidence over time"

    # Pre-compute final cases per (variant, run) if outbreak filter requested
    final_df = None
    outbreak_pairs = set()
    if create_two_rows:
        if "cumulative_cases" in df_long.columns:
            final_df = df_long.groupby([variant_col, run_col], as_index=False)["cumulative_cases"].max().rename(columns={"cumulative_cases": "final_cases"})
        elif "cumulative_incidence" in df_long.columns:
            if outbreak_threshold > 1 and population is None:
                raise ValueError("outbreak_threshold > 1 requires passing population to convert fractions to counts")
            scale = population if outbreak_threshold > 1 else 1.0
            tmp = df_long.groupby([variant_col, run_col], as_index=False)["cumulative_incidence"].max()
            tmp["final_cases"] = tmp["cumulative_incidence"] * scale
            final_df = tmp[[variant_col, run_col, "final_cases"]]
        else:
            raise ValueError("outbreak_threshold requires 'cumulative_cases' or 'cumulative_incidence' in df_long")

        outbreak_pairs = set(tuple(x) for x in final_df.loc[final_df["final_cases"] > float(outbreak_threshold), [variant_col, run_col]].to_records(index=False))

    # track N counts and per-row maxima
    top_counts = []
    bot_counts = []
    top_row_max = 0.0
    bottom_row_max = 0.0

    # helper to plot on an axis and return local_max and total_runs
    def _plot_for_facet(ax, sub_all_df):
        """
        Plot agg on provided axis, return (local_max, total_runs)
        """
        sub = sub_all_df[[time_col, value_col, variant_col, run_col]].dropna()
        sub["_plot_value"] = pd.to_numeric(sub[value_col], errors="coerce")
        sub = sub.dropna(subset=["_plot_value"])
        if not frac:
            sub["_plot_value"] = sub["_plot_value"] * float(population)

        if sub.empty:
            return 0.0, 0

        agg = sub.groupby([variant_col, run_col, time_col], as_index=False)["_plot_value"].agg(collapse_func).sort_values([variant_col, run_col, time_col])

        if spaghetti:
            for _, g in agg.groupby([variant_col, run_col], sort=False):
                ax.plot(g[time_col].to_numpy(), g["_plot_value"].to_numpy(), color=spaghetti_color, alpha=spaghetti_alpha, linewidth=0.7)
            mean_series = agg.groupby(time_col)["_plot_value"].mean().sort_index()
            if not mean_series.empty:
                ax.plot(mean_series.index.to_numpy(), mean_series.to_numpy(), color=mean_line_color, linewidth=mean_line_width)
            local_max = agg["_plot_value"].max() if not agg.empty else 0.0
        else:
            qs = agg.groupby(time_col)["_plot_value"].quantile([0.025, 0.05, 0.25, 0.50, 0.75, 0.95, 0.975]).unstack().sort_index()
            if not qs.empty:
                x = qs.index.to_numpy()
                ax.fill_between(x, qs[0.025], qs[0.975], color=ci95_color, alpha=band_alpha, linewidth=0)
                ax.fill_between(x, qs[0.05], qs[0.95], color=ci90_color, alpha=band_alpha, linewidth=0)
                ax.fill_between(x, qs[0.25], qs[0.75], color=ci50_color, alpha=band_alpha, linewidth=0)
                ax.plot(x, qs[0.50], color=median_color, linewidth=median_width)
                # safest local max across quantiles
                local_max = float(np.nanmax(qs.to_numpy()))
            else:
                local_max = 0.0

        total_runs = sub_all_df[[variant_col, run_col]].drop_duplicates().shape[0]
        return float(local_max), int(total_runs)

    # Main plotting loop: map each facet index -> grid position(s)
    for i, facet in enumerate(facets):
        if create_two_rows:
            # top axis at column i
            row_top, col_top = 0, i
            row_bot, col_bot = 1, i
        else:
            row_top, col_top = divmod(i, ncols_eff)
            row_bot, col_bot = None, None  # no bottom row

        ax_top = axes[row_top, col_top]
        if facet is None:
            sub_all = df_long.copy()
            facet_title = None
        else:
            sub_all = df_long[df_long[facet_col] == facet].copy()
            facet_title = f"{facet}"

        # Plot top
        local_max_top, total_runs = _plot_for_facet(ax_top, sub_all)
        top_counts.append(total_runs)
        top_row_max = max(top_row_max, local_max_top)
        ax_top.set_title(facet_title)
        ax_top.set_xlabel(x_label)
        ax_top.set_ylabel(y_label)
        ax_top.grid(True, alpha=0.25)
        ax_top.set_ylim(bottom=0)

        # Plot bottom if outbreak layout
        if create_two_rows:
            ax_bot = axes[row_bot, col_bot]
            # build sub_bot with outbreak pairs limited to this facet
            if outbreak_pairs:
                pairs_df = pd.DataFrame(list(outbreak_pairs), columns=[variant_col, run_col])
                facet_variants = sub_all[variant_col].drop_duplicates() if not sub_all.empty else pd.Series(dtype=object)
                valid_pairs = pairs_df[pairs_df[variant_col].isin(facet_variants)]
                if not valid_pairs.empty:
                    sub_bot = sub_all.merge(valid_pairs, on=[variant_col, run_col], how="inner")
                else:
                    sub_bot = pd.DataFrame(columns=sub_all.columns)
            else:
                sub_bot = pd.DataFrame(columns=sub_all.columns)

            local_max_bot, outbreak_count = _plot_for_facet(ax_bot, sub_bot) if not sub_bot.empty else (0.0, 0)
            bot_counts.append(outbreak_count)
            bottom_row_max = max(bottom_row_max, local_max_bot)
            ax_bot.set_xlabel(x_label)
            ax_bot.set_ylabel(y_label)
            ax_bot.grid(True, alpha=0.25)
            ax_bot.set_ylim(bottom=0)

    # Turn off any unused axes (non-two-row case)
    if not create_two_rows:
        total_slots = nrows_eff * ncols_eff
        for slot in range(n_facets, total_slots):
            r = slot // ncols_eff
            c = slot % ncols_eff
            axes[r, c].axis("off")

    # Layout and margins: leave space for left row label and N labels below plots
    left_margin = 0.11 if create_two_rows else 0.07   # keep smaller left margin than before
    bottom_margin = 0.12
    top_rect = 0.90 if not spaghetti else 0.94
    fig.tight_layout(rect=[left_margin, bottom_margin, 1.0, top_rect])

    # Row-wise fixed axes if requested
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
            lines = ax.get_lines()
            if lines:
                for ln in lines:
                    ydata = ln.get_ydata()
                    if len(ydata):
                        all_max = max(all_max, np.nanmax(ydata))
        if all_max > 0:
            ymax = all_max * 1.05
            for ax in axes.ravel():
                ax.set_ylim(0, ymax)

    # Add left-side row labels and 'Number of Runs' below each subplot.
    # Place row-labels slightly inside the left_margin (closer to axes than before)
    left_x = max(0.005, left_margin - 0.025)  # slightly left of left_margin
    if create_two_rows:
        # compute vertical centers of top and bottom rows
        top_centers = [axes[0, c].get_position() for c in range(ncols_eff)]
        bottom_centers = [axes[1, c].get_position() for c in range(ncols_eff)]
        top_row_center = float(np.mean([p.y0 + p.height / 2.0 for p in top_centers]))
        bottom_row_center = float(np.mean([p.y0 + p.height / 2.0 for p in bottom_centers]))
        fig.text(left_x, top_row_center, "All Runs", va="center", ha="center", rotation="vertical", fontsize=12, fontweight="bold")
        thr_text = outbreak_label_fmt if outbreak_label_fmt is not None else f"Outbreak Runs: Threshold (Infections > {outbreak_threshold})"
        fig.text(left_x, bottom_row_center, thr_text, va="center", ha="center", rotation="vertical", fontsize=11)

        # per-subplot N labels below each axis
        for c in range(ncols_eff):
            p_top = axes[0, c].get_position()
            x_center = p_top.x0 + p_top.width / 2.0
            y_label_pos = p_top.y0 - 0.055
            fig.text(x_center, y_label_pos, f"Number of Runs: {top_counts[c]}", ha="center", va="top", fontsize=9)

            p_bot = axes[1, c].get_position()
            x_center2 = p_bot.x0 + p_bot.width / 2.0
            y_label_pos2 = p_bot.y0 - 0.055
            fig.text(x_center2, y_label_pos2, f"Number of Runs: {bot_counts[c]}", ha="center", va="top", fontsize=9)
    else:
        # single-/multi-row grid: place N under each used axis
        for i, facet in enumerate(facets):
            r, c = divmod(i, ncols_eff)
            p = axes[r, c].get_position()
            x_center = p.x0 + p.width / 2.0
            y_label_pos = p.y0 - 0.055
            sub_all = df_long if facet is None else df_long[df_long[facet_col] == facet].copy()
            total_runs = sub_all[[variant_col, run_col]].drop_duplicates().shape[0]
            fig.text(x_center, y_label_pos, f"Number of Runs: {total_runs}", ha="center", va="top", fontsize=9)

    # Legend for fan plots (median + bands)
    if not spaghetti:
        handles = [
            Line2D([0], [0], color=median_color, lw=median_width, label="Median"),
            Patch(facecolor=ci50_color, alpha=band_alpha, label="50% band"),
            Patch(facecolor=ci90_color, alpha=band_alpha, label="90% band"),
            Patch(facecolor=ci95_color, alpha=band_alpha, label="95% band"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False)

    # Centered suptitle
    fig.suptitle(main_title, x=0.5, y=0.995, ha="center", fontsize=14, fontweight="bold")

    return fig, axes