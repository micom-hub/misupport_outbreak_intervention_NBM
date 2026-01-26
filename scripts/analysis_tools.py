from typing import Dict, List, Optional, Tuple, Any, Callable, Iterable
import itertools
import copy
import re
import warnings
import numpy as np
import pandas as pd
#Create parameter sweeps
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


    return overall_df, combined_summary




    
    
