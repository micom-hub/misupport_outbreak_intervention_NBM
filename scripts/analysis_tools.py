from typing import Dict, List, Optional, Tuple, Any, Callable, Iterable
import itertools
import copy
import warnings
import numpy as np

#Create parameter sweeps
def generate_sweep_variants(
    variants_list: List[Dict[str, Any]],
    sweep_spec: Dict[str, Tuple[float, float, int]],
    *,
    interior_points: bool = True,
    include_base: bool = False,
    group_by_base: bool = False,
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
    -group_by_base if True returns a dict mapping base_name -> [variants], otherwise returns a list of variants
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

#For a model run with a parameter sweep, reaggregate variants by 
