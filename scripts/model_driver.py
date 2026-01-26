"""
scripts/model_driver.py

Driver utilities to prepare contacts data and run NetworkModel instances

Functions:
- prepare_contacts(county, data_dir, save_files): returns contacts_df 
- run_single_model(contacts_df, params, algorithm_map = None, factory_map = None, seed = None, results_dir = None, save_exposures = False, plot = False)
    -> creates NetworkModel, registers LHD algorithms/actions, simulates, and returns model

- run_variants(contacts_df, base_params, variants, run_dir, base_seed = None)
    -> creates variant/replicate runs, returns list of results groups
"""
import os
import json
import copy
from copy import deepcopy
import itertools
from typing import Any, Dict, List, Tuple, Iterable, Callable, Optional
import numpy as np
import pandas as pd
import warnings

from scripts.synth_data_processing import synthetic_data_process
from scripts.fred_fetch import downloadPopData
from scripts.network_model import NetworkModel

def prepare_contacts(county: str, state: str, data_dir: str = "data", overwrite_files: bool = True, save_files: bool = False):
    """
    Ensures contacts dataframe exists 'county'
    - Looks for folder or zip under data_dir whos name starts with county
    - If none found, uses downloadPopData
    - If a .zip is found or produced by download, calls synthetic_data_process to make contacts.parquet
    - If there is a directory, look for contacts.parquet and load it, otherwise call synthetic_data_process

    """

    county_name = county.lower().capitalize()
    state_name = state.lower().capitalize() 

    os.makedirs(data_dir, exist_ok=True)



    #find any entries in data_dir that start with county_name
    entries = [f for f in os.listdir(data_dir) if f.startswith(county_name)]
    if not entries:
        downloadPopData(state=state_name, county=county_name)
        # re-list
        entries = [f for f in os.listdir(data_dir) if f.startswith(county_name)]
        if not entries:
            raise RuntimeError(f"downloadPopData did not produce any data files for {county_name} in {data_dir}")

    #prefer a directory if present, else take zip or folder
    dir_entries = [e for e in entries if os.path.isdir(os.path.join(data_dir, e))]
    if dir_entries:
        countyfoldername = dir_entries[0]
    else:
        countyfoldername = entries[0]

    countyfolder = os.path.join(data_dir, countyfoldername)

    #If data is a zipfile, use synthetic_data_process to extract contact data
    #delete zip file
    if os.path.isfile(countyfolder) and countyfolder.lower().endswith(".zip"):
        try:
            contacts_df = synthetic_data_process(county)
        except Exception as e:
            raise RuntimeError(f"Failed to process synthetic data from zip for {county_name}: {e}")from e
        
        if save_files and os.path.exists(countyfolder):
            try:
                os.remove(countyfolder)
            except OSError:
                pass

        return contacts_df

    #if countyfolder is a directory, check for contacts.parquet, if not there, process synthetic data
    if os.path.isdir(countyfolder):
        parquet_path = os.path.join(countyfolder, "contacts.parquet")
        #if overwrite_files, do not reread old contacts.parquet
        if os.path.exists(parquet_path) and not overwrite_files:
            return pd.read_parquet(parquet_path)
        else:
            try:
                contacts_df = synthetic_data_process(county_name, save_files = save_files)
                return contacts_df
            except Exception as e:
                raise RuntimeError(f"Could not find contacts.parquet in {countyfolder} and synthetic_data_process failed: {e}") from e

    raise RuntimeError(f"Unhandled data entry for county {county_name}: {countyfolder}")


def run_single_model(
    contacts_df: pd.DataFrame,
    params: Dict,
    algorithm_map: Optional[Dict[str, object]] = None,
    factory_map: Optional[Dict[str, callable]] = None,
    seed:Optional[int] = None,
    results_dir: Optional[str] = None,
    save_exposures: bool = False
) -> NetworkModel:
    """
    Instantiates a NetworkModel with contacts_df and params, registers algorithms and actions for LHD, runs simulate, and returns model for analysis
    """

    params_copy = deepcopy(params)
    if seed is not None:
        params_copy["seed"] = int(seed)

    #create a generator and pass it to the model, that way everything is deterministic including initialization
    rng = np.random.default_rng(int(seed)) if seed is not None else None
    
    #create model
    model = NetworkModel(
        contacts_df = contacts_df, 
        params = params_copy,
        rng = rng, 
        results_folder = results_dir,
        lhd_register_defaults = False, #only use provided actions for LHDs run by driver
        lhd_algorithm_map = algorithm_map,
        lhd_action_factory_map = factory_map
        )

    model.simulate()

    #Optionally save exposures
    if save_exposures and results_dir:
        np.savez_compressed(os.path.join(results_dir, "exposure_event_log.npz"), *model.exposure_event_log)

        return model
       


def run_variants(
    contacts_df: pd.DataFrame,
    base_params: Dict,
    variants: List[Dict],
    run_dir: str,
    base_seed: Optional[int] = None
) -> List[Dict]:
    """
        Run a list of model variants. Each variant is a dict with keys:
        - 'name' - string name of the variant
        - 'param_overrides' - dict of parameters to change and new values
        - algorithm_map' - dict of action_type -> Algorithm
        - 'factory_map' - dict action_type -> factory
        - 'save_exposures'

        Returns a list of dicts, one per variant, containing model instance and path
    """
    os.makedirs(run_dir, exist_ok = True)
    base_seed = int(base_seed) if base_seed is not None else int(base_params.get("seed", 0))

    variant_results = []
    for v_ind, variant in enumerate(variants):
        name = variant.get("name", f"variant_{v_ind}")
        var_dir = os.path.join(run_dir, f"{name}")
        os.makedirs(var_dir, exist_ok = True)

        #update seed for each variant, update param changes
        if base_seed is not None:
            seed = base_seed + v_ind*1000
            
        params = deepcopy(base_params)
        params.update(variant.get("param_overrides"))
        params["seed"] = int(seed)

        model = run_single_model(
            contacts_df = contacts_df,
            params = params,
            algorithm_map = variant.get("algorithm_map"),
            factory_map = variant.get("factory_map"),
            seed = seed,
            results_dir = var_dir,
            save_exposures = variant.get("save_exposures", False)
        )
        if model is None:
            raise RuntimeError("run_single_model() did not return a NetworkModel instance")

        #save summary and action log
        try: 
            summary = model.epi_outcomes()
            summary.to_csv(os.path.join(var_dir, "run_summary.csv"), index = False)
        except Exception as e:
            warnings.warn(f"Could not save summary for variant {name}: {e}")

        try:
            with open(os.path.join(var_dir, "action_log.json"), "w") as f:
                json.dump(model.lhd.action_log, f, default= str, indent = 4)
        except Exception as e:
            warnings.warn(f"Could not save action log: {e}")
        
    variant_results.append({
        "variant_name": name,
        "variant__dir": var_dir,
        "model": model
    })

    return variant_results


#Helper function to make parameter sweeps
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


