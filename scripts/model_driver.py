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

from copy import deepcopy
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import re
import warnings

from scripts.synth_data_processing import synthetic_data_process
from scripts.fred_fetch import downloadPopData
from scripts.network_model import NetworkModel
from scripts.analysis_tools import _try_parse_value, _parse_variant_name

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
    combined_action_log = []

    for v_ind, variant in enumerate(variants):
        name = variant.get("name", f"variant_{v_ind}")
        # do not create a variant subfolder; keep results at run_dir

        #update seed for each variant, update param changes
        if base_seed is not None:
            seed = base_seed + v_ind*1000
            
        params = deepcopy(base_params)
        params.update(variant.get("param_overrides"))
        params["seed"] = int(seed)

        if len(variants) < 10:
            print("Running model for variant: {name}")
        model = run_single_model(
            contacts_df = contacts_df,
            params = params,
            algorithm_map = variant.get("algorithm_map"),
            factory_map = variant.get("factory_map"),
            seed = seed,
            results_dir = None, #for variant runs, don't write per-variant file
            save_exposures = False
        )
        if model is None:
            raise RuntimeError("run_single_model() did not return a NetworkModel instance")

        #optionally save exposures per-variant to run_dir
        if variant.get("save_exposures", False):
            try:
                safe_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", name)
                exposure_fname = os.path.join(run_dir, f"exposure_event_log_{safe_name}.npz")
                np.savez_compressed(exposure_fname, *model.exposure_event_log)
            except Exception as e:
                warnings.warn(f"Could not save exposures for variant {name}: {e}")

        #append model results variant_results
        variant_results.append({
            "variant_name": name,
            "model": model
        })

        #Collect LHD action logs recorded per run in model.all_lhd_action_logs
        base_variant, swept_params = _parse_variant_name(name, name_sep="__")
        all_lhd_logs = getattr(model, "all_lhd_action_logs", []) or []
        for run_number, run_log in enumerate(all_lhd_logs):
            if not run_log:
                continue
            for action_entry in run_log:
                # copy and annotate with metadata
                try:
                    entry = dict(action_entry)
                except Exception:
                    entry = {"raw": str(action_entry)}
                entry["variant_name"] = name
                entry["base_variant"] = base_variant
                entry["run_number"] = int(run_number)
                # attach sweep params (if any) as top-level fields
                for k, v in swept_params.items():
                    entry[k] = v
                combined_action_log.append(entry)

        print(f"Run complete for Variant '{name}")

    #Save combined action log to run_dir/action_log.json
    logs = combined_action_log

    if logs:
        df_actions = pd.json_normalize(logs, sep = '_')

        #make lists strings to keep csv normal
        def _stringify_cell(x):
            if isinstance(x, (list, dict, np.ndarray)):
                return json.dumps(x, default = str)
            return x
        
        for col in df_actions.columns:
            if df_actions[col].apply(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
                df_actions[col] = df_actions[col].apply(_stringify_cell)
            
        order = [
            'base_variant', 'variant_name', 'run_number',
            'time', 'action_id', 'action_type', 'kind',
            'nodes_count', 'hours_used', 'duration',
            'reversible_tokens', 'nonreversible_tokens'
        ]

        cols = [c for c in order if c in df_actions.columns] + [c for c in df_actions.columns if c not in order]
        df_actions = df_actions[cols]

        csv_path = os.path.join(run_dir, "action_log.csv")
        df_actions.to_csv(csv_path, index = False, encoding = 'utf-8')

    else:
        empty_cols = ['variant_name','base_variant','run_number','time','action_id','action_type','kind','nodes_count','hours_used','duration','reversible_tokens','nonreversible_tokens']
        pd.DataFrame(columns=empty_cols).to_csv(os.path.join(run_dir, "action_log.csv"), index=False)
   
    return variant_results
