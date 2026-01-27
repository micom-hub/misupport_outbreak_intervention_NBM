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
import concurrent.futures
from copy import deepcopy
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import re
import warnings
import traceback

from scripts.synth_data_processing import synthetic_data_process, build_edge_list
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
    contacts_df: Union[pd.DataFrame, str],
    params: Dict,
    algorithm_map: Optional[Dict[str, object]] = None,
    factory_map: Optional[Dict[str, callable]] = None,
    seed:Optional[int] = None,
    results_dir: Optional[str] = None,
    save_exposures: bool = False,
    edge_list: Optional[Union[pd.DataFrame, str]] = None
) -> NetworkModel:
    """
    Instantiates a NetworkModel with contacts_df and params, registers algorithms and actions for LHD, runs simulate, and returns model for analysis

    If contacts_df and edge_list should be dataframe objects unless being run in parallel
    """

    params_copy = deepcopy(params)
    if seed is not None:
        params_copy["seed"] = int(seed)

    #read in large dataframes if they are filepaths
    if isinstance(contacts_df, str):
        contacts_path = contacts_df
        contacts_df = pd.read_parquet(contacts_path)
    if isinstance(edge_list, str):
        edge_list = pd.read_parquet(edge_list)

    #create a generator and pass it to the model, that way everything is deterministic including initialization
    rng = np.random.default_rng(int(seed)) if seed is not None else None
    
    #create model
    model = NetworkModel(
        contacts_df = contacts_df, 
        params = params_copy,
        edge_list = edge_list,
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
    base_seed: Optional[int] = None,
    variants_share_edge_list: Optional[bool] = None,
    parallel: bool = False,
    max_workers: Optional[int] = None
) -> List[Dict]:
    """
        IF VARIANTS_SHARE_EDGE_LIST, edge lists will not be altered from variant to variant, therefore, if varying factors like contact weight or density, variants_share_edge_list MUST BE FALSE

        Run a list of model variants. Each variant is a dict with keys:
        - 'name' - string name of the variant
        - 'param_overrides' - dict of parameters to change and new values
        - algorithm_map' - dict of action_type -> Algorithm
        - 'factory_map' - dict action_type -> factory
        - 'save_exposures

        Returns a list of dicts, one per variant, containing model instance and path
        If parallel = True, variants are dispatched to ProcessPoolExecutor, and each worker writes per-variant outputs to run_dir
    """
    os.makedirs(run_dir, exist_ok = True)
    base_seed = int(base_seed) if base_seed is not None else int(base_params.get("seed", 0))

    # resolve variants_share_edge_list flag
    if variants_share_edge_list is None:
        variants_share_edge_list_flag = bool(base_params.get("variants_share_edge_list", False))
    else:
        variants_share_edge_list_flag = bool(variants_share_edge_list)

    # Precompute per-variant params/n_runs if needed
    variant_param_list = []
    for v_ind, variant in enumerate(variants):
        params_tmp = deepcopy(base_params)
        params_tmp.update(variant.get("param_overrides") or {})
        # seed per variant for reproducibility
        seed = (int(base_seed) + int(v_ind) * 1000) if base_seed is not None else None
        if seed is not None:
            params_tmp["seed"] = int(seed)
        variant_param_list.append((v_ind, variant, params_tmp))

    # If parallel and contacts_df is a DataFrame, write it to disk for workers to read
    contacts_path = None
    if parallel:
        if isinstance(contacts_df, pd.DataFrame):
            contacts_path = os.path.join(run_dir, "contacts_for_variants.parquet")
            # write contacts once for workers
            contacts_df.to_parquet(contacts_path, index = False)
        elif isinstance(contacts_df, str):
            contacts_path = contacts_df
        else:
            raise TypeError("contacts_df must be a pandas DataFrame or a filepath string when parallel=True")
    else:
        # sequential: keep contacts_df in memory (pass DataFrame directly)
        contacts_path = None

    # Build shared edge list if requested (driver-level)
    shared_edge_list = None
    shared_edge_list_path = None
    if variants_share_edge_list_flag:
        # Ensure we have a contacts DataFrame to pass to build_edge_list
        contacts_for_build = contacts_df if isinstance(contacts_df, pd.DataFrame) else pd.read_parquet(contacts_path)
        rng_for_edges = np.random.default_rng(int(base_seed)) if base_seed is not None else None
        shared_edge_list = build_edge_list(
            contacts_df = contacts_for_build,
            params = base_params,
            rng = rng_for_edges,
            save = bool(base_params.get("save_data_files", False)),
            county = base_params.get("county", "")
        )
        if parallel:
            shared_edge_list_path = os.path.join(run_dir, "shared_edge_list.parquet")
            shared_edge_list.to_parquet(shared_edge_list_path, index = False)

    # Prepare results container
    variant_results: List[Dict[str, Any]] = []

    # Sequential execution path (keeps identical outputs to parallel mode)
    if not parallel:
        for v_ind, variant, params in variant_param_list:
            name = variant.get("name", f"variant_{v_ind}")
            safe_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", name)
            variant_dir = os.path.join(run_dir, safe_name)
            os.makedirs(variant_dir, exist_ok = True)

            # choose edge list to pass: shared edge list DataFrame or None (model will build)
            edge_list_to_pass = shared_edge_list if variants_share_edge_list_flag else None

            # run model in current process
            model = run_single_model(
                contacts_df = contacts_df,
                params = params,
                algorithm_map = variant.get("algorithm_map"),
                factory_map = variant.get("factory_map"),
                seed = params.get("seed"),
                results_dir = variant_dir,
                save_exposures = variant.get("save_exposures", False),
                edge_list = edge_list_to_pass
            )

            # compute epi outcomes and save to CSV
            try:
                df_out = model.epi_outcomes(reduced = False)
                df_out.to_csv(os.path.join(variant_dir, "epi_outcomes.csv"), index = False)
            except Exception as exc:
                df_out = None
                with open(os.path.join(variant_dir, "epi_outcomes_error.txt"), "w") as f:
                    f.write(str(exc) + "\n")

            # build per-variant action_log.csv (same logic as worker)
            combined_action_log = []
            base_variant, swept_params = _parse_variant_name(name, name_sep="__")
            all_lhd_logs = getattr(model, "all_lhd_action_logs", []) or []
            for run_number, run_log in enumerate(all_lhd_logs):
                if not run_log:
                    continue
                for action_entry in run_log:
                    try:
                        entry = dict(action_entry)
                    except Exception:
                        entry = {"raw": str(action_entry)}
                    entry["variant_name"] = name
                    entry["base_variant"] = base_variant
                    entry["run_number"] = int(run_number)
                    for k, v in swept_params.items():
                        entry[k] = v
                    combined_action_log.append(entry)

            action_log_path = os.path.join(variant_dir, "action_log.csv")
            if combined_action_log:
                df_actions = pd.json_normalize(combined_action_log, sep = '_')
                def _stringify_cell(x):
                    if isinstance(x, (list, dict, np.ndarray)):
                        return json.dumps(x, default = str)
                    return x
                for col in df_actions.columns:
                    if df_actions[col].apply(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
                        df_actions[col] = df_actions[col].apply(_stringify_cell)
                df_actions.to_csv(action_log_path, index = False)
            else:
                empty_cols = ['variant_name','base_variant','run_number','time','action_id','action_type','kind','nodes_count','hours_used','duration','reversible_tokens','nonreversible_tokens']
                pd.DataFrame(columns = empty_cols).to_csv(action_log_path, index = False)

            # store results in uniform format
            summary_df = df_out if df_out is not None else None
            variant_results.append({
                "variant_name": name,
                "variant_dir": variant_dir,
                "summary": summary_df
            })

        # end sequential loop
        # After all variants run, combine per-variant action logs into run_dir/action_log.csv
        combined_action_log = []
        for vr in variant_results:
            action_csv = os.path.join(vr["variant_dir"], "action_log.csv")
            if os.path.exists(action_csv):
                try:
                    df_a = pd.read_csv(action_csv, dtype = object)
                    # expand rows into list of dicts and extend
                    combined_action_log.extend(df_a.to_dict(orient = "records"))
                except Exception:
                    # if parse fails, skip
                    pass

        if combined_action_log:
            df_actions = pd.json_normalize(combined_action_log, sep = '_')
            # stringify non-scalar cells
            def _stringify_cell(x):
                if isinstance(x, (list, dict, np.ndarray)):
                    return json.dumps(x, default = str)
                return x
            for col in df_actions.columns:
                if df_actions[col].apply(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
                    df_actions[col] = df_actions[col].apply(_stringify_cell)
            csv_path = os.path.join(run_dir, "action_log.csv")
            df_actions.to_csv(csv_path, index = False)
        else:
            empty_cols = ['variant_name','base_variant','run_number','time','action_id','action_type','kind','nodes_count','hours_used','duration','reversible_tokens','nonreversible_tokens']
            pd.DataFrame(columns = empty_cols).to_csv(os.path.join(run_dir, "action_log.csv"), index = False)

        return variant_results

    # Parallel execution path
    # decide number of workers
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(len(variants), cpu_count)

    # path strings to pass to workers
    contact_arg = contacts_path if contacts_path else (None if isinstance(contacts_df, pd.DataFrame) else contacts_df)
    shared_edge_list_path_arg = shared_edge_list_path if variants_share_edge_list_flag else None

    futures = {}
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as exe:
        for v_ind, variant, params in variant_param_list:
            futures[exe.submit(
                _run_variant_job,
                v_ind,
                variant,
                base_params,
                contact_arg,
                run_dir,
                base_seed,
                variants_share_edge_list_flag,
                shared_edge_list_path_arg
            )] = v_ind

        for fut in concurrent.futures.as_completed(futures):
            v_ind = futures[fut]
            try:
                res = fut.result()
                if "error" in res:
                    # log failure
                    print(f"Variant {v_ind} ({res.get('variant_name')}) failed in worker; check {res.get('variant_dir')}/variant_error.txt")
                results.append(res)
            except Exception as exc:
                print(f"Unexpected exception collecting variant {v_ind} result: {exc}")
                import traceback as tb
                tb.print_exc()

    # sort results by variant_index
    results_sorted = sorted(results, key = lambda r: r.get("variant_index", 0))
    variant_results = []
    for r in results_sorted:
        variant_dir = r.get("variant_dir")
        # attempt to load epi_outcomes.csv saved by the worker
        summary_df = None
        if "summary" in r and r["summary"]:
            try:
                summary_df = pd.DataFrame(r["summary"])
            except Exception:
                summary_df = None
        else:
            # fallback: try to load CSV
            csv_path = os.path.join(variant_dir or "", "epi_outcomes.csv")
            if os.path.exists(csv_path):
                try:
                    summary_df = pd.read_csv(csv_path)
                except Exception:
                    summary_df = None

        variant_results.append({
            "variant_name": r.get("variant_name"),
            "variant_dir": variant_dir,
            "summary": summary_df
        })

    # Combine per-variant action logs into master run_dir/action_log.csv
    combined_action_log = []
    for vr in variant_results:
        action_csv = os.path.join(vr["variant_dir"], "action_log.csv")
        if os.path.exists(action_csv):
            try:
                df_a = pd.read_csv(action_csv, dtype = object)
                combined_action_log.extend(df_a.to_dict(orient = "records"))
            except Exception:
                pass

    if combined_action_log:
        df_actions = pd.json_normalize(combined_action_log, sep = '_')
        def _stringify_cell(x):
            if isinstance(x, (list, dict, np.ndarray)):
                return json.dumps(x, default = str)
            return x
        for col in df_actions.columns:
            if df_actions[col].apply(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
                df_actions[col] = df_actions[col].apply(_stringify_cell)
        csv_path = os.path.join(run_dir, "action_log.csv")
        df_actions.to_csv(csv_path, index = False)
    else:
        empty_cols = ['variant_name','base_variant','run_number','time','action_id','action_type','kind','nodes_count','hours_used','duration','reversible_tokens','nonreversible_tokens']
        pd.DataFrame(columns = empty_cols).to_csv(os.path.join(run_dir, "action_log.csv"), index = False)

    return variant_results


def _run_variant_job(
    v_ind: int,    
    variant: Dict,    
    base_params: Dict,    
    contacts_path: str,    
    run_dir: str,    
    base_seed: Optional[int],  
    variants_share_edge_list_flag: bool = False,    
    shared_edge_list_path: Optional[str] = None
    ) -> Dict[str, Any]:
    """
    Worker function executed in a child process for a variant, only if model is run in parallel

    Returns small dict with variant index/name, and path to a saved output
    """
    name = variant.get("name", f"variant_{v_ind}")
    # deterministic seed per variant
    seed = (int(base_seed) + int(v_ind) * 1000) if base_seed is not None else None

    params = deepcopy(base_params)
    params.update(variant.get("param_overrides") or {})
    if seed is not None:
        params["seed"] = int(seed)

    # make a safe folder for this variant
    safe_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", name)
    variant_dir = os.path.join(run_dir, safe_name)
    os.makedirs(variant_dir, exist_ok = True)

    try:
        # read contacts (worker-local)
        contacts_df = pd.read_parquet(contacts_path)

        # read shared edge list (if requested)
        edge_list_arg = None
        if variants_share_edge_list_flag and shared_edge_list_path:
            try:
                edge_list_arg = pd.read_parquet(shared_edge_list_path)
            except Exception:
                edge_list_arg = None

        # run the model for this variant
        model = run_single_model(
            contacts_df = contacts_df,
            params = params,
            algorithm_map = variant.get("algorithm_map"),
            factory_map = variant.get("factory_map"),
            seed = seed,
            results_dir = variant_dir,
            save_exposures = variant.get("save_exposures", False),
            edge_list = edge_list_arg
        )

        # compute epi outcomes (detailed)
        try:
            df_out = model.epi_outcomes(reduced = False)
            out_csv = os.path.join(variant_dir, "epi_outcomes.csv")
            df_out.to_csv(out_csv, index = False)
            summary_records = df_out.to_dict(orient = "records")
        except Exception as exc:
            # record the error and continue
            summary_records = []
            with open(os.path.join(variant_dir, "epi_outcomes_error.txt"), "w") as f:
                f.write(str(exc) + "\n")
                f.write(traceback.format_exc())

        # build per-variant action_log.csv (consistent format)
        combined_action_log = []
        base_variant, swept_params = _parse_variant_name(name, name_sep="__")
        all_lhd_logs = getattr(model, "all_lhd_action_logs", []) or []
        for run_number, run_log in enumerate(all_lhd_logs):
            if not run_log:
                continue
            for action_entry in run_log:
                try:
                    entry = dict(action_entry)
                except Exception:
                    entry = {"raw": str(action_entry)}
                entry["variant_name"] = name
                entry["base_variant"] = base_variant
                entry["run_number"] = int(run_number)
                for k, v in swept_params.items():
                    entry[k] = v
                combined_action_log.append(entry)

        action_log_path = os.path.join(variant_dir, "action_log.csv")
        if combined_action_log:
            df_actions = pd.json_normalize(combined_action_log, sep = '_')
            # stringify non-scalar cells
            def _stringify_cell(x):
                if isinstance(x, (list, dict, np.ndarray)):
                    return json.dumps(x, default = str)
                return x
            for col in df_actions.columns:
                if df_actions[col].apply(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
                    df_actions[col] = df_actions[col].apply(_stringify_cell)
            df_actions.to_csv(action_log_path, index = False)
        else:
            empty_cols = ['variant_name','base_variant','run_number','time','action_id','action_type','kind','nodes_count','hours_used','duration','reversible_tokens','nonreversible_tokens']
            pd.DataFrame(columns = empty_cols).to_csv(action_log_path, index = False)

        return {
            "variant_index": int(v_ind),
            "variant_name": name,
            "variant_dir": variant_dir,
            "summary": summary_records
        }

    except Exception as exc:
        tb = traceback.format_exc()
        # ensure the error is persisted for debugging
        with open(os.path.join(variant_dir, "variant_error.txt"), "w") as f:
            f.write(str(exc) + "\n")
            f.write(tb)
        return {
            "variant_index": int(v_ind),
            "variant_name": name,
            "variant_dir": variant_dir,
            "error": str(exc),
            "traceback": tb
        }