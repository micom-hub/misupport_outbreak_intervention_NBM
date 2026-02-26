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
import shutil
import glob
import ast
import traceback

from scripts.synth_data_processing import synthetic_data_process, build_edge_list
from scripts.fred_fetch import downloadPopData
from scripts.network_model import NetworkModel, ModelParameters, \
    DefaultModelParams


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
    contacts_df: Union[pd.DataFrame, str],
    base_params: Dict,
    variants: List[Dict],
    run_dir: str,
    base_seed: Optional[int] = None,
    variants_share_edge_list: Optional[bool] = None,
    parallel: bool = False,
    max_workers: Optional[int] = None
) -> List[Dict]:
    """
    Run a list of variants sequentially or in parallel. Returns a list of dicts, each containing:
        { "variant_name": str, "variant_dir": str, "summary": pd.DataFrame | None }
    Works with the save_variant_results(...) contract (which writes per-variant files).
    """

    os.makedirs(run_dir, exist_ok=True)
    base_seed = int(base_seed) if base_seed is not None else int(base_params.get("seed", 0))

    # resolve variants_share_edge_list flag
    if variants_share_edge_list is None:
        variants_share_edge_list_flag = bool(base_params.get("variants_share_edge_list", False))
    else:
        variants_share_edge_list_flag = bool(variants_share_edge_list)

    # If parallel, write contacts parquet for workers to read (avoid pickling DataFrame)
    contacts_path = None
    if parallel:
        if isinstance(contacts_df, pd.DataFrame):
            contacts_path = os.path.join(run_dir, "contacts_for_variants.parquet")
            # write once
            contacts_df.to_parquet(contacts_path, index=False)
        elif isinstance(contacts_df, str):
            contacts_path = contacts_df
        else:
            raise TypeError("contacts_df must be a pandas DataFrame or a file path when parallel=True")
    else:
        # sequential: keep contacts_df in memory (pass DataFrame directly)
        contacts_path = None

    # Build shared edge_list if requested by driver-level flag
    shared_edge_list_path = None
    shared_edge_list_df = None
    if variants_share_edge_list_flag:
        # need a contacts DataFrame to build edge list
        contacts_for_build = contacts_df if isinstance(contacts_df, pd.DataFrame) else pd.read_parquet(contacts_path)
        rng_for_edges = np.random.default_rng(int(base_seed)) if base_seed is not None else None
        try:
            shared_edge_list_df = build_edge_list(
                contacts_df = contacts_for_build,
                params = base_params,
                rng = rng_for_edges,
                save = bool(base_params.get("save_data_files", False)),
                county = base_params.get("county", "")
            )
            if parallel:
                shared_edge_list_path = os.path.join(run_dir, "shared_edge_list.parquet")
                shared_edge_list_df.to_parquet(shared_edge_list_path, index=False)
        except Exception as exc:
            # If building shared edge list fails, warn and continue without shared edge list
            print(f"[run_variants] Warning: building shared edge list failed: {exc}")
            shared_edge_list_df = None
            shared_edge_list_path = None

    variant_results: List[Dict[str, Any]] = []

    # Sequential execution path
    if not parallel:
        for v_ind, variant in enumerate(variants):
            # compute seed and params for this variant
            seed = (int(base_seed) + int(v_ind) * 1000) if base_seed is not None else None
            params = deepcopy(base_params)
            params.update(variant.get("param_overrides", {}) or {})
            if seed is not None:
                params["seed"] = int(seed)

            variant_name = variant.get("name") or f"variant_{v_ind}"
            safe_name = re.sub(r'[^0-9A-Za-z_.-]+', '_', str(variant_name)).strip('_')
            variant_dir = os.path.join(run_dir, safe_name)
            os.makedirs(variant_dir, exist_ok=True)

            # choose edge_list to pass (DataFrame if driver built it)
            edge_list_to_pass = shared_edge_list_df if variants_share_edge_list_flag else None

            try:
                model = run_single_model(
                    contacts_df = contacts_df if isinstance(contacts_df, pd.DataFrame) else pd.read_parquet(contacts_path) if contacts_path else contacts_df,
                    params = params,
                    algorithm_map = variant.get("algorithm_map"),
                    factory_map = variant.get("factory_map"),
                    seed = seed,
                    results_dir = variant_dir,
                    save_exposures = variant.get("save_exposures", False),
                    edge_list = edge_list_to_pass
                )

                # Persist using standard helper
                save_variant_results(model, variant, run_dir, name = variant_name)

                # Load per-variant epi_outcomes as the 'summary' DataFrame to return
                epi_path = os.path.join(variant_dir, "epi_outcomes.csv")
                if os.path.exists(epi_path):
                    try:
                        epi_df = pd.read_csv(epi_path)
                    except Exception:
                        epi_df = None
                else:
                    epi_df = None

                variant_results.append({
                    "variant_name": variant_name,
                    "variant_dir": variant_dir,
                    "summary": epi_df
                })

            except Exception as exc:
                # write an error file in variant_dir for debugging
                try:
                    with open(os.path.join(variant_dir, "variant_runtime_error.txt"), "w") as fh:
                        fh.write(str(exc) + "\n")
                        fh.write(traceback.format_exc())
                except Exception:
                    pass

                print(f"[run_variants] Variant {v_ind} ({variant_name}) failed: {exc}")
                variant_results.append({
                    "variant_name": variant_name,
                    "variant_dir": variant_dir,
                    "summary": None
                })

        return variant_results

    # Parallel execution path
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(len(variants), cpu_count)

    futures = {}
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as exe:
        for v_ind, variant in enumerate(variants):
            fut = exe.submit(
                _run_variant_job,
                v_ind,
                variant,
                base_params,
                contacts_path,
                run_dir,
                base_seed,
                variants_share_edge_list_flag,
                shared_edge_list_path
            )
            futures[fut] = v_ind

        for fut in concurrent.futures.as_completed(futures):
            v_ind = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                if "error" in res:
                    print(f"[run_variants] Worker failed for variant {v_ind} ({res.get('variant_name')}): {res.get('error')}")
            except Exception as exc:
                print(f"[run_variants] Unexpected executor error for variant {v_ind}: {exc}")
                traceback.print_exc()

    # Sort results by variant_index to preserve ordering
    results_sorted = sorted(results, key=lambda r: r.get("variant_index", 0))

    # Build returned variant_results list; always try to load epi_outcomes.csv into DataFrame
    for r in results_sorted:
        variant_dir = r.get("variant_dir")
        vname = r.get("variant_name")
        epi_df = None
        if variant_dir and os.path.exists(os.path.join(variant_dir, "epi_outcomes.csv")):
            try:
                epi_df = pd.read_csv(os.path.join(variant_dir, "epi_outcomes.csv"))
            except Exception:
                epi_df = None

        variant_results.append({
            "variant_name": vname,
            "variant_dir": variant_dir,
            "summary": epi_df
        })

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
    Worker executed in a child process: runs a single variant and saves per-variant outputs.
    Returns a small dict:
      - variant_index
      - variant_name
      - variant_dir
      - error / traceback if failed (or 'ok' marker)
    """
    # local imports to keep spawn-safe / avoid top-level cycle surprises
    import os
    import re
    import json
    import traceback
    from copy import deepcopy
    import pandas as pd

    def _sanitize_name(s: str) -> str:
        return re.sub(r'[^0-9A-Za-z_.-]+', '_', str(s)).strip('_')

    try:
        # derive deterministic seed for this variant
        seed = (int(base_seed) + int(v_ind) * 1000) if base_seed is not None else None

        # merge params
        params = deepcopy(base_params)
        params.update(variant.get("param_overrides", {}) or {})
        if seed is not None:
            params["seed"] = int(seed)

        # variant naming and directory
        variant_name = variant.get("name") or f"variant_{v_ind}"
        safe_name = _sanitize_name(variant_name)
        variant_dir = os.path.join(run_dir, safe_name)
        os.makedirs(variant_dir, exist_ok=True)

        # read contacts (worker-local)
        if contacts_path is None:
            raise RuntimeError("Worker expected a contacts_path (parquet) to read but got None")
        contacts_df = pd.read_parquet(contacts_path)

        # read shared edge list if required (worker-local)
        edge_list_arg = None
        if variants_share_edge_list_flag and shared_edge_list_path:
            try:
                edge_list_arg = pd.read_parquet(shared_edge_list_path)
            except Exception:
                # fall back to None; NetworkModel will build its own edge_list
                edge_list_arg = None

        # Execute model for this variant (run_single_model from same module)
        # run_single_model will instantiate NetworkModel and call simulate()
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

        # Persist per-variant results using centralized helper
        # save_variant_results writes manifest, epi_outcomes.csv, actions_summary.csv, timeseries.{parquet/csv}
        try:
            save_variant_results(model, variant, run_dir, name = variant_name)
        except Exception as exc:
            # persist a small error file but continue to return success (we still have variant files if model created)
            with open(os.path.join(variant_dir, "save_variant_results_error.txt"), "w") as fh:
                fh.write(str(exc) + "\n")
                fh.write(traceback.format_exc())

        return {
            "variant_index": int(v_ind),
            "variant_name": variant_name,
            "variant_dir": variant_dir,
            "status": "ok"
        }

    except Exception as exc:
        tb = traceback.format_exc()
        # attempt to write an error file in variant_dir for convenience
        try:
            safe_name = re.sub(r'[^0-9A-Za-z_.-]+', '_', str(variant.get("name") or f"variant_{v_ind}")).strip('_')
            variant_dir = os.path.join(run_dir, safe_name)
            os.makedirs(variant_dir, exist_ok=True)
            with open(os.path.join(variant_dir, "variant_runtime_error.txt"), "w") as fh:
                fh.write(str(exc) + "\n\n")
                fh.write(tb)
        except Exception:
            pass

        return {
            "variant_index": int(v_ind),
            "variant_name": variant.get("name") or f"variant_{v_ind}",
            "variant_dir": variant_dir if 'variant_dir' in locals() else None,
            "error": str(exc),
            "traceback": tb
        }


    
#Helper function to clean run names
def _sanitize_name(s: str) -> str:
    s = str(s) if s is not None else ""
    return re.sub(r'[^0-9A-Za-z_.-]+', "_", s).strip("_")

#Saves variant results for each model
def save_variant_results(model, variant: dict, run_dir: str, name: str = None):
    """
    Persist per-variant outputs for a completed NetworkModel 'model' run.
    Writes into run_dir/<safe_variant_name>/ :
      - variant_manifest.json
      - epi_outcomes.csv
      - actions_summary.csv
      - timeseries.parquet  (preferred) or timeseries.csv (fallback)
    Arguments:
        model: NetworkModel after simulate()
        variant: original variant dict (may contain 'sweep_params','param_overrides','sweep_index')
        run_dir: top-level run directory
        name: optional variant name override (otherwise variant['name'])
    Returns:
        variant_dir (str) path written
    """
    # Determine name and sanitized folder
    variant_name = name or variant.get("name") or f"variant_{variant.get('sweep_index', '')}"
    base_variant = variant.get("base_variant") or str(variant_name).split("__")[0]
    safe_name = _sanitize_name(variant_name)
    variant_dir = os.path.join(run_dir, safe_name)
    os.makedirs(variant_dir, exist_ok=True)

    # Write manifest
    manifest = {
        "variant_name": variant_name,
        "base_variant": base_variant,
        "param_overrides": variant.get("param_overrides", {}),
        "sweep_params": variant.get("sweep_params", {}),
        "sweep_index": variant.get("sweep_index", None)
    }
    try:
        with open(os.path.join(variant_dir, "variant_manifest.json"), "w") as fh:
            json.dump(manifest, fh, default=str, indent=2)
    except Exception:
        # best-effort; continue
        pass

    # 1) Save epi_outcomes (detailed per-run rows)
    try:
        epi_df = model.epi_outcomes(reduced=False)
        epi_path = os.path.join(variant_dir, "epi_outcomes.csv")
        epi_df.to_csv(epi_path, index=False)
    except Exception as exc:
        # save failure info for debugging
        with open(os.path.join(variant_dir, "epi_outcomes_error.txt"), "w") as fh:
            fh.write(str(exc))

    # 2) Build and save actions_summary (one row per run per action_type/kind)
    # model.all_lhd_action_logs expected shape: list per run of action_log entries (dicts)
    combined_action_entries = []
    try:
        all_lhd_logs = getattr(model, "all_lhd_action_logs", []) or []
        for run_idx, run_log in enumerate(all_lhd_logs):
            if not run_log:
                continue
            for entry in run_log:
                # entry should be dict-like: keys such as action_type, kind, nodes_count, hours_used, duration, reversible_tokens, nonreversible_tokens
                try:
                    e = dict(entry)
                except Exception:
                    e = {"raw": str(entry)}
                e["run_number"] = int(run_idx)
                combined_action_entries.append(e)

        if combined_action_entries:
            df_actions = pd.DataFrame(combined_action_entries)
            # ensure numeric columns exist
            for col in ("nodes_count", "hours_used", "duration", "reversible_tokens", "nonreversible_tokens"):
                if col not in df_actions.columns:
                    df_actions[col] = 0
                df_actions[col] = pd.to_numeric(df_actions[col], errors="coerce").fillna(0)

            if "action_type" not in df_actions.columns:
                df_actions["action_type"] = df_actions.get("action", "unknown")
            if "kind" not in df_actions.columns:
                df_actions["kind"] = df_actions.get("kind", "")

            # group and aggregate
            actions_summary = df_actions.groupby(["run_number", "action_type", "kind"], as_index=False).agg(
                n_actions = ("action_id" if "action_id" in df_actions.columns else df_actions.columns[0], "count"),
                total_nodes_count = ("nodes_count", "sum"),
                total_hours_used = ("hours_used", "sum"),
                total_duration = ("duration", "sum"),
                total_reversible_tokens = ("reversible_tokens", "sum"),
                total_nonreversible_tokens = ("nonreversible_tokens", "sum"),
            )
        else:
            actions_summary = pd.DataFrame(columns=["run_number","action_type","kind","n_actions","total_nodes_count","total_hours_used","total_duration","total_reversible_tokens","total_nonreversible_tokens"])
    except Exception as exc:
        actions_summary = pd.DataFrame(columns=["run_number","action_type","kind","n_actions","total_nodes_count","total_hours_used","total_duration","total_reversible_tokens","total_nonreversible_tokens"])
        # write error to file
        with open(os.path.join(variant_dir, "actions_summary_error.txt"), "w") as fh:
            fh.write(str(exc))

    actions_summary_path = os.path.join(variant_dir, "actions_summary.csv")
    try:
        actions_summary.to_csv(actions_summary_path, index=False)
    except Exception:
        # ignore write errors but try to continue
        pass

    # 3) Build per-run timeseries rows (one row per run, series columns are Python lists)
    timeseries_rows = []
    N = int(getattr(model, "N", 0))
    for run_idx in range(getattr(model, "n_runs", 1)):
        # states_over_time: list of [S,E,I,R] lists saved in all_states_over_time[run_idx]
        states = getattr(model, "all_states_over_time", [None])[run_idx]
        exposures = getattr(model, "all_new_exposures", [None])[run_idx]

        # ensure lists
        states_list = states if states is not None else []
        exposures_list_raw = exposures if exposures is not None else []

        T_states = len(states_list)
        T_exposures = len(exposures_list_raw)
        T = T_states if T_states > 0 else T_exposures

        # build exposures_list aligned to T, and ensure exposures_list[0] contains I0 if empty
        exposures_list = []
        for t in range(T):
            if t < T_exposures:
                arr = exposures_list_raw[t]
                try:
                    arr_np = np.array(arr, dtype=int) if (hasattr(arr, "__len__") and len(arr) > 0) else np.empty(0, dtype=int)
                except Exception:
                    # fallback
                    try:
                        arr_np = np.array(list(arr), dtype=int)
                    except Exception:
                        arr_np = np.empty(0, dtype=int)
            else:
                arr_np = np.empty(0, dtype=int)
            exposures_list.append(arr_np)

        # initial seeds handling
        if len(exposures_list) > 0 and exposures_list[0].size == 0:
            I0 = model.params.get("I0", [])
            try:
                exposures_list[0] = np.array(I0, dtype=int) if I0 is not None else np.empty(0, dtype=int)
            except Exception:
                try:
                    exposures_list[0] = np.array(list(I0), dtype=int)
                except Exception:
                    exposures_list[0] = np.empty(0, dtype=int)

        # compute cumulative unique infected series
        cum_bool = np.zeros(N, dtype=bool) if N > 0 else np.zeros(0, dtype=bool)
        cumulative_counts = []
        cumulative_fracs = []
        for t in range(T):
            arr = exposures_list[t] if t < len(exposures_list) else np.empty(0, dtype=int)
            if arr.size and N > 0:
                # mask invalid indices safely
                arr_safe = arr[(arr >= 0) & (arr < N)]
                if arr_safe.size:
                    cum_bool[arr_safe] = True
            ccount = int(cum_bool.sum()) if N > 0 else 0
            cumulative_counts.append(ccount)
            cumulative_fracs.append((ccount / float(N)) if N > 0 else 0.0)

        # compute prevalence series
        prevalence_counts = []
        prevalence_fracs = []
        for t in range(T):
            if t < len(states_list) and states_list[t] is not None and len(states_list[t]) > 2:
                I_nodes = states_list[t][2]
                try:
                    nI = int(len(I_nodes))
                except Exception:
                    # if stored as numpy array
                    try:
                        nI = int(np.asarray(I_nodes).size)
                    except Exception:
                        nI = 0
            else:
                nI = 0
            prevalence_counts.append(nI)
            prevalence_fracs.append((nI / float(N)) if N > 0 else 0.0)

        # build row
        row = {
            "variant_name": variant_name,
            "base_variant": base_variant,
            "run_number": int(run_idx),
            "series_length": int(T),
            "prevalence_count_series": prevalence_counts,
            "prevalence_frac_series": prevalence_fracs,
            "cumulative_infections_count_series": cumulative_counts,
            "cumulative_infections_frac_series": cumulative_fracs,
        }
        # flatten sweep params into columns as convenience
        sweep_params = variant.get("sweep_params", {}) or {}
        for k, v in sweep_params.items():
            row[str(k)] = v

        timeseries_rows.append(row)

    timeseries_df = pd.DataFrame(timeseries_rows)

    # Save timeseries per-variant (parquet preferred)
    timeseries_path_parquet = os.path.join(variant_dir, "timeseries.parquet")
    timeseries_path_csv = os.path.join(variant_dir, "timeseries.csv")
    try:
        timeseries_df.to_parquet(timeseries_path_parquet, index=False)
    except Exception:
        # fallback: JSON-serialize list-columns and save CSV
        def _jsonify(x):
            try:
                return json.dumps(x, default=str)
            except Exception:
                return str(x)
        df_csv = timeseries_df.copy()
        for col in ["prevalence_count_series","prevalence_frac_series","cumulative_infections_count_series","cumulative_infections_frac_series"]:
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].apply(_jsonify)
        df_csv.to_csv(timeseries_path_csv, index=False)

    return variant_dir


def clean_run_results(run_dir: Optional[str] = None, delete_variants: bool = True):
    """
    Takes messy outputs from run_variants and aggregates into run files for analysis
    Required per-variant files (created by save_variant_results):
      - variant_manifest.json
      - epi_outcomes.csv
      - actions_summary.csv
      - timeseries.parquet or timeseries.csv
    Produces and saves in run_dir:
      - aggregated_run_results.parquet  (and aggregated_run_results.csv)
      - timeseries.parquet  (concatenated per-run rows; one row per run; list-valued series columns)
    After successful saving, deletes per-variant directories (irreversible).
    Returns:
      (overall_df, timeseries_df)
    """
   

    run_dir = os.path.abspath(run_dir) if run_dir else os.getcwd()
    print(f"[clean_run_results] Aggregating from {run_dir}")

    # find epi_outcomes.csv anywhere under run_dir (recursive)
    pattern = os.path.join(run_dir, "**", "epi_outcomes.csv")
    epi_paths = glob.glob(pattern, recursive=True)

    if not epi_paths:
        # helpful diagnostic output
        sample = os.listdir(run_dir)[:20]
        raise RuntimeError(f"No variant subdirectories with epi_outcomes.csv found under {run_dir}. Top-level entries: {sample}")

    per_run_frames = []
    per_timeseries_frames = []
    processed_dirs = []

    def _sanitize(s: str) -> str:
        return re.sub(r'[^0-9A-Za-z]+', '_', str(s)).strip('_')

    for epi_path in sorted(epi_paths):
        try:
            variant_dir = os.path.dirname(epi_path)
            # try manifest if exists
            manifest_path = os.path.join(variant_dir, "variant_manifest.json")
            manifest = None
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r") as fh:
                        manifest = json.load(fh)
                except Exception:
                    manifest = None

            variant_name = manifest.get("variant_name") if (manifest and "variant_name" in manifest) else os.path.basename(variant_dir)
            base_variant = manifest.get("base_variant") if (manifest and "base_variant" in manifest) else (variant_name.split("__")[0] if "__" in variant_name else variant_name)
            sweep_params = manifest.get("sweep_params", {}) if manifest else {}

            # load epi outcomes
            epi_df = pd.read_csv(epi_path)
            epi_df["variant_name"] = variant_name
            epi_df["base_variant"] = base_variant
            for k, v in (sweep_params.items() if isinstance(sweep_params, dict) else []):
                epi_df[str(k)] = v
            per_run_frames.append(epi_df)

            # load timeseries (prefer parquet)
            ts_parq = os.path.join(variant_dir, "timeseries.parquet")
            ts_csv = os.path.join(variant_dir, "timeseries.csv")
            ts_df = None
            if os.path.exists(ts_parq):
                try:
                    ts_df = pd.read_parquet(ts_parq)
                except Exception:
                    ts_df = None
            if ts_df is None and os.path.exists(ts_csv):
                try:
                    ts_df = pd.read_csv(ts_csv, dtype=object)
                    # attempt to parse JSON-like list columns
                    for col in ["prevalence_count_series","prevalence_frac_series","cumulative_infections_count_series","cumulative_infections_frac_series"]:
                        if col in ts_df.columns:
                            def _parse_cell(x):
                                if pd.isna(x):
                                    return []
                                try:
                                    return json.loads(x)
                                except Exception:
                                    try:
                                        return ast.literal_eval(x)
                                    except Exception:
                                        return []
                            ts_df[col] = ts_df[col].apply(_parse_cell)
                except Exception:
                    ts_df = None

            if ts_df is not None:
                if "variant_name" not in ts_df.columns:
                    ts_df["variant_name"] = variant_name
                if "base_variant" not in ts_df.columns:
                    ts_df["base_variant"] = base_variant
                for k, v in (sweep_params.items() if isinstance(sweep_params, dict) else []):
                    if str(k) not in ts_df.columns:
                        ts_df[str(k)] = v
                per_timeseries_frames.append(ts_df)
            else:
                # no timeseries for this variant; create stub rows per run_number in epi_df
                if "run_number" in epi_df.columns:
                    rows = []
                    for rn in sorted(epi_df["run_number"].unique()):
                        rows.append({
                            "variant_name": variant_name,
                            "base_variant": base_variant,
                            "run_number": int(rn),
                            "series_length": 0,
                            "prevalence_count_series": [],
                            "prevalence_frac_series": [],
                            "cumulative_infections_count_series": [],
                            "cumulative_infections_frac_series": [],
                            **{str(k): v for k, v in (sweep_params.items() if isinstance(sweep_params, dict) else [])}
                        })
                    if rows:
                        per_timeseries_frames.append(pd.DataFrame(rows))

            processed_dirs.append(variant_dir)

        except Exception as exc:
            print(f"[clean_run_results] Error processing {epi_path}: {exc}")
            traceback.print_exc()
            continue

    if not per_run_frames:
        raise RuntimeError("No per-variant epi_outcomes loaded; aborting aggregation")

    overall_df = pd.concat(per_run_frames, ignore_index=True, sort=False)
    timeseries_df = pd.concat(per_timeseries_frames, ignore_index=True, sort=False) if per_timeseries_frames else pd.DataFrame()

    # Save aggregated result files
    agg_parquet = os.path.join(run_dir, "aggregated_run_results.parquet")
    agg_csv = os.path.join(run_dir, "aggregated_run_results.csv")
    ts_parquet = os.path.join(run_dir, "timeseries.parquet")
    ts_csv = os.path.join(run_dir, "timeseries.csv")

    # try parquet, fall back to csv
    try:
        overall_df.to_parquet(agg_parquet, index=False)
        print(f"[clean_run_results] Wrote {agg_parquet}")
    except Exception:
        overall_df.to_csv(agg_csv, index=False)
        print(f"[clean_run_results] Wrote {agg_csv}")

    try:
        # if list columns exist, parquet can store lists; if it fails, fallback to CSV with JSON-serialized lists
        timeseries_df.to_parquet(ts_parquet, index=False)
        print(f"[clean_run_results] Wrote {ts_parquet}")
    except Exception:
        df_csv = timeseries_df.copy()
        for col in ["prevalence_count_series","prevalence_frac_series","cumulative_infections_count_series","cumulative_infections_frac_series"]:
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].apply(lambda x: json.dumps(x, default=str))
        df_csv.to_csv(ts_csv, index=False)
        print(f"[clean_run_results] Wrote {ts_csv}")

    # If all aggregated files exist, optionally delete per-variant directories
    saved_ok = (os.path.exists(agg_parquet) or os.path.exists(agg_csv)) and (os.path.exists(ts_parquet) or os.path.exists(ts_csv))
    if saved_ok and delete_variants:
        for d in processed_dirs:
            try:
                shutil.rmtree(d)
                print(f"[clean_run_results] Removed {d}")
            except Exception as exc:
                print(f"[clean_run_results] Could not remove {d}: {exc}")

    return overall_df, timeseries_df

if __name__ == "__main__":
    contacts_df = prepare_contacts(county = "Keweenaw", state = "Michigan")
    model = run_single_model(contacts_df, params = DefaultModelParams)
