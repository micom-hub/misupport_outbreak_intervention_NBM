"""
Driver to structure data, configure, and run NetworkModel objects

The script:
1) Ensures data for the desired county exists
2) Checks for a master edge list, or builds it and a GraphData object
3) Samples an edge list from master edge list per model object
4) Builds GraphData object and caches it
5) Instantiates NetworkModel with ModelConfig and GraphData objects
"""

import os
import json
import shutil
import hashlib
from typing import Dict, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from filelock import FileLock, Timeout

from new_scripts.config import ModelConfig, DEFAULT_MODEL_CONFIG
from new_scripts.graph.cache_utils import (
    load_master_edge_list,
    save_master_edge_list,
)
from new_scripts.graph.graph_utils import (
    build_minimal_graphdata_from_edge_list,
    build_graph_data,
    sample_from_master_graphdata,
)

from new_scripts.simulation.outbreak_model import NetworkModel
from scripts.synth_data_processing import synthetic_data_process, build_edge_list
from scripts.fred_fetch import downloadPopData


# Prepare and structure contact data
def prepare_contacts(
    county: str,
    state: str,
    data_dir: str = "data",
    overwrite_files: bool = True,
    save_files: bool = False,
):
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

    # find any entries in data_dir that start with county_name
    entries = [f for f in os.listdir(data_dir) if f.startswith(county_name)]
    if not entries:
        downloadPopData(state=state_name, county=county_name)
        # re-list
        entries = [f for f in os.listdir(data_dir) if f.startswith(county_name)]
        if not entries:
            raise RuntimeError(
                f"downloadPopData did not produce any data files for {county_name} in {data_dir}"
            )

    # prefer a directory if present, else take zip or folder
    dir_entries = [e for e in entries if os.path.isdir(os.path.join(data_dir, e))]
    if dir_entries:
        countyfoldername = dir_entries[0]
    else:
        countyfoldername = entries[0]

    countyfolder = os.path.join(data_dir, countyfoldername)

    # If data is a zipfile, use synthetic_data_process to extract contact data
    # delete zip file
    if os.path.isfile(countyfolder) and countyfolder.lower().endswith(".zip"):
        try:
            contacts_df = synthetic_data_process(county)
        except Exception as e:
            raise RuntimeError(
                f"Failed to process synthetic data from zip for {county_name}: {e}"
            ) from e

        if save_files and os.path.exists(countyfolder):
            try:
                os.remove(countyfolder)
            except OSError:
                pass
        _validate_contacts_df(contacts_df)
        return contacts_df

    # if countyfolder is a directory, check for contacts.parquet, if not there, process synthetic data
    if os.path.isdir(countyfolder):
        parquet_path = os.path.join(countyfolder, "contacts.parquet")
        # if overwrite_files, do not reread old contacts.parquet
        if os.path.exists(parquet_path) and not overwrite_files:
            return pd.read_parquet(parquet_path)
        else:
            try:
                contacts_df = synthetic_data_process(county_name, save_files=save_files)
                _validate_contacts_df(contacts_df)
                return contacts_df
            except Exception as e:
                raise RuntimeError(
                    f"Could not find contacts.parquet in {countyfolder} and synthetic_data_process failed: {e}"
                ) from e

    raise RuntimeError(f"Unhandled data entry for county {county_name}: {countyfolder}")


# Read or build master edge list for county population
def get_master(
    contacts_df: pd.DataFrame,
    cfg: ModelConfig,
    cache_root: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
    force_rebuild: bool = False,
    lock_timeout: float = 600.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load master edge list from cache or build it.

    Returns (master_edges_df, metadata_dict). Also constructs and returns minimal graphdata if needed by callers.
    """
    # Normalize cache root
    cache_root = (
        cache_root
        or cfg.sim.graph_cache_dir
        or os.path.join(os.getcwd(), "graph_cache")
    )
    os.makedirs(cache_root, exist_ok=True)

    master_key = cfg.sim.master_cache_key or _master_cache_key_from_config(cfg)
    master_cache_dir = os.path.join(cache_root, master_key)
    lock = FileLock(master_cache_dir + ".lock")

    # Try to acquire lock and load/build
    try:
        with lock.acquire(timeout=lock_timeout):
            if not force_rebuild and os.path.isdir(master_cache_dir):
                try:
                    master_df, metadata = load_master_edge_list(master_cache_dir)
                    return master_df, metadata
                except Exception as exc:
                    # If cache is bad, remove it and rebuild
                    print(
                        f"[get_master] existing master cache at {master_cache_dir} failed to load ({exc}), will rebuild."
                    )
                    try:
                        shutil.rmtree(master_cache_dir)
                    except Exception:
                        pass

            # Need to build master edge list
            rng_for_master = (
                rng if rng is not None else np.random.default_rng(int(cfg.sim.seed))
            )
            master_df = build_edge_list(
                contacts_df=contacts_df,
                config=cfg,
                rng=rng_for_master,
                save=False,
                county=cfg.sim.county,
                master_casual_contacts=int(cfg.sim.master_casual_candidates),
            )
            # save master list
            save_master_edge_list(
                master_df,
                master_cache_dir,
                cache_key=master_key,
                params=cfg.to_dict(),
                params_keys_for_hash=[
                    "hh_weight",
                    "wp_weight",
                    "sch_weight",
                    "gq_weight",
                    "cas_weight",
                ],
                overwrite=False,
            )
            metadata = {"master_cache_key": master_key}
            return master_df, metadata
    except Timeout:
        # If we timed try again but don't save
        if os.path.isdir(master_cache_dir):
            try:
                master_df, metadata = load_master_edge_list(master_cache_dir)
                return master_df, metadata
            except Exception:
                pass
        # fallback: build in-memory and return (do not save)
        rng_for_master = (
            rng if rng is not None else np.random.default_rng(int(cfg.sim.seed))
        )
        master_df = build_edge_list(
            contacts_df=contacts_df,
            config=cfg,
            rng=rng_for_master,
            save=False,
            county=cfg.sim.county,
            master_casual_contacts=int(cfg.sim.master_casual_candidates),
        )
        metadata = {"master_cache_key": master_key, "saved": False}
        return master_df, metadata

# Single model run

def run_single_model(
    contacts_src: Union[pd.DataFrame, str],
    cfg: Union[ModelConfig, Dict[str, Any]],
    *,
    algorithm_map: Optional[Dict[str, object]] = None,
    factory_map: Optional[Dict[str, callable]] = None,
    seed: Optional[int] = None,
    results_dir: Optional[str] = None,
    save_exposures: bool = False,
    master_cache_root: Optional[str] = None,
    force_master_rebuild: bool = False,
    lock_timeout: float = 600.0
) -> NetworkModel:
    """
    High-level driver function:
      - ensure contacts DF;
      - load/build master edge list (cached);
      - build minimal master graphdata for fast sampling;
      - sample a model-level edge_list deterministically (run_index = 0 so all internal n_runs are identical);
      - build/persist full GraphData for the sampled network (cached per master+cfg);
      - instantiate NetworkModel with provided graphdata and config and run model.simulate()
    """

    # Normalize ModelConfig input
    if isinstance(cfg, dict):
        cfg = ModelConfig.from_nested_dict(cfg)
    if seed is not None:
        cfg = cfg.copy_with({"sim": {"seed": int(seed)}})

    # Resolve contacts_df
    if isinstance(contacts_src, pd.DataFrame):
        contacts_df = contacts_src.reset_index(drop=True)
    elif isinstance(contacts_src, str):
        # treat as path-to-parquet or county name
        if os.path.exists(contacts_src) and contacts_src.endswith(".parquet"):
            contacts_df = pd.read_parquet(contacts_src).reset_index(drop=True)
        else:
            # treat as county name (use cfg.sim.state to complement)
            contacts_df = prepare_contacts(contacts_src, cfg.sim.state, data_dir="data", overwrite_files=cfg.sim.overwrite_edge_list, save_files=cfg.sim.save_data_files)
    else:
        raise TypeError("contacts_src must be a DataFrame or a path/county string")

    # set results_dir
    results_dir = results_dir or os.path.join(os.getcwd(), "results", cfg.sim.run_name)
    if cfg.sim.save_data_files:
        os.makedirs(results_dir, exist_ok=True)

    # Build or load master
    master_df, master_meta = get_master(contacts_df, cfg, cache_root=master_cache_root, rng=np.random.default_rng(int(cfg.sim.seed)), force_rebuild=force_master_rebuild, lock_timeout=lock_timeout)

    # Build minimal graphdata (CSR-by-type only) for sampling
    minimal_graphdata = build_minimal_graphdata_from_edge_list(master_df, N=int(contacts_df.shape[0]))

    # Compute deterministic model-level run seed (we sample once per model so internal replicates share)
    model_run_index = 0
    run_seed = cfg.sim.seed  # the sampler function derives run_seed using base_run_seed + run_index
    # Build a run cache key: include master key and sampling-relevant params (to avoid collisions)
    master_key = cfg.sim.master_cache_key or _master_cache_key_from_config(cfg)

    sampled_edges_df = sample_from_master_graphdata(minimal_graphdata, cfg, base_run_seed=cfg.sim.seed, run_index=0, rng=None)
    # Build full run GraphData (this is the heavy step)
    run_graphdata = build_graph_data(edge_list=sampled_edges_df, contacts_df=contacts_df, params=cfg.to_dict(), rng=np.random.default_rng(int(cfg.sim.seed)), N=int(contacts_df.shape[0]))

    # Instantiate NetworkModel with provided prebuilt GraphData
    model = NetworkModel(
        graphdata = run_graphdata, 
        config = cfg,
        rng = np.random.default_rng(int(cfg.sim.seed)),
        results_folder = results_dir,
        lhd_register_defaults = False,
        lhd_algorithm_map = algorithm_map,
        lhd_action_factory_map = factory_map
    )

    # Run simulation
    model.simulate()

    # Optional save exposures (legacy behavior)
    if save_exposures and results_dir:
        np.savez_compressed(os.path.join(results_dir, "exposure_event_log.npz"), *model.exposure_event_log)

    return model


#Infrastructure for variant runs
def prepare_scenario(contacts_df, cfg: ModelConfig, *, cache_root = None, rng = None):
    #Ensure master edge list exists
    master_df, master_meta = get_master(contacts_df, cfg, cache_root=cache_root, rng=rng)
    master_graphdata = build_minimal_graphdata_from_edge_list(master_df, N=int(contacts_df.shape[0]))
    sampled_edges = sample_from_master_graphdata(master_graphdata, cfg, base_run_seed=cfg.sim.seed, run_index=0, rng=rng)
    run_graphdata = build_graph_data(edge_list=sampled_edges, contacts_df=contacts_df, params=cfg.to_dict(), rng=rng or np.random.default_rng(cfg.sim.seed), N=int(contacts_df.shape[0]))
    return run_graphdata

def run_variant_on_graphdata(contacts_df, cfg: ModelConfig, graphdata, variant_map, variant_dir):
    model = NetworkModel(
        contacts_df=contacts_df,
        config=cfg,
        graphdata=graphdata,
        rng=np.random.default_rng(int(cfg.sim.seed)),
        results_folder=variant_dir,
        lhd_register_defaults=False,
        lhd_algorithm_map=variant_map,
    )

    model.simulate()
    
    return model



# --------------------------
# Helpers: validation & cache keys
# --------------------------
def _validate_contacts_df(df: pd.DataFrame) -> None:
    """Raise if df is missing required columns for synthesis."""
    required = {"PID", "hh_id", "wp_id", "sch_id", "gq_id", "age", "sex", "gq"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Contacts DataFrame missing required columns: {missing}")


def _master_cache_key_from_config(cfg: ModelConfig) -> str:
    """Derive a stable master cache key from ModelConfig (covers inputs that affect master)."""
    key_obj = {
        "county": cfg.sim.county,
        "master_casual_candidates": int(cfg.sim.master_casual_candidates),
        # include the base weights which affect which edges are kept at master build-time
        "hh_weight": float(cfg.population.hh_weight),
        "wp_weight": float(cfg.population.wp_weight),
        "sch_weight": float(cfg.population.sch_weight),
        "gq_weight": float(cfg.population.gq_weight),
        "cas_weight": float(cfg.population.cas_weight),
    }
    h = hashlib.sha256(
        json.dumps(key_obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"master_{h[:16]}"


def _run_cache_key_from_master(master_key: str, cfg: ModelConfig) -> str:
    """
    Compute run-level cache key.
    """
    param_subset = {
        "wp_contacts": int(cfg.population.wp_contacts),
        "sch_contacts": int(cfg.population.sch_contacts),
        "cas_contacts": int(cfg.population.cas_contacts),
        "gq_contacts": int(cfg.population.gq_contacts),
        "seed": int(cfg.sim.seed),
    }
    base = (
        f"{master_key}:{json.dumps(param_subset, sort_keys=True, separators=(',',':'))}"
    )
    h = hashlib.sha256(base.encode()).hexdigest()
    return f"{master_key}_run_{h[:16]}"


if __name__ == "__main__":
    cfg = DEFAULT_MODEL_CONFIG.copy_with(
        {
            "sim": {"I0": [5, 10, 15], "display_plots": True},
            "epi": {"base_transmission_prob": 0.99},
        }
    )
    contacts = prepare_contacts(cfg.sim.county, cfg.sim.state)
    prepare_scenario(contacts, cfg)
    model = run_single_model(contacts, cfg, results_dir="results/testing")
    model.write_run_results()
