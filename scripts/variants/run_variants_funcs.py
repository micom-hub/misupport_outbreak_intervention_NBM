#scripts/variants/run_variants_funcs.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import json
import os
import traceback
import concurrent.futures
import multiprocessing
import time
from line_profiler import profile

import numpy as np
import pandas as pd

from scripts.variants.csv_to_lhs import csv_to_cfg
from scripts.graph.graph_utils import (GraphData,
    build_minimal_graphdata_from_edge_list,
    build_graph_data,
    sample_from_master_graphdata,
)
from scripts.driver import prepare_contacts, read_or_build_master
from scripts.config import ModelConfig
from scripts.lhd.lhdConfig import LhdConfig, validate_variant
from scripts.simulation.outbreak_model import NetworkModel


def prepare_run(
    csv_path: str,
    n_samples: int,
    output_dir: str,
    *,
    base_cfg: Optional[ModelConfig] = None,
    data_dir: str = "data",
    overwrite_files: bool = True,
    save_files: bool = True,
    seed: Optional[int] = None
) -> Tuple[List[ModelConfig], pd.DataFrame]:

    """
    Prepares all necessary data structures for an experiment
        -Ensures synthetic population is loaded and formatted
        - Loads/builds master edge list for experiment

    Args:
        - csv_path to a formatted csv for LHS sweep
        - base_config: a default model config object to be varied on

    Returns: (configs list, master_edge_df)
    """
    #Use defaults in config.py if no config is provided
    base_cfg = base_cfg or ModelConfig()

    #Create output directory      
    out_base = Path(output_dir).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok = True)

    #1) Ensure contact data exists
    contacts_df = prepare_contacts(
        county = base_cfg.sim.county,
        state = base_cfg.sim.state,
        data_dir = data_dir,
        overwrite_files = overwrite_files,
        save_files = save_files
    )

    #2) Load or build master edge list
    if seed is None:
        seed = base_cfg.sim.seed
    base_seed = int(seed)
    rng_master = np.random.default_rng(int(base_seed))

    master_df = read_or_build_master(contacts_df = contacts_df, cfg = base_cfg, 
    run_dir = str(out_base), rng = rng_master, variant = True) 

    #3) Create minimal graph data object with master
    master_gd = build_minimal_graphdata_from_edge_list(master_df, N = contacts_df.shape[0])

    #4) Sample LHS Object 
    configs_list = csv_to_cfg(
        csv_path = csv_path, 
        N = n_samples, 
        output_dir = str(out_base),
        default_config = base_cfg,
        seed = base_seed
        )

    return (contacts_df, configs_list, master_gd)


def run_variants(
    lhd_config: LhdConfig,
    cfg: ModelConfig,
    graphdata: GraphData,
    output_dir: Union[str, Path],
    i: int,
    *,
    seed: Optional[int] = None,
    register_defaults: bool = False,
    overwrite = False,
    save_summary: bool = True,
    save_incidence: bool = False,
    save_prevalence: bool = False,
    summary_metrics: Optional[List[str]] = None
) -> List[NetworkModel]:
    """
    Runs all variants in lhd_config on the same model configuration, contact structure, and rng, producing a run directory under output_dir which contains ModelConfig.json, and specified results files. 

    Args:
        save_summary saves a run summary output containing metrics in summary_metrics (see outbreak_model.results_to_df for compatible metrics)
        save_incidence saves incidence timeseries for each run
        save_prevalence saves prevalence timeseries for each run
        
    """

    #Ensure output directory and run name
    out_base = Path(output_dir).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok = True)

    run_index = int(i)
    run_dir = out_base / f"model_{run_index:04d}"

    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run Directory already exists: {run_dir} (use overwrite=True to replace)")
    else:
        run_dir.mkdir(parents = True, exist_ok = True)



    #Set up model defaults, ensure RNG is as specified
    if seed is not None:
        cfg = cfg.copy_with({"sim": {"seed": int(seed)}})
    seed = cfg.sim.seed
    rng_iteration = np.random.default_rng(seed)

    metrics = summary_metrics

    #save ModelConfig.json
    cfg.to_json(str(run_dir / "ModelConfig.json"))



    #Build containers for results
    models: List[NetworkModel] = []

    summary_dfs = []
    incidence_dfs = []
    prevalence_dfs = []

    #Loop across each variant, instantiate and simulate model, run, write result
    for variant in lhd_config.variants:
        validate_variant(variant)

        model = NetworkModel(
            config = cfg,
            graphdata = graphdata,
            run_dir = str(run_dir),
            rng = rng_iteration,
            lhd_register_defaults = register_defaults,
            lhd_algorithm_map = dict(variant.algorithm_map),
            lhd_action_factory_map = dict(variant.action_factory_map)
        )

        try:
            model.simulate()
        except Exception as exc:
            print(f"[run_variants] simulation failed for variant '{variant.name}':{exc}")

        if save_summary:
            df_summary = model.results_to_df(metrics)
            df_summary.insert(0, "variant_name", variant.name)
            summary_dfs.append(df_summary)
        
        if save_incidence:
            df_incidence = model.timeseries_to_df("incidence")
            df_incidence.insert(0, "variant_name", variant.name)
            incidence_dfs.append(df_incidence)
        
        if save_prevalence:
            df_prevalence = model.timeseries_to_df("prevalence")
            df_prevalence.insert(0, "variant_name", variant.name)
            prevalence_dfs.append(df_prevalence)

        models.append(model)

    #Write all results into an aggregated file under run_dir
    if save_summary:
            df_overall_summary = pd.concat(summary_dfs, ignore_index=True, sort=False)
            df_overall_summary.to_parquet(str(run_dir / "summary.parquet"))
    if save_incidence:
        df_overall_incidence = pd.concat(incidence_dfs, ignore_index=True, sort=False)
        df_overall_incidence.to_parquet(str(run_dir / "incidence.parquet"))

    if save_prevalence:
        df_overall_prevalence = pd.concat(prevalence_dfs, ignore_index=True, sort = False)
        df_overall_prevalence.to_parquet(str(run_dir / "prevalence.parquet"))


    return models

        

def run_parameter_set(
    contacts_df: pd.DataFrame,
    lhd_config: LhdConfig,
    cfg: ModelConfig,
    master_gd: Dict,
    output_dir: Union[str, Path],
    i: int,
    *,
    seed: Optional[int] = None,
    register_defaults: bool = True,
    save_summary: bool = False,
    save_incidence: bool = False,
    save_prevalence: bool = False,
    summary_metrics: Optional[List[str]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Runs a full variant set for a given model config index 'i'

    Args:
        contacts_df: Contact dataframe
        lhd_config: LhdConfig object detailing variants to run
        cfg: ModelConfig for given parameter set (derived from a row of LHS)
        master_gd: minimal graphdata object
        output_dir: base experiment output directory
        i: parameter sest index (0 - N(configs))
        seed: optional base seed (cfg.sim.seed default), run seed is seed+1
        register_defaults
        save_*: instructions to save incidence/prevalence/summary
        summary_metrics: list of summary metrics to run if save_summary True

    
    Returns:
        run_log dict with fields: index, run_dir, success, error

    Saves:
        Variants create model_{i:04d} under output_dir which detail model configurations and results summaries

    """
    #1) Set up output and RNG - each run a different seed
    out_base = Path(output_dir).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    base_seed = int(seed) if seed is not None else int(cfg.sim.seed)
    run_seed = base_seed + int(i)
    rng_for_sampling = np.random.default_rng(run_seed)


    #2) Sample master graphdata and build run graphdata
    sampled_edges_df = sample_from_master_graphdata(
        master_gd,
        cfg,
        rng=rng_for_sampling
    )

    run_graphdata = build_graph_data(
        edge_list = sampled_edges_df,
        contacts_df = contacts_df,
        config = cfg,
        rng = rng_for_sampling,
        N = int(contacts_df.shape[0])
    )

    #3) Initialize and simulate model
    models = run_variants(
        lhd_config,
        cfg,
        graphdata = run_graphdata,
        output_dir = str(out_base),
        i = int(i),

        seed = run_seed,
        register_defaults = register_defaults,
        save_summary=save_summary,
        save_incidence=save_incidence,
        save_prevalence=save_prevalence,
        summary_metrics=summary_metrics,
        overwrite=overwrite
    )

    run_dir = out_base / f"model_{int(i):04d}"
    return{
        "index": int(i),
        "run_dir":str(run_dir),
        "success":True,
        "models_run": len(models)
    }




