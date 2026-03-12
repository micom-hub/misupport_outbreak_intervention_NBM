#scripts/variants/csv_to_lhs.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from scripts.lhd.lhdConfig import LhdConfig, validate_variant
from scripts.simulation.outbreak_model import NetworkModel
from scripts.config import ModelConfig
from scripts.graph.graph_utils import GraphData



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
    run_dir = out_base / f"param_{run_index:04d}"

    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run Directory already exists: {run_dir} (use overwrite=True to replace)")



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

        