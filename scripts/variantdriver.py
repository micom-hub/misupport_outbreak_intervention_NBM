#scripts/variantdriver.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json
import os
import traceback
import concurrent.futures
import multiprocessing as mp
import shutil
from line_profiler import profile

import pandas as pd

# imports from your codebase
from scripts.variants.run_variants_funcs import prepare_run, run_parameter_set
from scripts.lhd.policy_config import PolicyConfig, POLICY_CONFIGURATION
from scripts.config import ModelConfig

from scripts.visualization.viz import (
    plot_metric_boxplot,
    plot_trajectories_by_variant,
    plot_paired_difference_trajectory,
)

def run_experiment(
    csv_path: str,
    n_samples: int,
    policy_config: PolicyConfig,
    output_dir: Union[str, Path],
    *,
    base_cfg: Optional[ModelConfig] = None,
    data_dir: str = "data",
    overwrite_files: bool = True, #Input files
    save_files: bool = True,
    seed: Optional[int] = None,
    workers: Optional[int] = 1,
    register_defaults: bool = False,
    save_summary: bool = True,
    save_incidence: bool = False,
    save_prevalence: bool = False,
    save_lhd_results: bool = True,
    summary_metrics: Optional[List[str]] = None,
    overwrite_runs: bool = True,#Run results
    clean_dir: bool = False
) -> Dict[str, Any]:
    """
    Runs the whole shebang for LHDsim variants

    1) Prepares runs from LHS with prepare_run
    2) Runs run_parameter_set for each parameter-set
        - Sequential or in parallel if workers > 1 (if None, uses max available)
    3) Aggregate results across each run into a 
        #if clean_dir, deletes variant subdirectories
    """
    out_base = Path(output_dir).expanduser().resolve()

    if out_base.is_dir() and overwrite_runs:
        shutil.rmtree(str(out_base))

    out_base.mkdir(parents = True, exist_ok = True)



    #1) Prepare run
    print("[run_experiment] Initializing run data... ")
    contacts_df, configs_list, master_gd = prepare_run(
        csv_path = csv_path,
        n_samples = n_samples,
        output_dir = str(out_base),
        base_cfg = base_cfg,
        data_dir = data_dir,
        overwrite_files = overwrite_files,
        save_files = save_files,
        seed = seed
    )
    n_configs = len(configs_list)
    print(f"[run_experiment] {n_configs} Models Initialized")

    #2) Execute individual runs
    indices = list(range(n_configs))
    statuses: List[Dict[str, Any]] = []

    #check worker count
    #If None, use one less than maximum CPUs 
    if workers is None:
        workers = max(1, mp.cpu_count()-1)

    #If more than 1 workers, attempt parallel process
    if workers > 1:
        print(f"[run_experiment] Attempting parallel run with {workers} workers")
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers = workers) as exe:
                futures = {
                    exe.submit(
                        run_parameter_set,
                        contacts_df,
                        policy_config,
                        configs_list[i],
                        master_gd,
                        str(out_base),
                        i,
                        seed=seed,
                        register_defaults=register_defaults,
                        save_summary=save_summary,
                        save_incidence=save_incidence,
                        save_prevalence=save_prevalence,
                        save_lhd_results = save_lhd_results,
                        summary_metrics=summary_metrics,
                        overwrite=overwrite_runs,
                    ): i
                    for i in indices
                }
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {
                        "index": int(i),
                        "run_dir": str(out_base / f"model_{int(i):04d}"),
                        "success": False,
                        "error": str(exc),
                        "trace": traceback.format_exc()
                    }
                    statuses.append(res)

        #anticipating problem pickling LHD objects
        except Exception as exc:
            print(f"[run_experiment] Parallel execution failed ({exc}); falling back to sequential execution.")
            statuses = []
            for i in indices:
                res = run_parameter_set(
                    contacts_df,
                    policy_config,
                    configs_list[i],
                    master_gd,
                    str(out_base),
                    i,
                    seed=seed,
                    register_defaults=register_defaults,
                    save_summary=save_summary,
                    save_incidence=save_incidence,
                    save_prevalence=save_prevalence,
                    save_lhd_results = save_lhd_results,
                    summary_metrics=summary_metrics,
                    overwrite=overwrite_runs,
                )
                statuses.append(res)
            
    #Run in sequence
    else:
        print("[run_experiment] Running sequentially...")
        statuses = []
        for i in indices:
            res = run_parameter_set(
                contacts_df,
                policy_config,
                configs_list[i],
                master_gd,
                str(out_base),
                i,
                seed=seed,
                register_defaults=register_defaults,
                save_summary=save_summary,
                save_incidence=save_incidence,
                save_prevalence=save_prevalence,
                save_lhd_results = save_lhd_results,
                summary_metrics=summary_metrics,
                overwrite=overwrite_runs
            )
            statuses.append(res)

    #Write a run status manifest
    sorted_statuses = sorted(statuses, key = lambda x: int(x.get("index", -1)))
    try:
        with open(out_base / "run_status.json", "w") as fh:
            json.dump(sorted_statuses, fh, indent = 2)
    except Exception:
        pass


    #3) Aggregate per-run results to a single file 
    print("[run_experiment] Runs completed, aggregating data...")
    aggregated_paths: Dict[str, str] = {}

    lhs_csv = out_base / "LHS.csv"
    lhs_df = None
    if lhs_csv.exists():
        try:
            lhs_df = pd.read_csv(lhs_csv)
            lhs_df["model_index"] = range(len(lhs_df))
        except:
            lhs_df = None

    def _collect_and_concat(fname: str) -> Optional[pd.DataFrame]:
        #Quick helper function to aggregate files by name to a big pd
        parts = []
        for i in indices:
            run_dir = out_base / f"model_{int(i):04d}"
            fpath = run_dir / fname
            if not fpath.exists():
                continue
            try:
                df = pd.read_parquet(str(fpath))
                # ensure column run_number present
                if "run_number" not in df.columns:
                    df.insert(0, "run_number", range(len(df)))
                df.insert(0, "model_index", i)
                parts.append(df)
            except Exception as exc:
                print(f"[run_experiment] Failed to read {fpath}: {exc}")
        if not parts:
            return None
        big = pd.concat(parts, ignore_index=True, sort=False)

        return big

    if save_summary:
        df_all = _collect_and_concat("summary.parquet")
        if df_all is not None:
            outp = out_base / "aggregated_summary.parquet"
            _atomic_write_parquet(df_all, outp)
            aggregated_paths["summary"] = str(outp)

    if save_incidence:
        df_all = _collect_and_concat("incidence.parquet")
        if df_all is not None:
            outp = out_base / "aggregated_incidence.parquet"
            _atomic_write_parquet(df_all, outp)
            aggregated_paths["incidence"] = str(outp)

    if save_prevalence:
        df_all = _collect_and_concat("prevalence.parquet")
        if df_all is not None:
            outp = out_base / "aggregated_prevalence.parquet"
            _atomic_write_parquet(df_all, outp)
            aggregated_paths["prevalence"] = str(outp)
            
    if save_lhd_results:
        df_all = _collect_and_concat("lhd_results.parquet")
        if df_all is not None:
            outp = out_base / "aggregated_prevalence.parquet"
            _atomic_write_parquet(df_all, outp)
            aggregated_paths["prevalence"] = str(outp)

    if clean_dir:
        print("[run_experiment] cleaning per-model run directories...")
        for i in indices:
            run_dir = out_base / f"model_{int(i):04d}"
            if run_dir.exists() and run_dir.is_dir():
                    shutil.rmtree(run_dir)


    return {
        "run_dir": str(out_base),
        "n_parameter_sets": n_configs,
        "statuses": sorted_statuses,
        "aggregated_paths": aggregated_paths
    }


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    #Helper to atomic-write parquets
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(str(tmp), index=False)
    os.replace(str(tmp), str(path))


if __name__ == "__main__":

    #Uses POLICY_CONFIGURATION defined in scripts/lhd/policy_config.py
    result = run_experiment(
        csv_path="testLHS.csv",
        n_samples=2,
        policy_config=POLICY_CONFIGURATION,
        output_dir="model_runs/experiment_002",
        base_cfg=None,
        seed=3,
        workers=1,
        save_summary=True,
        save_incidence=True,
        save_prevalence=False,
        clean_dir=True
    )
    print("Done. aggregated:", result["aggregated_paths"])

    # summary = pd.read_parquet("model_runs/experiment_002/aggregated_summary.parquet")
    # inc = pd.read_parquet("model_runs/experiment_002/aggregated_incidence.parquet")

    # fig, ax = plot_metric_boxplot(summary, metric="outbreakSize", variant_col="variant")
    # plt.show()

    # fig, axes = plot_trajectories_by_variant(inc, variant_col="variant")
    # plt.show()