# scripts/variantdriver.py
from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd

from scripts.config import ModelConfig
from scripts.lhd.policy_config import (
    POLICY_CONFIGURATION,
    PolicyConfig,
)
from scripts.utils.output_profiles import (
    AGGREGATED_FILENAMES,
    OUTPUT_FILENAMES,
    atomic_write_parquet,
    resolve_outputs,
    write_json,
)
from scripts.variants.run_variants_funcs import (
    ParameterSetResult,
    VariantRunResult,
    prepare_run,
    run_parameter_set,
)


@dataclass
class PolicyComparisonResult:
    """Interactive paired-policy run on one shared graph."""

    models: Dict[str, Any]
    graphdata: Any
    sampled_edges: pd.DataFrame
    run_dir: Optional[Path]
    output_paths: Dict[str, Path]


@dataclass
class ExperimentResult:
    """Result of a multi-parameter-set experiment."""

    run_dir: Path
    n_parameter_sets: int
    statuses: List[Dict[str, Any]]
    aggregated_paths: Dict[str, Path]
    models: Optional[
        Dict[int, Dict[str, Any]]
    ] = None


def run_policy_comparison(
    contacts_df: pd.DataFrame,
    cfg: ModelConfig,
    policy_config: PolicyConfig,
    *,
    output_dir: Optional[
        Union[str, Path]
    ] = None,
    seed: Optional[int] = None,
    output_profile: str = "none",
    extra_outputs: Optional[
        Iterable[str]
    ] = None,
    overwrite: bool = False,
) -> PolicyComparisonResult:
    """
    Run all policies on one shared graph and return models by variant name.

    This uses the same index-zero seed hierarchy as run_experiment().
    """
    from scripts.graph.graph_utils import (
        build_graph_data,
        build_minimal_graphdata_from_edge_list,
        sample_from_master_graphdata,
    )
    from scripts.singledriver import (
        read_or_build_master,
    )
    from scripts.utils.rng_utility import (
        derive_seed_from_base,
    )
    from scripts.variants.run_variants_funcs import (
        run_variants,
    )

    cfg.validate()

    base_seed = int(
        seed
        if seed is not None
        else cfg.sim.seed
    )
    run_seed = int(
        derive_seed_from_base(
            base_seed,
            "run_0",
        )
    )
    graph_seed = int(
        derive_seed_from_base(run_seed)
    )

    if output_dir is None:
        runtime_dir = Path(".")
        result_dir = None
    else:
        result_dir = Path(
            output_dir
        ).expanduser().resolve()
        result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        runtime_dir = result_dir

    master = read_or_build_master(
        contacts_df=contacts_df,
        cfg=cfg,
        run_dir=runtime_dir,
        seed=graph_seed,
        save_master=False,
    )

    minimal_master = (
        build_minimal_graphdata_from_edge_list(
            master,
            N=int(len(contacts_df)),
        )
    )

    sampled_edges = sample_from_master_graphdata(
        minimal_master,
        cfg,
        seed=graph_seed,
    )

    graphdata = build_graph_data(
        edge_list=sampled_edges,
        contacts_df=contacts_df,
        config=cfg,
        seed=graph_seed,
        N=int(len(contacts_df)),
    )

    variant_result: VariantRunResult = run_variants(
        policy_config=policy_config,
        cfg=cfg,
        graphdata=graphdata,
        output_dir=result_dir,
        i=0,
        seed=run_seed,
        output_profile=output_profile,
        extra_outputs=extra_outputs,
        overwrite=overwrite,
        return_models=True,
    )

    return PolicyComparisonResult(
        models=variant_result.models,
        graphdata=graphdata,
        sampled_edges=sampled_edges,
        run_dir=variant_result.run_dir,
        output_paths=variant_result.output_paths,
    )


def run_experiment(
    csv_path: str,
    n_samples: int,
    policy_config: PolicyConfig,
    output_dir: Union[str, Path],
    *,
    base_cfg: Optional[ModelConfig] = None,
    data_dir: str = "data",
    seed: Optional[int] = None,
    workers: Optional[int] = 1,
    output_profile: str = "core",
    extra_outputs: Optional[
        Iterable[str]
    ] = None,
    return_models: bool = False,
    overwrite_contact_files: bool = False,
    save_contact_files: bool = True,
    save_master_graph: bool = False,
    overwrite_run_dirs: bool = False,
    clean_output_dir: bool = False,
    clean_model_dirs: bool = False,
) -> ExperimentResult:
    """
    Run paired policies over an LHS parameter design.

    Returning live models is supported only for sequential execution.
    """
    if return_models and workers not in {
        None,
        1,
    }:
        raise ValueError(
            "return_models=True requires workers=1."
        )

    outputs = resolve_outputs(
        output_profile,
        extra_outputs,
    )

    base_cfg = base_cfg or ModelConfig()
    base_cfg.validate()

    output_dir = Path(
        output_dir
    ).expanduser().resolve()

    if output_dir.exists() and clean_output_dir:
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_seed = int(
        seed
        if seed is not None
        else base_cfg.sim.seed
    )

    contacts, configs, master_graphdata = prepare_run(
        csv_path=csv_path,
        n_samples=n_samples,
        output_dir=output_dir,
        base_cfg=base_cfg,
        data_dir=data_dir,
        overwrite_files=overwrite_contact_files,
        save_contact_files=save_contact_files,
        save_master_graph=save_master_graph,
        seed=base_seed,
    )

    indices = list(range(len(configs)))

    write_json(
        {
            "csv_path": str(
                Path(csv_path)
                .expanduser()
                .resolve()
            ),
            "n_parameter_sets": len(configs),
            "base_seed": base_seed,
            "workers": workers,
            "output_profile": output_profile,
            "outputs": sorted(outputs),
            "policies": [
                {
                    "variant_name":
                        variant.name,
                    "policy_name":
                        variant.policy_name,
                    "lhd_overrides": dict(
                        variant.lhd_overrides
                    ),
                    "description":
                        variant.description,
                }
                for variant
                in policy_config.variants
            ],
            "base_config":
                base_cfg.to_dict(),
        },
        output_dir / "experiment_manifest.json",
    )

    if workers is None:
        workers = max(
            1,
            mp.cpu_count() - 1,
        )

    statuses: List[Dict[str, Any]] = []

    retained_models: Optional[
        Dict[int, Dict[str, Any]]
    ] = (
        {}
        if return_models
        else None
    )

    if int(workers) > 1:
        statuses = _run_parallel(
            contacts=contacts,
            configs=configs,
            master_graphdata=master_graphdata,
            policy_config=policy_config,
            output_dir=output_dir,
            indices=indices,
            base_seed=base_seed,
            workers=int(workers),
            output_profile=output_profile,
            extra_outputs=extra_outputs,
            overwrite_run_dirs=
                overwrite_run_dirs,
        )

    else:
        for index in indices:
            status, models = (
                _run_one_parameter_set(
                    contacts=contacts,
                    policy_config=
                        policy_config,
                    cfg=configs[index],
                    master_graphdata=
                        master_graphdata,
                    output_dir=output_dir,
                    index=index,
                    base_seed=base_seed,
                    output_profile=
                        output_profile,
                    extra_outputs=
                        extra_outputs,
                    overwrite_run_dirs=
                        overwrite_run_dirs,
                    return_models=
                        return_models,
                )
            )

            statuses.append(status)

            if (
                retained_models is not None
                and models is not None
            ):
                retained_models[index] = models

    statuses.sort(
        key=lambda row: int(row["index"])
    )

    write_json(
        statuses,
        output_dir / "run_status.json",
    )

    aggregated_paths = (
        _aggregate_experiment_outputs(
            output_dir=output_dir,
            indices=indices,
            outputs=outputs,
        )
    )

    if clean_model_dirs:
        for index in indices:
            model_dir = (
                output_dir
                / f"model_{index:04d}"
            )

            if model_dir.exists():
                shutil.rmtree(model_dir)

    return ExperimentResult(
        run_dir=output_dir,
        n_parameter_sets=len(configs),
        statuses=statuses,
        aggregated_paths=aggregated_paths,
        models=retained_models,
    )


def _run_parallel(
    *,
    contacts: pd.DataFrame,
    configs: List[ModelConfig],
    master_graphdata: Dict[str, Any],
    policy_config: PolicyConfig,
    output_dir: Path,
    indices: List[int],
    base_seed: int,
    workers: int,
    output_profile: str,
    extra_outputs: Optional[
        Iterable[str]
    ],
    overwrite_run_dirs: bool,
) -> List[Dict[str, Any]]:
    statuses: List[Dict[str, Any]] = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = {
            executor.submit(
                run_parameter_set,
                contacts,
                policy_config,
                configs[index],
                master_graphdata,
                output_dir,
                index,
                seed=base_seed,
                output_profile=output_profile,
                extra_outputs=extra_outputs,
                overwrite=overwrite_run_dirs,
                return_models=False,
            ): index
            for index in indices
        }

        for future in (
            concurrent.futures.as_completed(
                futures
            )
        ):
            index = futures[future]

            try:
                result = future.result()
                statuses.append(
                    _status_from_result(result)
                )
            except Exception as exception:
                statuses.append(
                    _failure_status(
                        index=index,
                        output_dir=output_dir,
                        exception=exception,
                    )
                )

    return statuses


def _run_one_parameter_set(
    *,
    contacts: pd.DataFrame,
    policy_config: PolicyConfig,
    cfg: ModelConfig,
    master_graphdata: Dict[str, Any],
    output_dir: Path,
    index: int,
    base_seed: int,
    output_profile: str,
    extra_outputs: Optional[
        Iterable[str]
    ],
    overwrite_run_dirs: bool,
    return_models: bool,
) -> tuple[
    Dict[str, Any],
    Optional[Dict[str, Any]],
]:
    try:
        result = run_parameter_set(
            contacts_df=contacts,
            policy_config=policy_config,
            cfg=cfg,
            master_graphdata=master_graphdata,
            output_dir=output_dir,
            i=index,
            seed=base_seed,
            output_profile=output_profile,
            extra_outputs=extra_outputs,
            overwrite=overwrite_run_dirs,
            return_models=return_models,
        )

        return (
            _status_from_result(result),
            result.models,
        )

    except Exception as exception:
        return (
            _failure_status(
                index=index,
                output_dir=output_dir,
                exception=exception,
            ),
            None,
        )


def _status_from_result(
    result: ParameterSetResult,
) -> Dict[str, Any]:
    return {
        "index": int(result.index),
        "run_dir": result.run_dir,
        "success": bool(result.success),
        "models_run": int(result.models_run),
        "output_paths": dict(
            result.output_paths
        ),
        "error": result.error,
    }


def _failure_status(
    *,
    index: int,
    output_dir: Path,
    exception: Exception,
) -> Dict[str, Any]:
    return {
        "index": int(index),
        "run_dir": str(
            output_dir
            / f"model_{int(index):04d}"
        ),
        "success": False,
        "models_run": 0,
        "output_paths": {},
        "error": str(exception),
        "trace": traceback.format_exc(),
    }


def _aggregate_experiment_outputs(
    *,
    output_dir: Path,
    indices: List[int],
    outputs: set[str],
) -> Dict[str, Path]:
    aggregated_paths: Dict[str, Path] = {}

    aggregatable = (
        set(AGGREGATED_FILENAMES)
        & outputs
    )

    for output_name in sorted(aggregatable):
        filename = OUTPUT_FILENAMES[
            output_name
        ]
        parts = []

        for index in indices:
            path = (
                output_dir
                / f"model_{index:04d}"
                / filename
            )

            if not path.exists():
                continue

            dataframe = pd.read_parquet(path)

            if dataframe.empty:
                continue

            dataframe.insert(
                0,
                "model_index",
                int(index),
            )

            if "run_number" in dataframe.columns:
                dataframe.insert(
                    1,
                    "pair_id",
                    (
                        f"model_{index:04d}"
                        + "__run_"
                        + dataframe[
                            "run_number"
                        ]
                        .astype(int)
                        .astype(str)
                        .str.zfill(4)
                    ),
                )
            else:
                dataframe.insert(
                    1,
                    "pair_id",
                    f"model_{index:04d}",
                )

            parts.append(dataframe)

        if not parts:
            continue

        combined = pd.concat(
            parts,
            ignore_index=True,
            sort=False,
        )

        aggregate_path = (
            output_dir
            / AGGREGATED_FILENAMES[
                output_name
            ]
        )

        atomic_write_parquet(
            combined,
            aggregate_path,
        )
        aggregated_paths[
            output_name
        ] = aggregate_path

    return aggregated_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "model_runs/"
            "control_boundary_pilot"
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--profile",
        choices=[
            "none",
            "core",
            "audit",
            "full",
        ],
        default="core",
    )
    args = parser.parse_args()

    result = run_experiment(
        csv_path="testLHS.csv",
        n_samples=args.samples,
        policy_config=
            POLICY_CONFIGURATION,
        output_dir=args.output_dir,
        base_cfg=ModelConfig(),
        seed=5,
        workers=args.workers,
        output_profile=args.profile,
        return_models=False,
        overwrite_run_dirs=True,
        clean_output_dir=True,
        clean_model_dirs=False,
    )

    print(
        "Experiment completed:",
        result.run_dir,
    )
    print(
        "Aggregated outputs:",
        result.aggregated_paths,
    )