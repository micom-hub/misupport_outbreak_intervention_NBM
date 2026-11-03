# scripts/variants/run_variants_funcs.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd

from scripts.config import ModelConfig
from scripts.graph.graph_utils import (
    GraphData,
    build_graph_data,
    build_minimal_graphdata_from_edge_list,
    sample_from_master_graphdata,
)
from scripts.lhd.policy_config import (
    PolicyConfig,
    PolicyVariant,
    validate_variant,
)
from scripts.simulation.outbreak_model import NetworkModel
from scripts.singledriver import (
    prepare_contacts,
    read_or_build_master,
)
from scripts.utils.output_profiles import (
    OUTPUT_FILENAMES,
    atomic_write_parquet,
    resolve_outputs,
    write_json,
)
from scripts.utils.rng_utility import derive_seed_from_base
from scripts.variants.csv_to_lhs import csv_to_cfg


@dataclass
class VariantRunResult:
    """All policy variants run on one shared graph and parameter set."""

    index: int
    run_dir: Optional[Path]
    models: Dict[str, NetworkModel]
    output_paths: Dict[str, Path] = field(
        default_factory=dict
    )
    sampled_edges: Optional[
        pd.DataFrame
    ] = None
    graphdata: Optional[GraphData] = None


@dataclass
class ParameterSetResult:
    """Result returned for one parameter set."""

    index: int
    run_dir: Optional[str]
    success: bool
    models_run: int
    output_paths: Dict[str, str]
    models: Optional[
        Dict[str, NetworkModel]
    ] = None
    error: Optional[str] = None


def prepare_run(
    csv_path: str,
    n_samples: int,
    output_dir: Union[str, Path],
    *,
    base_cfg: Optional[ModelConfig] = None,
    data_dir: str = "data",
    overwrite_files: bool = False,
    save_contact_files: bool = True,
    save_master_graph: bool = False,
    seed: Optional[int] = None,
) -> tuple[
    pd.DataFrame,
    List[ModelConfig],
    Dict[str, Any],
]:
    """
    Prepare contacts, a master candidate graph, and LHS configurations.

    RNG behavior:
        preparation_seed =
            derive_seed_from_base(base_seed, "preparation")
    """
    base_cfg = base_cfg or ModelConfig()
    base_cfg.validate()

    output_dir = Path(
        output_dir
    ).expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    contacts = prepare_contacts(
        county=base_cfg.sim.county,
        state=base_cfg.sim.state,
        data_dir=data_dir,
        overwrite_files=overwrite_files,
        save_files=save_contact_files,
    )

    base_seed = int(
        seed
        if seed is not None
        else base_cfg.sim.seed
    )
    preparation_seed = int(
        derive_seed_from_base(
            base_seed,
            "preparation",
        )
    )

    master = read_or_build_master(
        contacts_df=contacts,
        cfg=base_cfg,
        run_dir=output_dir,
        seed=preparation_seed,
        save_master=save_master_graph,
    )

    minimal_master = (
        build_minimal_graphdata_from_edge_list(
            master,
            N=int(len(contacts)),
        )
    )

    configs = csv_to_cfg(
        csv_path=csv_path,
        N=int(n_samples),
        output_dir=str(output_dir),
        default_config=base_cfg,
        seed=preparation_seed,
    )

    return contacts, configs, minimal_master


def run_variants(
    policy_config: PolicyConfig,
    cfg: ModelConfig,
    graphdata: GraphData,
    output_dir: Optional[
        Union[str, Path]
    ],
    i: int,
    *,
    seed: Optional[int] = None,
    output_profile: str = "core",
    extra_outputs: Optional[
        Iterable[str]
    ] = None,
    overwrite: bool = False,
    return_models: bool = True,
) -> VariantRunResult:
    """
    Run every policy variant on the same graph and model seed.

    This preserves paired-policy comparisons: all variants share graph
    structure, imported cases, vaccination draws, and replicate seed sequence.
    """
    outputs = resolve_outputs(
        output_profile,
        extra_outputs,
    )
    run_index = int(i)

    if output_dir is None:
        run_dir = None
        model_run_dir = "."
    else:
        output_base = Path(
            output_dir
        ).expanduser().resolve()
        run_dir = (
            output_base
            / f"model_{run_index:04d}"
        )

        if outputs:
            if run_dir.exists():
                if not overwrite:
                    raise FileExistsError(
                        "Parameter-set directory "
                        f"already exists: {run_dir}"
                    )

                import shutil
                shutil.rmtree(run_dir)

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        model_run_dir = str(run_dir)

    if seed is not None:
        cfg = cfg.copy_with(
            {
                "sim": {
                    "seed": int(seed),
                }
            }
        )

    cfg.validate()
    model_seed = int(cfg.sim.seed)

    models: Dict[str, NetworkModel] = {}

    collected_outputs: Dict[
        str,
        List[pd.DataFrame],
    ] = {
        output_name: []
        for output_name in outputs
        if output_name != "sampled_edges"
    }

    variant_manifest = {
        "model_index": run_index,
        "model_seed": model_seed,
        "variants": [],
    }

    for variant in policy_config.variants:
        validate_variant(variant)

        variant_cfg = _cfg_with_policy_variant(
            cfg,
            variant,
        )

        model = NetworkModel(
            config=variant_cfg,
            graphdata=graphdata,
            run_dir=model_run_dir,
            seed=model_seed,
        )
        model.simulate()

        if return_models:
            models[variant.name] = model

        for output_name in collected_outputs:
            dataframe = _produce_model_output(
                model,
                output_name,
            )

            if dataframe.empty:
                continue

            collected_outputs[
                output_name
            ].append(
                _add_variant_metadata(
                    dataframe,
                    variant=variant,
                )
            )

        variant_manifest["variants"].append(
            {
                "variant_name": variant.name,
                "policy_name": variant.policy_name,
                "description": variant.description,
                "lhd_overrides": dict(
                    variant.lhd_overrides
                ),
                "resolved_config":
                    variant_cfg.to_dict(),
                "replicate_seeds":
                    model.all_seed_metadata,
            }
        )

    output_paths: Dict[str, Path] = {}

    if outputs and run_dir is not None:
        config_path = (
            run_dir / "ModelConfig.json"
        )
        cfg.to_json(str(config_path))
        output_paths["config"] = config_path

        manifest_path = (
            run_dir / "variant_manifest.json"
        )
        write_json(
            variant_manifest,
            manifest_path,
        )
        output_paths[
            "variant_manifest"
        ] = manifest_path

        for output_name, frames in (
            collected_outputs.items()
        ):
            if not frames:
                continue

            combined = pd.concat(
                frames,
                ignore_index=True,
                sort=False,
            )
            path = (
                run_dir
                / OUTPUT_FILENAMES[
                    output_name
                ]
            )

            atomic_write_parquet(
                combined,
                path,
            )
            output_paths[output_name] = path

    return VariantRunResult(
        index=run_index,
        run_dir=run_dir,
        models=models,
        output_paths=output_paths,
        graphdata=(
            graphdata
            if return_models
            else None
        ),
    )


def run_parameter_set(
    contacts_df: pd.DataFrame,
    policy_config: PolicyConfig,
    cfg: ModelConfig,
    master_graphdata: Dict[str, Any],
    output_dir: Optional[
        Union[str, Path]
    ],
    i: int,
    *,
    seed: Optional[int] = None,
    output_profile: str = "core",
    extra_outputs: Optional[
        Iterable[str]
    ] = None,
    overwrite: bool = False,
    return_models: bool = False,
) -> ParameterSetResult:
    """
    Sample one parameter-set graph and run every policy on it.

    RNG behavior:
        run_seed =
            derive_seed_from_base(base_seed, f"run_{i}")
        graph_seed =
            derive_seed_from_base(run_seed)
    """
    index = int(i)
    base_seed = int(
        seed
        if seed is not None
        else cfg.sim.seed
    )

    run_seed = int(
        derive_seed_from_base(
            base_seed,
            f"run_{index}",
        )
    )
    graph_seed = int(
        derive_seed_from_base(run_seed)
    )

    sampled_edges = sample_from_master_graphdata(
        master_graphdata,
        cfg,
        seed=graph_seed,
    )

    run_graphdata = build_graph_data(
        edge_list=sampled_edges,
        contacts_df=contacts_df,
        config=cfg,
        seed=graph_seed,
        N=int(len(contacts_df)),
    )

    outputs = resolve_outputs(
        output_profile,
        extra_outputs,
    )

    variant_result = run_variants(
        policy_config=policy_config,
        cfg=cfg,
        graphdata=run_graphdata,
        output_dir=output_dir,
        i=index,
        seed=run_seed,
        output_profile=output_profile,
        extra_outputs=extra_outputs,
        overwrite=overwrite,
        return_models=return_models,
    )

    output_paths = {
        name: str(path)
        for name, path
        in variant_result.output_paths.items()
    }

    if (
        outputs
        and "sampled_edges" in outputs
        and variant_result.run_dir is not None
    ):
        sampled_path = (
            variant_result.run_dir
            / OUTPUT_FILENAMES[
                "sampled_edges"
            ]
        )
        atomic_write_parquet(
            sampled_edges,
            sampled_path,
        )
        output_paths[
            "sampled_edges"
        ] = str(sampled_path)

    run_dir = (
        str(variant_result.run_dir)
        if variant_result.run_dir is not None
        else None
    )

    # Metadata is written only when the selected profile writes run outputs.
    if outputs and variant_result.run_dir is not None:
        metadata_path = (
            variant_result.run_dir
            / "run_metadata.json"
        )

        write_json(
            {
                "model_index": index,
                "base_seed": base_seed,
                "run_seed": run_seed,
                "graph_seed": graph_seed,
                "n_variants": len(
                    policy_config.variants
                ),
                "output_profile":
                    output_profile,
                "config": cfg.to_dict(),
            },
            metadata_path,
        )
        output_paths[
            "run_metadata"
        ] = str(metadata_path)

    return ParameterSetResult(
        index=index,
        run_dir=run_dir,
        success=True,
        models_run=len(
            policy_config.variants
        ),
        output_paths=output_paths,
        models=(
            variant_result.models
            if return_models
            else None
        ),
    )


def _produce_model_output(
    model: NetworkModel,
    output_name: str,
) -> pd.DataFrame:
    producers = {
        "factual_summary":
            model.factual_run_summary_to_df,
        "daily_epi":
            model.daily_epi_to_df,
        "daily_lhd":
            model.lhd_daily_to_df,
        "events":
            model.event_log_to_df,
        "tasks":
            model.lhd_tasks_to_df,
        "incidence":
            lambda: model.timeseries_to_df(
                "incidence"
            ),
        "prevalence":
            lambda: model.timeseries_to_df(
                "prevalence"
            ),
        "transmissions":
            model.transmissions_to_df,
    }

    producer = producers.get(output_name)

    if producer is None:
        return pd.DataFrame()

    dataframe = producer()

    if not isinstance(dataframe, pd.DataFrame):
        return pd.DataFrame(dataframe)

    return dataframe


def _add_variant_metadata(
    dataframe: pd.DataFrame,
    *,
    variant: PolicyVariant,
) -> pd.DataFrame:
    output = dataframe.copy()

    if "variant_name" in output.columns:
        output["variant_name"] = variant.name
    else:
        output.insert(
            0,
            "variant_name",
            variant.name,
        )

    if "policy_name" in output.columns:
        output["policy_name"] = (
            variant.policy_name
        )
    else:
        output.insert(
            1,
            "policy_name",
            variant.policy_name,
        )

    return output


def _cfg_with_policy_variant(
    cfg: ModelConfig,
    variant: PolicyVariant,
) -> ModelConfig:
    patch = {
        "lhd": {
            "policy_name":
                str(variant.policy_name),
            **dict(variant.lhd_overrides),
        }
    }

    return cfg.copy_with(patch)