# scripts/singledriver.py
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd

from scripts.config import ModelConfig
from scripts.graph.graph_utils import (
    GraphData,
    build_graph_data,
    build_minimal_graphdata_from_edge_list,
    sample_from_master_graphdata,
)
from scripts.simulation.outbreak_model import NetworkModel
from scripts.utils.fred_fetch import downloadPopData
from scripts.utils.output_profiles import (
    OUTPUT_FILENAMES,
    atomic_write_parquet,
    resolve_outputs,
)
from scripts.utils.synth_data_processing import (
    build_edge_list,
    synthetic_data_process,
)


ContactsSource = Union[pd.DataFrame, str, Path]
ConfigSource = Union[ModelConfig, str, Path]


def prepare_contacts(
    county: str,
    state: str,
    data_dir: str = "data",
    *,
    overwrite_files: bool = False,
    save_files: bool = True,
) -> pd.DataFrame:
    """Load or prepare a county synthetic population."""
    county_name = str(county).strip().title()
    state_name = str(state).strip().title()

    data_path = Path(data_dir).expanduser().resolve()
    data_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    county_dir = data_path / county_name
    contacts_path = county_dir / "contacts.parquet"
    county_zip = data_path / f"{county_name}.zip"

    if (
        contacts_path.exists()
        and not overwrite_files
    ):
        contacts = pd.read_parquet(contacts_path)
        _validate_contacts_df(contacts)
        return contacts.reset_index(drop=True)

    if (
        not county_dir.exists()
        and not county_zip.exists()
    ):
        downloadPopData(
            state=state_name,
            county=county_name,
            projectDirectory=str(data_path.parent),
        )

    try:
        contacts = synthetic_data_process(
            county_name,
            save_files=save_files,
        )
    except Exception as exception:
        raise RuntimeError(
            "Failed to prepare synthetic population "
            f"for {county_name}, {state_name}: "
            f"{exception}"
        ) from exception

    _validate_contacts_df(contacts)
    return contacts.reset_index(drop=True)


def load_contacts_source(
    contacts_src: ContactsSource,
    cfg: ModelConfig,
    *,
    data_dir: str = "data",
) -> pd.DataFrame:
    """Normalize a DataFrame, Parquet path, or county name."""
    if isinstance(contacts_src, pd.DataFrame):
        contacts = contacts_src.copy()

    elif isinstance(contacts_src, (str, Path)):
        candidate = Path(contacts_src).expanduser()

        if candidate.exists():
            if candidate.suffix.lower() != ".parquet":
                raise ValueError(
                    "A contacts filepath must point "
                    "to a Parquet file."
                )

            contacts = pd.read_parquet(candidate)
        else:
            contacts = prepare_contacts(
                county=str(contacts_src),
                state=cfg.sim.state,
                data_dir=data_dir,
                overwrite_files=False,
                save_files=True,
            )

    else:
        raise TypeError(
            "contacts_src must be a DataFrame, "
            "Parquet path, or county name."
        )

    contacts = contacts.reset_index(drop=True)
    _validate_contacts_df(contacts)
    return contacts


def read_or_build_master(
    contacts_df: pd.DataFrame,
    cfg: ModelConfig,
    run_dir: Union[str, Path],
    *,
    seed: Optional[int] = None,
    save_master: bool = False,
) -> pd.DataFrame:
    """
    Load a cached master graph or build one using the supplied seed.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    master_path = run_dir / "MasterEdgelist.parquet"

    if (
        master_path.exists()
        and not bool(cfg.sim.overwrite_master)
    ):
        return pd.read_parquet(master_path)

    master = build_edge_list(
        contacts_df=contacts_df,
        config=cfg,
        seed=(
            int(seed)
            if seed is not None
            else int(cfg.sim.seed)
        ),
        save=False,
        county=cfg.sim.county,
        master_casual_contacts=int(
            cfg.sim.master_casual_candidates
        ),
    )

    if save_master:
        atomic_write_parquet(
            master,
            master_path,
        )

    return master


def build_run_graphdata(
    contacts_df: pd.DataFrame,
    cfg: ModelConfig,
    *,
    run_dir: Union[str, Path],
    seed: int,
    save_master: bool = False,
) -> tuple[
    GraphData,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build the master graph, sample one run graph, and construct GraphData.

    The same seed is used throughout this single-run graph pipeline, preserving
    the established single-driver RNG behavior.
    """
    master = read_or_build_master(
        contacts_df=contacts_df,
        cfg=cfg,
        run_dir=run_dir,
        seed=int(seed),
        save_master=save_master,
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
        seed=int(seed),
    )

    graphdata = build_graph_data(
        edge_list=sampled_edges,
        contacts_df=contacts_df,
        config=cfg,
        seed=int(seed),
        N=int(len(contacts_df)),
    )

    return graphdata, sampled_edges, master


def run_single_model(
    contacts_src: ContactsSource,
    cfg: ConfigSource,
    output_dir: Union[
        str,
        Path,
    ] = "model_runs",
    *,
    seed: Optional[int] = None,
    policy_name: Optional[str] = None,
    lhd_overrides: Optional[
        Dict[str, Any]
    ] = None,
    data_dir: str = "data",
    output_profile: str = "core",
    extra_outputs: Optional[
        Iterable[str]
    ] = None,
    save_master_graph: bool = False,
    overwrite: bool = False,
) -> NetworkModel:
    """
    Build and run one model.

    The fully populated model is returned regardless of the output profile.
    """
    cfg = _normalize_config(cfg)

    if seed is not None:
        cfg = cfg.copy_with(
            {
                "sim": {
                    "seed": int(seed),
                }
            }
        )

    if policy_name is not None or lhd_overrides:
        lhd_patch: Dict[str, Any] = {}

        if policy_name is not None:
            lhd_patch["policy_name"] = str(
                policy_name
            )

        if lhd_overrides:
            lhd_patch.update(
                dict(lhd_overrides)
            )

        cfg = cfg.copy_with(
            {
                "lhd": lhd_patch,
            }
        )

    cfg.validate()

    outputs = resolve_outputs(
        output_profile,
        extra_outputs,
    )

    contacts = load_contacts_source(
        contacts_src,
        cfg,
        data_dir=data_dir,
    )

    base_output_dir = Path(
        output_dir
    ).expanduser().resolve()
    run_dir = base_output_dir / str(cfg.sim.run_name)

    if outputs:
        _prepare_output_directory(
            run_dir,
            overwrite=overwrite,
        )

    # Preserve the original single-driver behavior: the configured simulation
    # seed controls both graph construction and the model base seed.
    run_seed = int(cfg.sim.seed)

    graphdata, sampled_edges, _master = (
        build_run_graphdata(
            contacts,
            cfg,
            run_dir=run_dir,
            seed=run_seed,
            save_master=save_master_graph,
        )
    )

    model = NetworkModel(
        config=cfg,
        graphdata=graphdata,
        run_dir=str(run_dir),
        seed=run_seed,
    )
    model.simulate()

    output_paths = _save_single_outputs(
        model=model,
        cfg=cfg,
        sampled_edges=sampled_edges,
        run_dir=run_dir,
        outputs=outputs,
    )

    # Attach runtime objects for interactive use without requiring disk reloads.
    model.run_dir = run_dir
    model.output_paths = output_paths
    model.run_graphdata = graphdata
    model.sampled_edges = sampled_edges

    return model


def _save_single_outputs(
    *,
    model: NetworkModel,
    cfg: ModelConfig,
    sampled_edges: pd.DataFrame,
    run_dir: Path,
    outputs: set[str],
) -> Dict[str, Path]:
    """Write only outputs selected by the output profile."""
    output_paths: Dict[str, Path] = {}

    if not outputs:
        return output_paths

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = run_dir / "ModelConfig.json"
    cfg.to_json(str(config_path))
    output_paths["config"] = config_path

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

    for output_name in sorted(outputs):
        if output_name == "sampled_edges":
            dataframe = sampled_edges
        else:
            producer = producers.get(output_name)

            if producer is None:
                continue

            dataframe = producer()

        if (
            not isinstance(dataframe, pd.DataFrame)
            or dataframe.empty
        ):
            continue

        path = (
            run_dir
            / OUTPUT_FILENAMES[output_name]
        )

        atomic_write_parquet(
            dataframe,
            path,
        )
        output_paths[output_name] = path

    return output_paths


def _normalize_config(
    cfg: ConfigSource,
) -> ModelConfig:
    if isinstance(cfg, ModelConfig):
        return cfg

    path = Path(cfg).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"ModelConfig JSON not found: {path}"
        )

    return ModelConfig.from_json(str(path))


def _prepare_output_directory(
    run_dir: Path,
    *,
    overwrite: bool,
) -> None:
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                "Use overwrite=True to replace it."
            )

        shutil.rmtree(run_dir)

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def _validate_contacts_df(
    dataframe: pd.DataFrame,
) -> None:
    required = {
        "PID",
        "hh_id",
        "wp_id",
        "sch_id",
        "gq_id",
        "age",
        "sex",
        "gq",
    }
    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Contacts DataFrame is missing: "
            f"{sorted(missing)}"
        )


if __name__ == "__main__":
    cfg = ModelConfig().copy_with(
        {
            "sim": {
                "county": "Keweenaw",
                "I0": 2,
                "n_replicates": 3,
                "simulation_duration": 60,
            },
            "lhd": {
                "policy_name":
                    "adaptive_escalation",
                "lhd_daily_capacity": 100.0,
            },
        }
    )

    model = run_single_model(
        contacts_src=cfg.sim.county,
        cfg=cfg,
        seed=13,
        policy_name="adaptive_escalation",
        output_profile="core",
        overwrite=True,
    )

    print(
        model.factual_run_summary_to_df()
    )