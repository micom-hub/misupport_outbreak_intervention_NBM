from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.lhd.policy_config import (
    PolicyConfig,
    PolicyVariant,
)
from scripts.singledriver import run_single_model
from scripts.variantdriver import (
    run_experiment,
    run_policy_comparison,
)


def minimal_policy_config():
    return PolicyConfig(
        variants=[
            PolicyVariant(
                name="observe_only",
                policy_name="observe_only",
            ),
            PolicyVariant(
                name="case_isolation_only",
                policy_name="case_isolation_only",
            ),
        ]
    )


def patch_single_graph_builder(
    monkeypatch,
    toy_model_factory,
):
    def fake_build_run_graphdata(
        contacts_df,
        cfg,
        *,
        run_dir,
        seed,
        save_master=False,
    ):
        return (
            toy_model_factory.graphdata,
            toy_model_factory.edge_list.copy(),
            toy_model_factory.edge_list.copy(),
        )

    monkeypatch.setattr(
        "scripts.singledriver.build_run_graphdata",
        fake_build_run_graphdata,
    )


def patch_policy_comparison_graph_pipeline(
    monkeypatch,
    toy_model_factory,
):
    monkeypatch.setattr(
        "scripts.singledriver.read_or_build_master",
        lambda **kwargs:
            toy_model_factory.edge_list.copy(),
    )

    monkeypatch.setattr(
        "scripts.graph.graph_utils."
        "build_minimal_graphdata_from_edge_list",
        lambda edge_list, N: {
            "N": N,
            "edge_list": edge_list,
        },
    )

    monkeypatch.setattr(
        "scripts.graph.graph_utils."
        "sample_from_master_graphdata",
        lambda *args, **kwargs:
            toy_model_factory.edge_list.copy(),
    )

    monkeypatch.setattr(
        "scripts.graph.graph_utils.build_graph_data",
        lambda *args, **kwargs:
            toy_model_factory.graphdata,
    )


def test_single_none_returns_model_without_outputs(
    monkeypatch,
    toy_model_factory,
    tmp_path,
):
    patch_single_graph_builder(
        monkeypatch,
        toy_model_factory,
    )

    config = toy_model_factory.config(
        sim_overrides={
            "run_name": "no_output",
        },
    )

    output_root = tmp_path / "runs"

    model = run_single_model(
        contacts_src=toy_model_factory.contacts,
        cfg=config,
        output_dir=output_root,
        seed=123,
        output_profile="none",
    )

    assert model.all_daily_epi[0] is not None
    assert model.output_paths == {}
    assert not (
        output_root / "no_output"
    ).exists()


def test_core_profile_writes_only_core_files(
    monkeypatch,
    toy_model_factory,
    tmp_path,
):
    patch_single_graph_builder(
        monkeypatch,
        toy_model_factory,
    )

    config = toy_model_factory.config(
        sim_overrides={
            "run_name": "core_output",
        },
    )

    model = run_single_model(
        contacts_src=toy_model_factory.contacts,
        cfg=config,
        output_dir=tmp_path,
        seed=123,
        output_profile="core",
        overwrite=True,
    )

    filenames = {
        path.name
        for path in model.run_dir.iterdir()
        if path.is_file()
    }

    assert filenames == {
        "ModelConfig.json",
        "factual_run_summary.parquet",
        "daily_epi.parquet",
        "daily_lhd.parquet",
    }


def test_policy_comparison_returns_named_paired_models(
    monkeypatch,
    toy_model_factory,
):
    patch_policy_comparison_graph_pipeline(
        monkeypatch,
        toy_model_factory,
    )

    config = toy_model_factory.config(
        sim_overrides={
            "n_replicates": 2,
        },
    )

    comparison = run_policy_comparison(
        contacts_df=toy_model_factory.contacts,
        cfg=config,
        policy_config=minimal_policy_config(),
        seed=123,
        output_profile="none",
    )

    assert set(comparison.models) == {
        "observe_only",
        "case_isolation_only",
    }

    observe = comparison.models["observe_only"]
    isolation = comparison.models[
        "case_isolation_only"
    ]

    for run in range(config.sim.n_replicates):
        assert (
            observe.all_seed_metadata[run]
            ["replicate_seed"]
            == isolation.all_seed_metadata[run]
            ["replicate_seed"]
        )
        np.testing.assert_array_equal(
            observe.all_imported_ids[run],
            isolation.all_imported_ids[run],
        )
        np.testing.assert_array_equal(
            observe.all_vax_status[run],
            isolation.all_vax_status[run],
        )
        assert (
            observe.all_states_over_time[run][0]
            == isolation.all_states_over_time[run][0]
        )


def test_repeated_run_with_same_seed_reproduces_summary(
    toy_model_factory,
):
    first = toy_model_factory.model(
        policy_name="adaptive_escalation",
        transmission_probability=0.25,
        seed=555,
        simulate=True,
    )
    second = toy_model_factory.model(
        policy_name="adaptive_escalation",
        transmission_probability=0.25,
        seed=555,
        simulate=True,
    )

    pd.testing.assert_frame_equal(
        first.factual_run_summary_to_df(),
        second.factual_run_summary_to_df(),
    )


def test_return_models_requires_sequential_execution(
    toy_model_factory,
    tmp_path,
):
    parameter_file = tmp_path / "ranges.csv"

    pd.DataFrame(
        {
            "parameter": [
                "epi.base_transmission_prob"
            ],
            "minimum": [0.0],
            "maximum": [0.1],
            "integer": [0],
        }
    ).to_csv(parameter_file, index=False)

    with pytest.raises(ValueError):
        run_experiment(
            csv_path=str(parameter_file),
            n_samples=1,
            policy_config=minimal_policy_config(),
            output_dir=tmp_path / "experiment",
            base_cfg=toy_model_factory.config(),
            workers=2,
            output_profile="core",
            return_models=True,
        )


def test_experiment_aggregation_contains_pairing_fields(
    monkeypatch,
    toy_model_factory,
    tmp_path,
):
    config = toy_model_factory.config(
        sim_overrides={
            "n_replicates": 1,
        },
    )

    parameter_file = tmp_path / "ranges.csv"
    pd.DataFrame(
        {
            "parameter": [
                "epi.base_transmission_prob"
            ],
            "minimum": [0.0],
            "maximum": [0.0],
            "integer": [0],
        }
    ).to_csv(parameter_file, index=False)

    monkeypatch.setattr(
        "scripts.variantdriver.prepare_run",
        lambda **kwargs: (
            toy_model_factory.contacts.copy(),
            [config],
            {"toy": True},
        ),
    )

    monkeypatch.setattr(
        "scripts.variants.run_variants_funcs."
        "sample_from_master_graphdata",
        lambda *args, **kwargs:
            toy_model_factory.edge_list.copy(),
    )
    monkeypatch.setattr(
        "scripts.variants.run_variants_funcs."
        "build_graph_data",
        lambda *args, **kwargs:
            toy_model_factory.graphdata,
    )

    result = run_experiment(
        csv_path=str(parameter_file),
        n_samples=1,
        policy_config=minimal_policy_config(),
        output_dir=tmp_path / "experiment",
        base_cfg=config,
        seed=123,
        workers=1,
        output_profile="core",
        return_models=True,
        clean_output_dir=True,
        overwrite_run_dirs=True,
    )

    factual = pd.read_parquet(
        result.aggregated_paths[
            "factual_summary"
        ]
    )

    assert {
        "model_index",
        "run_number",
        "pair_id",
        "variant_name",
        "policy_name",
    } <= set(factual.columns)

    policy_counts = (
        factual.groupby("pair_id")
        ["variant_name"]
        .nunique()
    )

    assert (
        policy_counts
        == len(
            minimal_policy_config().variants
        )
    ).all()


def test_simulation_failure_is_recorded_in_status(
    monkeypatch,
    toy_model_factory,
    tmp_path,
):
    config = toy_model_factory.config()
    parameter_file = tmp_path / "ranges.csv"

    pd.DataFrame(
        {
            "parameter": [
                "epi.base_transmission_prob"
            ],
            "minimum": [0.0],
            "maximum": [0.0],
            "integer": [0],
        }
    ).to_csv(parameter_file, index=False)

    monkeypatch.setattr(
        "scripts.variantdriver.prepare_run",
        lambda **kwargs: (
            toy_model_factory.contacts.copy(),
            [config],
            {"toy": True},
        ),
    )

    def fail_parameter_set(*args, **kwargs):
        raise RuntimeError(
            "intentional simulation failure"
        )

    monkeypatch.setattr(
        "scripts.variantdriver.run_parameter_set",
        fail_parameter_set,
    )

    result = run_experiment(
        csv_path=str(parameter_file),
        n_samples=1,
        policy_config=minimal_policy_config(),
        output_dir=tmp_path / "failure_experiment",
        base_cfg=config,
        seed=123,
        workers=1,
        output_profile="core",
        return_models=False,
        clean_output_dir=True,
    )

    assert len(result.statuses) == 1
    assert result.statuses[0]["success"] is False
    assert (
        "intentional simulation failure"
        in result.statuses[0]["error"]
    )