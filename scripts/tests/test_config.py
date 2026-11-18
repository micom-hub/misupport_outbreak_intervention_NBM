from __future__ import annotations

import json

import pytest

from scripts.config import ModelConfig


def test_default_configuration_validates():
    ModelConfig().validate()


def test_copy_with_changes_only_requested_fields():
    original = ModelConfig()

    modified = original.copy_with(
        {
            "epi": {
                "base_transmission_prob": 0.10,
            },
            "lhd": {
                "lhd_daily_capacity": 50.0,
            },
        }
    )

    assert (
        modified.epi.base_transmission_prob
        == 0.10
    )
    assert modified.lhd.lhd_daily_capacity == 50.0

    assert (
        modified.epi.infectious_period
        == original.epi.infectious_period
    )
    assert (
        modified.population
        == original.population
    )
    assert modified.sim == original.sim

    assert (
        original.epi.base_transmission_prob
        != 0.10
    )


def test_json_round_trip_preserves_configuration(
    tmp_path,
):
    config = ModelConfig().copy_with(
        {
            "epi": {
                "base_transmission_prob": 0.12,
            },
            "sim": {
                "I0": 3,
                "seed": 999,
            },
        }
    )

    path = tmp_path / "config.json"
    config.to_json(path)

    loaded = ModelConfig.from_json(path)

    assert loaded == config
    assert loaded.to_dict() == config.to_dict()


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_transmission_prob", -0.01),
        ("base_transmission_prob", 1.01),
        ("vax_uptake", -0.1),
        ("vax_efficacy", 1.1),
        ("symptomatic_fraction", 1.5),
    ],
)
def test_invalid_epi_probabilities_rejected(
    field,
    value,
):
    config = ModelConfig().copy_with(
        {
            "epi": {
                field: value,
            }
        }
    )

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("latent_period", 0.0),
        ("latent_period", -1.0),
        ("infectious_period", 0.0),
        ("infectious_period_vax", -2.0),
        ("gamma_alpha", 0.0),
        ("symptom_onset_shape", 0.0),
    ],
)
def test_nonpositive_disease_parameters_rejected(
    field,
    value,
):
    config = ModelConfig().copy_with(
        {
            "epi": {
                field: value,
            }
        }
    )

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("lhd", "policy_name", "not_a_policy"),
        (
            "lhd",
            "escalation_trigger_metric",
            "not_a_trigger",
        ),
        (
            "lhd",
            "escalation_intervention_profile",
            "not_a_profile",
        ),
        (
            "lhd",
            "phase2_trace_mode",
            "not_a_mode",
        ),
    ],
)
def test_invalid_named_options_rejected(
    section,
    field,
    value,
):
    config = ModelConfig().copy_with(
        {
            section: {
                field: value,
            }
        }
    )

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "value",
    [-1, 1.5, True],
)
def test_i0_must_be_nonnegative_integer(value):
    config = ModelConfig().copy_with(
        {
            "sim": {
                "I0": value,
            }
        }
    )

    with pytest.raises(ValueError):
        config.validate()


def test_casual_candidates_must_cover_requested_contacts():
    config = ModelConfig().copy_with(
        {
            "population": {
                "cas_contacts": 20,
            },
            "sim": {
                "master_casual_candidates": 10,
            },
        }
    )

    with pytest.raises(ValueError):
        config.validate()