from __future__ import annotations

import numpy as np


def test_imported_cases_start_infectious_generation_zero(
    toy_model_factory,
):
    model = toy_model_factory.model(
        transmission_probability=0.0,
        simulate=False,
    )
    model._initialize_replicate(0)

    imported = model.imported_ids

    assert imported.size == 1
    assert np.all(model.state[imported] == 2)
    assert np.all(
        model.infection_generation[imported]
        == 0
    )
    assert np.all(
        model.infection_day[imported] == 0
    )
    assert np.all(
        model.infectious_onset_day[imported]
        == 0
    )


def test_imports_excluded_from_local_incidence_and_size(
    toy_model_factory,
):
    model = toy_model_factory.model(
        transmission_probability=0.0,
        simulate=True,
    )

    summary = model.results_to_df()
    daily = model.daily_epi_to_df()
    incidence = model.timeseries_to_df(
        "incidence"
    )

    assert summary.loc[
        0,
        "outbreakSize",
    ] == 0
    assert (
        daily["cumulative_local_infections"]
        == 0
    ).all()
    assert (
        incidence.filter(like="t_")
        .to_numpy()
        == 0
    ).all()


def test_guaranteed_transmission_infects_neighbors(
    toy_model_factory,
):
    model = toy_model_factory.model(
        transmission_probability=1.0,
        simulate=False,
    )
    model._initialize_replicate(0)

    imported = int(model.imported_ids[0])
    expected_neighbors = {
        target
        for target, _weight, _contact_type
        in model.neighbor_map[imported]
    }

    sources, targets, _types = (
        model.determine_transmissions()
    )

    assert set(sources.tolist()) == {imported}
    assert set(targets.tolist()) == expected_neighbors


def test_zero_transmission_produces_no_local_infections(
    toy_model_factory,
):
    model = toy_model_factory.model(
        transmission_probability=0.0,
        simulate=True,
    )

    generations = (
        model.all_infection_generation[0]
    )

    assert np.count_nonzero(generations > 0) == 0


def test_local_infections_have_valid_infector_and_generation(
    toy_model_factory,
):
    model = toy_model_factory.model(
        transmission_probability=1.0,
        sim_overrides={
            "simulation_duration": 3,
        },
        simulate=True,
    )

    generations = (
        model.all_infection_generation[0]
    )
    infectors = model.all_infector[0]

    for node in np.where(generations > 0)[0]:
        source = int(infectors[node])

        assert source >= 0
        assert (
            generations[node]
            == generations[source] + 1
        )


def test_natural_history_event_ordering(
    toy_model_factory,
):
    model = toy_model_factory.model(
        transmission_probability=1.0,
        sim_overrides={
            "simulation_duration": 8,
        },
        simulate=True,
    )

    infection_day = model.all_infection_day[0]
    infectious_day = (
        model.all_infectious_onset_day[0]
    )
    recovery_day = model.all_recovery_day[0]
    symptom_day = model.all_symptom_onset_day[0]
    generations = (
        model.all_infection_generation[0]
    )

    for node in np.where(generations >= 0)[0]:
        assert infection_day[node] >= 0

        if infectious_day[node] >= 0:
            assert (
                infectious_day[node]
                >= infection_day[node]
            )

        if recovery_day[node] >= 0:
            assert infectious_day[node] >= 0
            assert (
                recovery_day[node]
                >= infectious_day[node]
            )

        if symptom_day[node] >= 0:
            assert infectious_day[node] >= 0
            assert (
                symptom_day[node]
                >= infectious_day[node]
            )


def test_asymptomatic_people_have_no_symptom_events(
    toy_model_factory,
):
    model = toy_model_factory.model(
        transmission_probability=1.0,
        epi_overrides={
            "symptomatic_fraction": 0.0,
        },
        simulate=True,
    )

    events = model.event_log_to_df()

    assert not (
        events["event_type"] == "symptom_onset"
    ).any()
    assert np.all(
        model.all_symptom_onset_day[0] == -1
    )


def test_daily_compartment_accounting(
    toy_model_factory,
):
    model = toy_model_factory.model(
        transmission_probability=1.0,
        simulate=True,
    )

    daily = model.daily_epi_to_df()

    accounted = (
        daily["local_exposed"]
        + daily["local_infectious"]
        + daily["local_recovered"]
    )

    np.testing.assert_array_equal(
        accounted.to_numpy(),
        daily[
            "cumulative_local_infections"
        ].to_numpy(),
    )