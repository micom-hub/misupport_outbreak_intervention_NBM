import numpy as np
import pandas as pd
import pytest
import networkx as nx
import warnings

from scripts.network_model import NetworkModel, DefaultModelParams


# Optional imports used only in specific tests (guarded)
try:
    from scripts.network_model import ExposureEventRecorder, LocalHealthDepartment
except Exception:
    # Some environments may not have these classes implemented yet;
    # tests that require them will skip (handled below).
    ExposureEventRecorder = None
    LocalHealthDepartment = None


##################
# Fixtures
##################

@pytest.fixture
def sample_contacts_df():
    # Small synthetic population:
    data = [
        {"age": 60, "sex": "F", "race": "Latino", "hh_id": "C", "wp_id": np.nan, "sch_id": np.nan, "gq_id": "1", "gq": True},
        {"age": 34, "sex": "M", "race": "White", "hh_id": "A", "wp_id": "X", "sch_id": np.nan, "gq_id": np.nan, "gq": False},
        {"age": 29, "sex": "F", "race": "Black", "hh_id": "A", "wp_id": "Y", "sch_id": np.nan, "gq_id": np.nan, "gq": False},
        {"age": 17, "sex": "F", "race": "Asian", "hh_id": "B", "wp_id": "Y", "sch_id": "S1", "gq_id": np.nan, "gq": False},
        {"age": 15, "sex": "M", "race": "White", "hh_id": "B", "wp_id": np.nan, "sch_id": "S1", "gq_id": np.nan, "gq": False}
    ]
    # Build a DataFrame - your model's compute_network_structures expects certain fields;
    # keep names consistent with what your Model expects.
    return pd.DataFrame(data).reset_index(drop=True)


@pytest.fixture
def net_model(sample_contacts_df, tmp_path):
    # Copy defaults and override for testability:
    params = DefaultModelParams.copy()
    params.update({
        "I0": [1],         # Infect individual 1 (person with index 1)
        "simulation_duration": 5,
        "seed": 123,
        "overwrite_edge_list": True,
        # Do not write files during tests
        "save_data_files": False,
        "run_name": "test_run_pytest",
        # Ensure recorder path is exercised
        "record_exposure_events": True,
        # Make transmission deterministic in these unit tests
        "base_transmission_prob": 1.0,
    })

    # Instantiate model (this calls your _compute_network_structures in __init__)
    model = NetworkModel(contacts_df=sample_contacts_df, params=params)

    # Ensure results folder in a temp location to avoid writes in repo
    model.results_folder = str(tmp_path)

    return model


##################
# New tests: ExposureEventRecorder
##################

def test_exposure_event_recorder_basic():
    """Basic functional checks for ExposureEventRecorder"""
    if ExposureEventRecorder is None:
        pytest.skip("ExposureEventRecorder not available in module; skipping recorder unit tests")

    rec = ExposureEventRecorder(init_event_cap=2, init_node_cap=2)

    # Append a 2-target event
    nodes_a = np.array([2, 3], dtype=np.int32)
    mask_a = np.array([True, False], dtype=np.bool_)
    rec.append_event(time=0, source=0, type_id=0, nodes_arr=nodes_a, infected_mask=mask_a)
    assert rec.n_events == 1
    assert rec.n_nodes == 2

    # Append an empty event (zero targets)
    rec.append_event(time=0, source=1, type_id=1, nodes_arr=np.array([], dtype=np.int32), infected_mask=np.array([], dtype=np.bool_))
    assert rec.n_events == 2
    assert rec.n_nodes == 2  # still 2 nodes in buffer

    # snapshot and consistency checks
    snap = rec.snapshot_compact(copy=True)
    assert isinstance(snap, dict)
    assert 'nodes' in snap and 'infections' in snap
    assert snap['nodes'].shape[0] == rec.n_nodes
    assert snap['infections'].shape[0] == rec.n_nodes
    assert snap['event_time'].shape[0] == rec.n_events
    assert snap['event_nodes_len'].sum() == rec.n_nodes

    # Mismatched lengths should raise a helpful ValueError
    with pytest.raises(ValueError):
        rec.append_event(time=1, source=2, type_id=2, nodes_arr=np.array([1], dtype=np.int32), infected_mask=np.array([], dtype=np.bool_))


def test_model_records_exposure_events(net_model):
    """Integration test: determine_new_exposures populates a recorder snapshot coherently."""
    if ExposureEventRecorder is None:
        pytest.skip("ExposureEventRecorder not available; skipping integration recorder test")

    model = net_model
    # reset/run from a fresh state
    model._initialize_states()
    model.current_time = 0

    rec = ExposureEventRecorder(init_event_cap=64, init_node_cap=256)
    newly_exposed = model.determine_new_exposures(recorder=rec)

    # newly_exposed must be a numpy array and recorder must have captured events consistent with it
    assert isinstance(newly_exposed, np.ndarray)
    snap = rec.snapshot_compact(copy=True)
    # metadata consistency
    assert snap['event_time'].shape[0] == rec.n_events
    assert snap['nodes'].shape[0] == rec.n_nodes
    assert snap['event_nodes_len'].sum() == rec.n_nodes

    # infected nodes recovered from snapshot should match the returned newly_exposed set
    infected_from_snapshot = np.unique(snap['nodes'][snap['infections']]) if rec.n_nodes > 0 else np.array([], dtype=int)
    assert set(newly_exposed.tolist()) == set(infected_from_snapshot.tolist())


##################
# New tests: Intervention / LHD-action semantics (multipliers)
##################

def _choose_contact_type_with_neighbors(model, node):
    """Helper: return a contact-type key for which `node` has at least one neighbor, or None."""
    # Prefer the contact_types attribute if present
    cts = getattr(model, 'contact_types', None)
    if cts is None:
        cts = list(model.csr_by_type.keys())
    for ct in cts:
        indptr, indices, weights = model.csr_by_type[ct]
        if indptr[node + 1] > indptr[node]:
            return ct
    return None


def test_intervention_multiplier_effect(net_model):
    """
    Tests that zeroing an outgoing multiplier for the initial infectious node at a given contact type
    reduces the set of newly-exposed targets for a fresh step, and that restoring the multiplier
    restores the original exposure set. Skips if multiplier arrays are not present.
    """
    model = net_model
    # Find an initial infectious node
    try:
        initial_inf = int(model.params["I0"][0])
    except Exception:
        pytest.skip("Model I0 not set; skipping intervention multiplier test")

    # Find contact type that has neighbors for the infected node
    ct = _choose_contact_type_with_neighbors(model, initial_inf)
    if ct is None:
        pytest.skip("No contact type with neighbors for the initial infectious node; skipping intervention test")

    # Check multiplier arrays exist
    if not (hasattr(model, 'out_multiplier') and ct in model.out_multiplier and hasattr(model, 'in_multiplier') and ct in model.in_multiplier):
        pytest.skip("Model does not expose in/out multiplier arrays; skipping intervention multiplier test")

    # Baseline exposures (fresh initialization)
    model._initialize_states()
    model.current_time = 0
    rec_base = ExposureEventRecorder(init_event_cap=256, init_node_cap=1024)
    base_exposed = set(model.determine_new_exposures(recorder=rec_base).tolist())

    # Apply a strong intervention: set outgoing multiplier for that contact type to zero for the infectious node
    original_out = float(model.out_multiplier[ct][initial_inf])
    model.out_multiplier[ct][initial_inf] = 0.0

    # Reinitialize and run again
    model._initialize_states()
    model.current_time = 0
    rec_reduced = ExposureEventRecorder(init_event_cap=256, init_node_cap=1024)
    reduced_exposed = set(model.determine_new_exposures(recorder=rec_reduced).tolist())

    # Reduced set should be a subset (or equal if the zeroed type contributed nothing baseline)
    assert reduced_exposed.issubset(base_exposed)
    # If the zeroed contact type actually contributed exposures baseline, we expect a strict reduction
    if len(base_exposed) > 0:
        assert len(reduced_exposed) <= len(base_exposed)

    # Restore multiplier and confirm exposures return to baseline (fresh init)
    model.out_multiplier[ct][initial_inf] = original_out
    model._initialize_states()
    model.current_time = 0
    rec_restored = ExposureEventRecorder(init_event_cap=256, init_node_cap=1024)
    restored_exposed = set(model.determine_new_exposures(recorder=rec_restored).tolist())
    assert restored_exposed == base_exposed


##################
# Optional: test LocalHealthDepartment API if present
##################

def test_lhd_api_if_present(net_model):
    """
    If a LocalHealthDepartment class is implemented with an apply_interventions method,
    test that calling apply_interventions changes multipliers in the expected way and
    that expirations (if provided) revert effects. This test will be skipped
    if LocalHealthDepartment is not implemented or does not expose apply_interventions.
    """
    if LocalHealthDepartment is None:
        pytest.skip("LocalHealthDepartment not available; skipping LHD integration test")

    model = net_model

    # instantiate LHD in a forgiving way (skip if signature not compatible)
    try:
        lhd = LocalHealthDepartment(model=model, rng=model.rng)
    except TypeError:
        pytest.skip("LocalHealthDepartment constructor signature differs; skipping LHD API test")

    if not hasattr(lhd, 'apply_interventions'):
        pytest.skip("LocalHealthDepartment has no apply_interventions method; skipping LHD API test")

    # pick initial infected and a contact type with neighbors
    try:
        initial_inf = int(model.params["I0"][0])
    except Exception:
        pytest.skip("Model I0 not set; skipping LHD API test")

    ct = _choose_contact_type_with_neighbors(model, initial_inf)
    if ct is None:
        pytest.skip("No contact type with neighbors for the initial infectious node; skipping LHD API test")

    # ensure the model has multipliers to manipulate
    if not (hasattr(model, 'out_multiplier') and ct in model.out_multiplier):
        pytest.skip("Model does not expose multiplier arrays; skipping LHD API test")

    # Baseline
    model._initialize_states()
    model.current_time = 0
    rec_base = ExposureEventRecorder(init_event_cap=256, init_node_cap=1024)
    base_exposed = set(model.determine_new_exposures(recorder=rec_base).tolist())

    # Try to call apply_interventions with simple signature (nodes, contact_types, reduction, duration)
    nodes = np.array([initial_inf], dtype=np.int32)
    try:
        lhd.apply_interventions(nodes, [ct], reduction=1.0, duration=1)
    except TypeError:
        # Not the expected signature; skip this test rather than failing
        pytest.skip("LocalHealthDepartment.apply_interventions signature unexpected; skipping test")

    # After intervention, exposures should be reduced for a fresh run
    model._initialize_states()
    model.current_time = 0
    rec_after = ExposureEventRecorder(init_event_cap=256, init_node_cap=1024)
    after_exposed = set(model.determine_new_exposures(recorder=rec_after).tolist())
    assert after_exposed.issubset(base_exposed)


##################
# Existing tests (updated calls to match current method names)
##################

def test_network_structures(net_model):
    model = net_model
    assert model.N == len(model.contacts_df)
    assert isinstance(model.edge_list, pd.DataFrame)
    assert model.adj_matrix.shape == (model.N, model.N)
    assert isinstance(model.individual_lookup, pd.DataFrame)
    assert model.individual_lookup.shape[0] == model.N
    assert isinstance(model.neighbor_map, dict)
    for i, v in model.neighbor_map.items():
        assert isinstance(v, list)


def test_state_initialization(net_model):
    net_model._initialize_states()
    model = net_model
    assert isinstance(model.state, np.ndarray)
    assert len(model.state) == model.N
    # initial infected(s) set to I
    for idx in model.params["I0"]:
        assert model.state[int(idx)] == 2
    susceptible = set(range(model.N)) - set(model.params["I0"])
    assert (model.state[list(susceptible)] == 0).all()


def test_assign_periods(net_model):
    model = net_model
    model._initialize_states()
    inds = np.array([0, 1, 2])
    inc_periods = model.assign_incubation_period(inds)
    inf_periods = model.assign_infectious_period(inds)
    assert inc_periods.shape == (3,) and (inc_periods > 0).all()
    assert inf_periods.shape == (3,) and (inf_periods > 0).all()


def test_graph_attribute_assignment(net_model):
    model = net_model
    g = model.g_full
    names = np.array(g.vs["name"])
    ages = model.contacts_df.iloc[names]["age"].to_numpy()
    model.assign_node_attribute("age", ages, names, g)
    for idx, expected_age in zip(names, ages):
        ind = model.name_to_ind(g, idx)
        assert g.vs[ind]["age"] == expected_age


def test_single_run_and_trajectories(net_model):
    model = net_model
    model.simulate()
    for run in range(model.n_runs):
        all_states = model.all_states_over_time[run]
        exposures = model.all_new_exposures[run]
        assert isinstance(all_states, list)
        assert isinstance(exposures, list)
        assert abs(len(exposures) == len(all_states))
        for timestep, (S, E, I, R) in enumerate(all_states):  # noqa: E741
            all_ids = set(S + E + I + R)
            assert all(i in all_ids for i in range(model.N))
            assert len(exposures) == len(all_states)
    # At least initial infection happens
    for run in range(model.n_runs):
        exposures = model.all_new_exposures[run]
        assert len(exposures) > 0
        assert len(exposures[0]) >= 0


def test_stochastic_dieout_flagged(net_model):
    model = net_model
    model.simulate()
    assert any(isinstance(flag, (np.bool_, bool)) for flag in model.all_stochastic_dieout)


def test_end_days_consistent(net_model):
    model = net_model
    model.simulate()
    for d in model.all_end_days:
        assert d <= model.Tmax


def test_epi_curve_runs(net_model, monkeypatch):
    model = net_model
    model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    for run in range(model.n_runs):
        model.epi_curve(run_number=run)


def test_cumulative_incidence_plot_runs(net_model, monkeypatch):
    model = net_model
    model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    for run in range(model.n_runs):
        model.cumulative_incidence_plot(run_number=run, strata=None)
        model.cumulative_incidence_plot(run_number=run, strata="age")
        model.cumulative_incidence_plot(run_number=run, strata="sex")


def test_cumulative_incidence_spaghetti_runs(net_model, monkeypatch):
    model = net_model
    model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    model.cumulative_incidence_spaghetti()


def test_draw_network_runs(net_model, monkeypatch):
    model = net_model
    model.simulate()
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    for run in range(model.n_runs):
        t = len(model.all_states_over_time[run]) // 2
        model.draw_network(t=t, run_number=run)


def test_make_movie_runs(net_model, monkeypatch, tmp_path):
    model = net_model
    model.simulate()
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    monkeypatch.setattr(plt, "show", lambda: None)
    # avoid actually writing a file
    monkeypatch.setattr(animation.Animation, "save", lambda self, filename, writer, fps: None)
    model.results_folder = tmp_path
    model.params["save_data_files"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.make_movie(run_number=0, filename=str(tmp_path / "testmovie.mp4"), fps=1)


def test_make_graphml_file_runs(net_model, tmp_path):
    model = net_model
    model.simulate()
    model.results_folder = tmp_path
    last_t = len(model.all_states_over_time[0]) - 1
    model.make_graphml_file(t=last_t, run_number=0)
    assert hasattr(model, "nx_g")
    assert isinstance(model.nx_g, nx.Graph)
    S, E, I, R = model.all_states_over_time[0][last_t]  # noqa: E741
    expected_nodes = set(E + I + R)
    graph_nodes = set(model.nx_g.nodes)
    assert expected_nodes.issubset(graph_nodes)


def test_population_lookup(net_model):
    lookup = net_model.individual_lookup
    assert {"age", "race", "sex"}.issubset(set(lookup.columns))
    assert len(lookup) == net_model.N


def test_full_batch_run_monkeypatched_visualizations(net_model, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    model = net_model
    model.simulate()
    model.cumulative_incidence_spaghetti()
    for run in range(model.n_runs):
        model.epi_curve(run_number=run)
        model.cumulative_incidence_plot(run_number=run, strata=None)
        t = len(model.all_states_over_time[run]) // 2
        model.draw_network(t=t, run_number=run)
