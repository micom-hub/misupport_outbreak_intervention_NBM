#Tests for features related to disease modeling and tracking

import numpy as np
import pandas as pd
import pytest

from scripts.network_model import NetworkModel, DefaultModelParams, ExposureEventRecorder

@pytest.fixture
def sample_contacts_df():
    # Small synthetic population with household/work/school ids
    N = 12
    ages = [70, 34, 29, 17, 15, 10, 45, 66, 30, 5, 80, 25]
    sexes = ["F","M","F","F","M","F","M","F","M","F","M","F"]
    races = ["Latino","White","Black","Asian","White","White","Black","White","Asian","Latino","White","Black"]
    # households: four households
    hh_ids = [1]*3 + [2]*3 + [3]*3 + [4]*3
    # workplaces: assign some X and Y and NaN
    wp_ids = [11,12,11,14, np.nan, np.nan, 12, 13, 11, np.nan, 13, 12]
    sch_ids = [np.nan, np.nan, np.nan, 21, 21, 21, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    gq_ids = [np.nan]*N
    gq_flags = [False]*N

    rows = []
    for i in range(N):
        rows.append({
            "age": ages[i],
            "sex": sexes[i],
            "race": races[i],
            "hh_id": hh_ids[i],
            "wp_id": wp_ids[i],
            "sch_id": sch_ids[i],
            "gq_id": gq_ids[i],
            "gq": gq_flags[i]
        })
    return pd.DataFrame(rows)

@pytest.fixture
def net_model(sample_contacts_df, tmp_path):
    params = DefaultModelParams.copy()
    params.update({
        "I0": [1],
        "simulation_duration": 3,
        "seed": 42,
        "overwrite_edge_list": True,
        "save_data_files": False,
        "run_name": "test_model_basic",
        "record_exposure_events": True,
        "n_runs": 1,
        "base_transmission_prob": 1,  # deterministic high transmission for tests
    })
    model = NetworkModel(contacts_df=sample_contacts_df, params=params)
    model.results_folder = str(tmp_path)
    return model

def test_network_structures_build(net_model):
    m = net_model
    # Basic attributes exist
    assert hasattr(m, "edge_list") and isinstance(m.edge_list, pd.DataFrame)
    assert hasattr(m, "csr_by_type") and isinstance(m.csr_by_type, dict)
    assert hasattr(m, "contact_types")
    assert isinstance(m.contact_types, list)

def test_initialize_states_and_vaccination(net_model):
    m = net_model
    m._initialize_states()
    assert hasattr(m, "is_vaccinated")
    assert m.is_vaccinated.shape[0] == m.N
    for ind in m.params["I0"]:
        assert m.state[int(ind)] == 2

def test_recorder_and_determine_new_exposures(net_model):
    m = net_model
    m._initialize_states()
    m.current_time = 0
    rec = ExposureEventRecorder(init_event_cap=32, init_node_cap=128)
    newly = m.determine_new_exposures(recorder=rec)
    snap = rec.snapshot_compact(copy=True)
    # snapshot consistency
    assert snap['event_time'].shape[0] == rec.n_events
    assert snap['nodes'].shape[0] == rec.n_nodes
    assert snap['event_nodes_len'].sum() == rec.n_nodes
    # newly is consistent with infections in snapshot
    infected_nodes = np.unique(snap['nodes'][snap['infections']]) if rec.n_nodes > 0 else np.array([], dtype=int)
    assert set(newly.tolist()) == set(infected_nodes.tolist())

def test_step_and_simulate_run(net_model):
    m = net_model
    # one manual step with recorder
    m._initialize_states()
    rec = m.recorder_template
    rec.reset()
    m.step(rec)
    # day recorded
    assert len(m.states_over_time) >= 1
    # small simulate run (smoke)
    m.simulate()
    assert m.all_states_over_time[0]  # should exist
    # epi_outcomes returns DataFrame
    df = m.epi_outcomes()
    assert isinstance(df, pd.DataFrame)
    assert "total_infections" in df.columns