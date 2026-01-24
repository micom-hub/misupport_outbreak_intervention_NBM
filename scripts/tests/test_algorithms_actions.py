#Tests for validity of LHD algorithms and actions

import numpy as np
import pandas as pd
import pytest

from scripts.network_model import (
    NetworkModel, DefaultModelParams, CallIndividualsAction,
    EqualPriority, RandomPriority, PrioritizeElders, ExposureEventRecorder
)

@pytest.fixture
def contacts_50():
    rng = np.random.RandomState(123)
    N = 50
    ages = rng.choice(range(1,90), size=N)
    sexes = rng.choice(['F','M'], size=N)
    races = rng.choice(['White','Black','Asian','Latino'], size=N)
    # household grouping: groups sizes 2-4 randomly
    hh = []
    cur = 0
    hh_id = 0
    while cur < N:
        size = rng.choice([2,3,4])
        for _ in range(min(size, N-cur)):
            hh.append(f"HH{hh_id}")
            cur += 1
        hh_id += 1
    # workplaces: assign 10 workplaces
    wp_ids = [f"W{rng.randint(0,10)}" if rng.rand() < 0.7 else None for _ in range(N)]
    # schools for younger ages
    sch_ids = [f"S{rng.randint(0,5)}" if age < 18 else None for age in ages]
    rows = []
    for i in range(N):
        rows.append({
            "age": int(ages[i]),
            "sex": sexes[i],
            "race": races[i],
            "hh_id": hh[i],
            "wp_id": wp_ids[i] if wp_ids[i] is not None else np.nan,
            "sch_id": sch_ids[i] if sch_ids[i] is not None else np.nan,
            "gq_id": np.nan,
            "gq": False
        })
    return pd.DataFrame(rows)

@pytest.fixture
def model_50(contacts_50, tmp_path):
    params = DefaultModelParams.copy()
    params.update({
        "I0": [0,1,2],
        "simulation_duration": 2,
        "seed": 2023,
        "overwrite_edge_list": True,
        "save_data_files": False,
        "run_name": "test_alg_action",
        "record_exposure_events": True,
        "n_runs": 1,
        "base_transmission_prob": 0.5,  # non-extreme
    })
    m = NetworkModel(contacts_df=contacts_50, params=params)
    m.results_folder = str(tmp_path)
    # deterministic rng for tests
    m.rng = np.random.default_rng(2023)
    return m

def make_snapshot(m, nodes):
    nodes = np.asarray(nodes, dtype=np.int32)
    return {
        'event_time': np.array([0], dtype=np.int32),
        'event_source': np.array([0], dtype=np.int32),
        'event_type': np.array([0], dtype=np.int16),
        'event_nodes_start': np.array([0], dtype=np.int64),
        'event_nodes_len': np.array([len(nodes)], dtype=np.int32),
        'nodes': nodes,
        'infections': np.zeros(len(nodes), dtype=bool)
    }

def test_equalpriority_alg(model_50):
    m = model_50
    alg = EqualPriority()
    snap = make_snapshot(m, [3,10,20,30])
    out = alg.generate_candidates(snap, m, np.array([0]))
    assert out['nodes'].shape[0] == 4
    assert np.all(out['priority'] == 1.0)

def test_randompriority_alg(model_50):
    m = model_50
    alg = RandomPriority()
    snap = make_snapshot(m, [5,6,7])
    out = alg.generate_candidates(snap, m, np.array([0]))
    assert out['nodes'].shape[0] == 3
    assert out['priority'].dtype == np.float32
    assert np.all((out['priority'] >= 0) & (out['priority'] < 1.0))

def test_prioritize_elders_alg(model_50):
    m = model_50
    alg = PrioritizeElders(base_priority=1.0, elder_boost=5.0, elder_cost=0.2)
    # choose some nodes and ensure at least one elder present
    # pick node indices with ages >=65
    elders = np.where(m.ages >= 65)[0]
    if elders.size == 0:
        pytest.skip("No elders in synthetic population; adjust fixture")
    sample_nodes = list(elders[:2]) + [i for i in range(50) if i not in elders][:3]
    snap = make_snapshot(m, sample_nodes)
    out = alg.generate_candidates(snap, m, np.array([0]))
    nodes = out['nodes']
    prios = out['priority']
    # ensure elders have higher priority
    for e in elders[:2]:
        idx = np.where(nodes == e)[0]
        if idx.size:
            assert prios[idx[0]] > 1.0

def test_call_action_apply_revert(model_50):
    m = model_50
    m._initialize_states()
    # choose a node and ct
    node = 1
    ct = m.contact_types[0]
    action = CallIndividualsAction(nodes=np.array([node], dtype=np.int32), contact_types=[ct], reduction=0.5, duration=1, call_cost=0.05)
    tokens = action.apply(m, current_time=1)
    assert isinstance(tokens, list) and len(tokens) >= 1
    token = tokens[0]
    assert token.action_id == action.id
    assert m.out_multiplier[ct][node] < 1.0
    action.revert_token(m, token)
    assert np.isclose(m.out_multiplier[ct][node], 1.0, atol=1e-6)