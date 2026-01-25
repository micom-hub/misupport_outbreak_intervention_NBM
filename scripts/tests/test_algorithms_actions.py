# tests/test_algorithms_actions.py
import numpy as np
import pandas as pd
import pytest

from scripts.network_model import (
    NetworkModel, DefaultModelParams, CallIndividualsAction,
    EqualPriority, RandomPriority, PrioritizeElders
)

@pytest.fixture
def contacts_50():
    rng = np.random.RandomState(123)
    N = 50
    ages = rng.choice(range(1, 90), size=N)
    sexes = rng.choice(['F', 'M'], size=N)
    races = rng.choice(['White', 'Black', 'Asian', 'Latino'], size=N)

    # household grouping: groups sizes 2-4 randomly
    hh = []
    cur = 0
    hh_id = 0
    while cur < N:
        size = rng.choice([2, 3, 4])
        for _ in range(min(size, N - cur)):
            hh.append(f"HH{hh_id}")          # string household id (FIXED)
            cur += 1
        hh_id += 1

    # workplaces: assign 10 workplaces (or None)
    wp_ids = [f"W{rng.randint(11,20)}" if rng.rand() < 0.7 else np.nan for _ in range(N)]

    # schools for younger ages
    sch_ids = [f"S{rng.randint(30,35)}" if age < 18 else np.nan for age in ages]

    rows = []
    for i in range(N):
        rows.append({
            "age": int(ages[i]),
            "sex": sexes[i],
            "race": races[i],
            "hh_id": hh[i],
            "wp_id": wp_ids[i],
            "sch_id": sch_ids[i],
            "gq_id": np.nan,
            "gq": False
        })
    return pd.DataFrame(rows)


@pytest.fixture
def model_50(contacts_50, tmp_path):
    params = DefaultModelParams.copy()
    params.update({
        "I0": [0, 1, 2],
        "simulation_duration": 2,
        "seed": 2023,
        "overwrite_edge_list": True,
        "save_data_files": False,
        "run_name": "test_alg_action",
        "record_exposure_events": True,
        "n_runs": 1,
        "base_transmission_prob": 0.5,
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

    expected_nodes = np.array([3, 10, 20, 30], dtype=np.int32)
    snap = make_snapshot(m, expected_nodes)
    out = alg.generate_candidates(snap, m, np.array([0]))

    actual_nodes = np.asarray(out['nodes'], dtype=np.int32)

    print("\nEqualPriority expected nodes:", expected_nodes)
    print("EqualPriority actual nodes:", actual_nodes)
    print("EqualPriority priorities:", out['priority'])

    # The algorithm should return exactly the exposed nodes (set equality)
    assert set(actual_nodes.tolist()) == set(expected_nodes.tolist())
    assert np.all(out['priority'] == 1.0)


def test_randompriority_alg(model_50):
    m = model_50
    alg = RandomPriority()

    expected_nodes = np.array([5, 6, 7], dtype=np.int32)
    snap = make_snapshot(m, expected_nodes)
    out = alg.generate_candidates(snap, m, np.array([0]))

    actual_nodes = np.asarray(out['nodes'], dtype=np.int32)
    priorities = out['priority']

    print("\nRandomPriority expected nodes:", expected_nodes)
    print("RandomPriority actual nodes:", actual_nodes)
    print("RandomPriority priorities:", priorities)

    # Node set should be the same as the snapshot
    assert set(actual_nodes.tolist()) == set(expected_nodes.tolist())
    # priorities numeric and in [0,1)
    assert priorities.dtype == np.float32
    assert np.all((priorities >= 0) & (priorities < 1.0))


def test_prioritize_elders_alg(model_50):
    m = model_50
    alg = PrioritizeElders(base_priority=1.0, elder_boost=5.0, elder_cost=0.2)

    # find at least two elders in the synthetic population
    elders = np.where(m.ages >= 65)[0]
    if elders.size == 0:
        pytest.skip("No elders in synthetic population; adjust fixture")

    # choose sample: two elders + three other nodes
    non_elders = [i for i in range(50) if i not in elders]
    sample_nodes = list(elders[:2]) + non_elders[:3]
    expected_nodes = np.array(sample_nodes, dtype=np.int32)

    snap = make_snapshot(m, expected_nodes)
    out = alg.generate_candidates(snap, m, np.array([0]))

    nodes = np.asarray(out['nodes'], dtype=np.int32)
    prios = out['priority']
    costs = out['costs']

    print("\nPrioritizeElders expected nodes:", expected_nodes)
    print("PrioritizeElders actual nodes:", nodes)
    print("PrioritizeElders priorities (node:priority):")
    for n, p, c in zip(nodes, prios, costs):
        print(f"  node {n}: priority={p}, cost={c}, age={m.ages[n]}")

    # Ensure node set matches
    assert set(nodes.tolist()) == set(expected_nodes.tolist())
    # Check elders in our sample get boosted priority
    for elder in elders[:2]:
        idxs = np.where(nodes == elder)[0]
        if idxs.size:
            assert prios[idxs[0]] > 1.0  # elder priority > base


def test_call_action_apply_revert(model_50):
    m = model_50
    m._initialize_states()
    # choose a node and contact type
    node = 1
    ct = m.contact_types[0]

    print("\nBefore CallIndividualsAction:")
    print("  out_multiplier[ct][node] =", m.out_multiplier[ct][node])
    print("  in_multiplier[ct][node]  =", m.in_multiplier[ct][node])

    action = CallIndividualsAction(
        nodes=np.array([node], dtype=np.int32),
        contact_types=[ct],
        reduction=0.5,
        duration=1,
        call_cost=0.05
    )

    tokens = action.apply(m, current_time=1)
    assert isinstance(tokens, list) and len(tokens) >= 1
    token = tokens[0]
    assert token.action_id == action.id

    print("\nAfter apply():")
    print("  out_multiplier[ct][node] =", m.out_multiplier[ct][node])
    print("  in_multiplier[ct][node]  =", m.in_multiplier[ct][node])
    print("  token.factor =", token.factor)

    # Assert that multipliers decreased
    assert m.out_multiplier[ct][node] < 1.0
    assert m.in_multiplier[ct][node] < 1.0

    # Revert and check restoration
    action.revert_token(m, token)
    print("\nAfter revert_token():")
    print("  out_multiplier[ct][node] =", m.out_multiplier[ct][node])
    print("  in_multiplier[ct][node]  =", m.in_multiplier[ct][node])

    assert np.isclose(m.out_multiplier[ct][node], 1.0, atol=1e-6)
    assert np.isclose(m.in_multiplier[ct][node], 1.0, atol=1e-6)