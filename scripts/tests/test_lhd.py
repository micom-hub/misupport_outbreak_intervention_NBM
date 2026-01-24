#Tests related to LHD methods and instantiation

import numpy as np
import pandas as pd
import pytest
import warnings

from scripts.network_model import (
    NetworkModel, DefaultModelParams, ExposureEventRecorder,
    LocalHealthDepartment, CallIndividualsAction, ActionBase, ActionToken
)

@pytest.fixture
def small_model(tmp_path):
    # small sample (reuse same fixture pattern)
    data = [
        {"age": 60, "sex": "F", "race": "Latino", "hh_id": "C", "wp_id": None, "sch_id": None, "gq_id": None, "gq": False},
        {"age": 34, "sex": "M", "race": "White", "hh_id": "A", "wp_id": "W1", "sch_id": None, "gq_id": None, "gq": False},
        {"age": 29, "sex": "F", "race": "Black", "hh_id": "A", "wp_id": "W2", "sch_id": None, "gq_id": None, "gq": False},
        {"age": 17, "sex": "F", "race": "Asian", "hh_id": "B", "wp_id": "W2", "sch_id": "S1", "gq_id": None, "gq": False},
        {"age": 15, "sex": "M", "race": "White", "hh_id": "B", "wp_id": None, "sch_id": "S1", "gq_id": None, "gq": False},
    ]
    contacts_df = pd.DataFrame(data)
    params = DefaultModelParams.copy()
    params.update({
        "I0": [1],
        "simulation_duration": 3,
        "seed": 123,
        "overwrite_edge_list": True,
        "save_data_files": False,
        "run_name": "test_lhd",
        "record_exposure_events": True,
        "n_runs": 1,
        "base_transmission_prob": 1.0
    })
    m = NetworkModel(contacts_df=contacts_df, params=params)
    m.results_folder = str(tmp_path)
    return m

def test_discover_and_gather(small_model):
    m = small_model
    lhd = m.lhd
    # force discovery of all events
    lhd.discovery_prob = 1.0
    # create a simple snapshot exposing nodes 2 and 3 in one event
    nodes = np.array([2,3], dtype=np.int32)
    snap = {
        'event_time': np.array([0], dtype=np.int32),
        'event_source': np.array([1], dtype=np.int32),
        'event_type': np.array([0], dtype=np.int16),
        'event_nodes_start': np.array([0], dtype=np.int64),
        'event_nodes_len': np.array([len(nodes)], dtype=np.int32),
        'nodes': nodes,
        'infections': np.zeros(len(nodes), dtype=bool)
    }
    discovered = lhd.discover_exposures(snap)
    assert discovered.size == 1
    # gather candidates (algorithms registered by default)
    action_types, nodes_arr, prios, cts, costs, params = lhd.gather_candidates(snap, discovered)
    assert nodes_arr.size == 2

def test_schedule_and_expire(small_model):
    m = small_model
    lhd = m.lhd
    # pick a node and contact type
    node = 2
    ct = m.contact_types[0]
    # create a CallIndividualsAction with duration 1 day
    action = CallIndividualsAction(nodes=np.array([node], dtype=np.int32),
                                   contact_types=[ct],
                                   reduction=0.8, duration=1, call_cost=0.1)
    # initial multiplier should be 1.0
    assert np.isclose(m.out_multiplier[ct][node], 1.0)
    # schedule the action at day 1
    lhd.schedule_action(action, current_time=1, resource_cost=0.1)
    # after scheduling, multiplier should be less than 1
    assert m.out_multiplier[ct][node] < 1.0
    # action registered
    assert action.id in lhd._active_actions
    # process expiration at day 2
    lhd.process_expirations(current_time=2)
    # multiplier restored
    assert np.isclose(m.out_multiplier[ct][node], 1.0, atol=1e-6)
    # action cleanup
    assert action.id not in lhd._active_actions

def test_nonreversible_action(small_model):
    m = small_model
    lhd = m.lhd

    class VaccinateTestAction(ActionBase):
        def __init__(self, nodes):
            super().__init__(action_type="vaccinate", duration=0, kind="vaccinate")
            self.nodes = np.asarray(nodes, dtype=np.int32)
            self.reversible = False

        def apply(self, model, current_time):
            prev = model.is_vaccinated[self.nodes].copy()
            model.is_vaccinated[self.nodes] = True
            token = ActionToken(action_id=self.id, action_type=self.action_type, contact_type=None,
                                nodes=self.nodes.copy(), factor=None, reversible=False, meta={'stored_prev': prev})
            return [token]

        def revert_token(self, model, token):
            raise NotImplementedError("Not reversible")

    nodes = np.array([0], dtype=np.int32)
    act = VaccinateTestAction(nodes)
    # apply via schedule_action
    lhd.schedule_action(act, current_time=1, resource_cost=0.05)
    assert m.is_vaccinated[nodes].all()
    assert act.id not in lhd._active_actions