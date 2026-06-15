import pytest
import numpy as np
from unittest.mock import MagicMock
from scripts.config import ModelConfig
from scripts.lhd.surveillance import SurveillanceModel, STAGE_INF
from scripts.lhd.state import LHDState
from scripts.lhd.planner import GreedyPlanner
from scripts.lhd.executor import Executor
from scripts.lhd.response_types import ActionProposal
from scripts.lhd.lhd import LocalHealthDepartment


@pytest.fixture
def base_config():
    return ModelConfig()


@pytest.fixture
def mock_model(base_config):
    model = MagicMock()
    model.config = base_config
    model.N = 100
    model.neighbor_map = {i: [] for i in range(100)}
    model.ct_to_id = {"hh": 0, "wp": 1, "sch": 2, "cas": 3}
    model.ages = np.random.randint(0, 80, 100)
    model.is_vaccinated = np.zeros(100, dtype=bool)
    model.in_multiplier = {ct: np.ones(100) for ct in model.ct_to_id}
    model.out_multiplier = {ct: np.ones(100) for ct in model.ct_to_id}
    return model


def test_surveillance_baseline_detection(mock_model):
    """Verify that baseline surveillance picks up infectious cases with probability p."""
    surv = SurveillanceModel(
        neighbor_map=mock_model.neighbor_map,
        ct_to_id=mock_model.ct_to_id,
        seed=42,
        N=mock_model.N,
        ages=mock_model.ages,
        is_vax=mock_model.is_vaccinated,
        p_detect_inf=1.0,  # Force detection
    )

    # Simulate new infections transitioning to I
    epi_state = {"new_inf_ids": np.array([1, 2, 3], dtype=np.int32)}
    batch = surv.step(t=1, epi_state=epi_state)

    # With p=1.0 and delay=0, they should show up in reported_cases immediately
    assert len(batch["reported_cases"]) == 3
    assert set(batch["reported_cases"]) == {1, 2, 3}


def test_lhd_state_integration():
    """Verify LHDState correctly integrates batches from surveillance."""
    state = LHDState(N=10)
    batch = {
        "t": 5,
        "reported_cases": np.array([0, 1]),
        "reported_stage": np.array([STAGE_INF, STAGE_INF]),
        "trace_src": np.array([0]),
        "trace_tgt": np.array([5]),
        "trace_ct": np.array([0]),
    }
    state.process_batch(batch)

    assert state.known_case[0] == True
    assert 5 in [nbr[0] for nbr in state.neighbors(0)]
    assert state.new_cases_today.size == 2


def test_greedy_planner_capacity():
    """Verify the planner respects daily capacity and prioritizes correctly."""
    planner = GreedyPlanner()
    proposals = [
        ActionProposal("isolate", "node", 1, priority=1.0, cost_units=1),
        ActionProposal("isolate", "node", 2, priority=2.0, cost_units=1),
        ActionProposal("trace", "node", 3, priority=0.5, cost_units=1),
    ]

    # Capacity only for 2 actions
    plan = planner.select(proposals, capacity=2)

    assert len(plan.selected) == 2
    # Node 2 (higher priority) and Node 1 should be selected
    selected_targets = [p.target for p in plan.selected]
    assert 2 in selected_targets
    assert 1 in selected_targets
    assert 3 not in selected_targets


def test_executor_delegation(mock_model):
    """Verify the executor correctly dispatches calls to LHD logic."""
    lhd = LocalHealthDepartment(seed=1, surv_seed=2, model=mock_model)
    lhd._apply_isolation = MagicMock(return_value=(1, 1))

    executor = Executor()
    proposal = ActionProposal("isolate", "node", 10, priority=1.0, cost_units=1)
    plan = MagicMock()
    plan.selected = [proposal]

    executor.execute(lhd=lhd, t=1, plan=plan)

    # Ensure lhd method was called with the correct node
    lhd._apply_isolation.assert_called_once()
    args, kwargs = lhd._apply_isolation.call_args
    assert 10 in kwargs["nodes"]


def test_lhd_step_integration(mock_model):
    """Integration test for a full LHD cycle."""
    # Configure LHD with a simple policy
    mock_model.config = mock_model.config.copy_with(
        {
            "lhd": {
                "policy_name": "isolate_only",
                "lhd_daily_capacity": 10,
                "p_detect_inf": 1.0,
                "report_delay_days": 0,
            }
        }
    )

    lhd = LocalHealthDepartment(seed=1, surv_seed=2, model=mock_model)

    # Day 1: 5 new infections reported
    epi_state = {"new_inf_ids": np.array([1, 2, 3, 4, 5], dtype=np.int32)}
    batch = lhd.step(t=1, epi_state=epi_state)

    # Check if actions were taken (Isolation should have been applied)
    results = lhd.results_to_df()
    assert results.iloc[-1]["applied_isolate"] == 5
    assert results.iloc[-1]["reported_cases_today"] == 5
