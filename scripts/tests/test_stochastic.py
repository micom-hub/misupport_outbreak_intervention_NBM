import copy
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

# Project imports (adjust module paths if your package layout differs)
from scripts.variants.run_variants_funcs import run_variants
from scripts.lhd.lhdConfig import LhdVariant, LhdConfig
from scripts.simulation.outbreak_model import NetworkModel
from scripts.graph.graph_utils import GraphData


# --- Helpers ---------------------------------------------------------------

def canonicalize(obj: Any):
    """Recursively convert numpy/scalar types to Python built-ins for stable comparison & JSON output."""
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return canonicalize(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [canonicalize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted(canonicalize(x) for x in obj)
    # bit generator state dicts can contain numpy arrays; canonicalize nested
    try:
        # attempt JSON-serializable fallback
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def arrays_equal(a, b) -> bool:
    """Robust equality for arrays/lists/None."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    try:
        return bool(np.array_equal(a_np, b_np))
    except Exception:
        # fallback to canonicalized equality
        return canonicalize(a) == canonicalize(b)


def dicts_equal(d0, d1) -> bool:
    return canonicalize(d0) == canonicalize(d1)


# --- Minimal Dummy Config & GraphData -------------------------------------

class DummyModelConfig:
    """Minimal config object providing the attributes your model expects."""
    def __init__(self, seed=202600, n_replicates=2, sim_duration=4, I0=1, vax_uptake=0.2):
        self.sim = type("S", (), {})()
        self.sim.seed = int(seed)
        self.sim.n_replicates = int(n_replicates)
        self.sim.simulation_duration = int(sim_duration)
        self.sim.I0 = I0
        self.sim.record_exposure_events = False
        self.sim.save_data_files = False
        self.sim.display_plots = False
        self.sim.county = "X"
        self.sim.state = "Y"

        self.epi = type("E", (), {})()
        self.epi.base_transmission_prob = 0.0  # keep epidemic simple for debugging
        self.epi.vax_efficacy = 0.0
        self.epi.susceptibility_multiplier_under_five = 1.0
        self.epi.susceptibility_multiplier_elderly = 1.0
        self.epi.relative_infectiousness_vax = 1.0
        self.epi.gamma_alpha = 2.0
        self.epi.incubation_period = 1.0
        self.epi.infectious_period = 1.0
        self.epi.incubation_period_vax = 1.0
        self.epi.infectious_period_vax = 1.0
        self.epi.vax_uptake = float(vax_uptake)
        self.epi.conferred_immunity_duration = None
        self.epi.lasting_partial_immunity = None

        self.lhd = type("L", (), {})()
        self.lhd.lhd_discovery_prob = 0.0
        self.lhd.lhd_workday_hrs = 8.0
        self.lhd.mean_compliance = 1.0
        self.lhd.lhd_default_int_reduction = 0.0
        self.lhd.lhd_default_int_duration = 0
        self.lhd.lhd_default_call_duration = 0.1

    def validate(self):
        return True

    def to_json(self, path: str):
        # minimal JSON for compatibility with code that writes ModelConfig.json
        d = {
            "sim": {k: getattr(self.sim, k) for k in vars(self.sim)},
            "epi": {k: getattr(self.epi, k) for k in vars(self.epi)},
            "lhd": {k: getattr(self.lhd, k) for k in vars(self.lhd)},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(d, fh)

    def copy_with(self, updates: dict):
        # implement only the shape used by run_variants: {"sim": {"seed": ...}}
        seed = self.sim.seed
        sim_updates = updates.get("sim", {}) if isinstance(updates, dict) else {}
        if "seed" in sim_updates:
            seed = int(sim_updates["seed"])
        return DummyModelConfig(seed=seed, n_replicates=self.sim.n_replicates, sim_duration=self.sim.simulation_duration, I0=self.sim.I0)


def make_minimal_graphdata(N: int = 8) -> GraphData:
    # minimal edge_list no edges (so transmission is trivial), and simple metadata
    edge_list = pd.DataFrame(columns=["source", "target", "weight", "contact_type"])
    adj = csr_matrix((N, N))
    individual_lookup = pd.DataFrame({"age": np.arange(N), "sex": np.array(["M"] * N)})
    ages = individual_lookup["age"].to_numpy()
    sexes = individual_lookup["sex"].to_numpy()
    compliances = np.ones(N, dtype=np.float32)
    neighbor_map = {i: [] for i in range(N)}
    fast_neighbor_map = {i: {} for i in range(N)}
    csr_by_type = {}  # empty contact types
    contact_types = []
    ct_to_id = {}
    id_to_ct = {}
    full_node_list = list(range(N))
    degrees_arr = np.zeros(N, dtype=np.int32)

    return GraphData(
        N=int(N),
        edge_list=edge_list,
        adj_matrix=adj,
        individual_lookup=individual_lookup,
        ages=ages,
        sexes=sexes,
        compliances=compliances,
        neighbor_map=neighbor_map,
        fast_neighbor_map=fast_neighbor_map,
        csr_by_type=csr_by_type,
        contact_types=contact_types,
        ct_to_id=ct_to_id,
        id_to_ct=id_to_ct,
        full_node_list=full_node_list,
        degrees_arr=degrees_arr,
    )


# --- Instrumentation wrappers ---------------------------------------------

def instrument_networkmodel(monkeypatch, records: Dict[int, Dict[str, list]]):
    """
    Monkeypatch NetworkModel._initialize_states and NetworkModel.step to record snapshots.
    records is a dict keyed by id(model) with values {'inits': [...], 'steps': [...]}
    """

    orig_init = NetworkModel._initialize_states
    orig_step = NetworkModel.step

    def wrap_init(orig):
        def wrapped(self, *args, **kwargs):
            res = orig(self, *args, **kwargs)
            mid = id(self)
            recs = records.setdefault(mid, {"inits": [], "steps": []})
            rep_idx = getattr(self, "replicate_ind", None)
            rep_seed = getattr(self, "replicate_seed", None)
            # record RNG bit generator state if available
            rng_state = None
            try:
                if getattr(self, "rng", None) is not None:
                    rng_state = copy.deepcopy(self.rng.bit_generator.state)
            except Exception:
                rng_state = repr(getattr(self, "rng", None))
            is_vax = None
            try:
                if getattr(self, "is_vaccinated", None) is not None:
                    is_vax = np.asarray(self.is_vaccinated).tolist()
            except Exception:
                is_vax = None
            # capture initial states_over_time[0] if present
            init_state = None
            try:
                if getattr(self, "states_over_time", None):
                    first = self.states_over_time[0]
                    init_state = [sorted(x) if isinstance(x, list) else x for x in first]
            except Exception:
                init_state = None

            recs["inits"].append({
                "replicate_ind": rep_idx,
                "replicate_seed": rep_seed,
                "rng_state": canonicalize(rng_state),
                "is_vaccinated": canonicalize(is_vax),
                "I0": canonicalize(getattr(self, "I0", None)),
                "initial_state": canonicalize(init_state),
            })
            return res
        return wrapped

    def wrap_step(orig):
        def wrapped(self, recorder=None):
            mid = id(self)
            recs = records.setdefault(mid, {"inits": [], "steps": []})
            pre_state = None
            try:
                pre_state = np.copy(self.state) if getattr(self, "state", None) is not None else None
            except Exception:
                pre_state = None
            pre = {
                "replicate_ind": getattr(self, "replicate_ind", None),
                "time": getattr(self, "current_time", None),
                "state_before": canonicalize(pre_state),
                "rng_before": canonicalize(self.rng.bit_generator.state if getattr(self, "rng", None) is not None else None),
                "lhd_action_log_before": canonicalize(list(self.lhd.action_log) if getattr(self, "lhd", None) is not None else None),
            }
            recs["steps"].append({"pre": pre, "post": None})
            # call original step
            out = orig(self, recorder)
            # capture post state: latest new_exposures and lhd action log snapshot
            try:
                post_state = np.copy(self.state) if getattr(self, "state", None) is not None else None
            except Exception:
                post_state = None
            post = {
                "replicate_ind": getattr(self, "replicate_ind", None),
                "time": getattr(self, "current_time", None),
                "state_after": canonicalize(post_state),
                "new_exposures": canonicalize(self.new_exposures[-1] if getattr(self, "new_exposures", None) else None),
                "new_infections": canonicalize(self.new_infections[-1] if getattr(self, "new_infections", None) else None),
                "rng_after": canonicalize(self.rng.bit_generator.state if getattr(self, "rng", None) is not None else None),
                "lhd_action_log_after": canonicalize(list(self.lhd.action_log) if getattr(self, "lhd", None) is not None else None),
            }
            recs["steps"][-1]["post"] = post
            return out
        return wrapped

    # apply monkeypatches
    monkeypatch.setattr(NetworkModel, "_initialize_states", wrap_init(orig_init), raising=True)
    monkeypatch.setattr(NetworkModel, "step", wrap_step(orig_step), raising=True)


# --- Comparison logic -----------------------------------------------------

def first_mismatch_between_models(m0, m1, records) -> Tuple[bool, str]:
    """
    Compare records for two models and return (match_bool, message).
    If match_bool is False then message describes the first mismatch.
    """
    mid0 = id(m0)
    mid1 = id(m1)
    rec0 = records.get(mid0, {"inits": [], "steps": []})
    rec1 = records.get(mid1, {"inits": [], "steps": []})

    # 1) Compare initialization records per replicate
    n_inits = max(len(rec0["inits"]), len(rec1["inits"]))
    for r in range(n_inits):
        if r >= len(rec0["inits"]) or r >= len(rec1["inits"]):
            return False, f"Different number of _initialize_states calls: model0 has {len(rec0['inits'])}, model1 has {len(rec1['inits'])}"
        a = rec0["inits"][r]
        b = rec1["inits"][r]
        # compare replicate_seed (if present)
        if a.get("replicate_seed") != b.get("replicate_seed"):
            return False, f"Mismatch at init replicate {r}: replicate_seed differs: {a.get('replicate_seed')} vs {b.get('replicate_seed')}"
        # compare vaccination arrays
        if not arrays_equal(a.get("is_vaccinated"), b.get("is_vaccinated")):
            return False, f"Mismatch at init replicate {r}: is_vaccinated differs.\nmodel0 sample: {a.get('is_vaccinated')[:10] if a.get('is_vaccinated') else a.get('is_vaccinated')}\nmodel1 sample: {b.get('is_vaccinated')[:10] if b.get('is_vaccinated') else b.get('is_vaccinated')}"
        # compare I0
        if canonicalize(a.get("I0")) != canonicalize(b.get("I0")):
            return False, f"Mismatch at init replicate {r}: I0 differs: {a.get('I0')} vs {b.get('I0')}"
        # initial state list
        if canonicalize(a.get("initial_state")) != canonicalize(b.get("initial_state")):
            return False, f"Mismatch at init replicate {r}: initial states_over_time[0] differs: {a.get('initial_state')} vs {b.get('initial_state')}"

    # 2) Compare step-by-step by (replicate, time)
    def build_step_map(step_list):
        m = {}
        for entry in step_list:
            pre = entry.get("pre", {})
            key = (pre.get("replicate_ind"), pre.get("time"))
            # keep first entry for key if duplicates exist (should not)
            if key not in m:
                m[key] = entry
        return m

    map0 = build_step_map(rec0["steps"])
    map1 = build_step_map(rec1["steps"])
    all_keys = sorted(set(map0.keys()) | set(map1.keys()), key=lambda k: (k[0] if k[0] is not None else -1, k[1] if k[1] is not None else -1))
    for key in all_keys:
        if key not in map0:
            return False, f"Step missing in model0 for (replicate,time) = {key}; model1 has it"
        if key not in map1:
            return False, f"Step missing in model1 for (replicate,time) = {key}; model0 has it"
        ent0 = map0[key]
        ent1 = map1[key]
        pre0 = ent0["pre"]
        pre1 = ent1["pre"]
        post0 = ent0["post"]
        post1 = ent1["post"]
        # compare pre state
        if not arrays_equal(pre0.get("state_before"), pre1.get("state_before")):
            return False, f"Mismatch PRE at replicate={key[0]} time={key[1]}: state_before differs.\nmodel0(before)={pre0.get('state_before')}\nmodel1(before)={pre1.get('state_before')}"
        # compare post state
        if not arrays_equal(post0.get("state_after"), post1.get("state_after")):
            return False, f"Mismatch POST at replicate={key[0]} time={key[1]}: state_after differs.\nmodel0(after)={post0.get('state_after')}\nmodel1(after)={post1.get('state_after')}"
        # compare new_exposures
        if not arrays_equal(post0.get("new_exposures"), post1.get("new_exposures")):
            return False, f"Mismatch at replicate={key[0]} time={key[1]}: new_exposures differs.\nmodel0={post0.get('new_exposures')}\nmodel1={post1.get('new_exposures')}"
        # compare LHD action logs
        if not dicts_equal(post0.get("lhd_action_log_after"), post1.get("lhd_action_log_after")):
            return False, f"Mismatch at replicate={key[0]} time={key[1]}: lhd_action_log differs.\nmodel0={post0.get('lhd_action_log_after')}\nmodel1={post1.get('lhd_action_log_after')}"
    # 3) compare aggregated outputs if present
    df0 = m0.results_to_df() if hasattr(m0, "results_to_df") else None
    df1 = m1.results_to_df() if hasattr(m1, "results_to_df") else None
    if df0 is not None and df1 is not None:
        try:
            pd.testing.assert_frame_equal(df0.reset_index(drop=True), df1.reset_index(drop=True))
        except AssertionError as e:
            return False, f"Aggregate summary DataFrame differs: {e}"
    return True, "No mismatches found"


# --- The test -------------------------------------------------------------
def test_two_identical_variants_stepwise_equality(monkeypatch, tmp_path: Path):
    """
    Run two identical variants in one run (via run_variants) and assert all per-replicate
    outputs (initialization and each step) are identical. If not, the test reports
    the first location where they differ.
    """
    records: Dict[int, Dict[str, list]] = {}

    # instrument NetworkModel lifecycle
    instrument_networkmodel(monkeypatch, records)

    # build test inputs
    seed = 202600
    n_reps = 2
    sim_duration = 4
    cfg = DummyModelConfig(seed=seed, n_replicates=n_reps, sim_duration=sim_duration, I0=1, vax_uptake=0.35)
    gd = make_minimal_graphdata(N=12)

    # create two identical variants (empty maps -> LocalHealthDepartment.register_defaults will populate defaults)
    v1 = LhdVariant(name="identical_1", algorithm_map={}, action_factory_map={}, description="copy1")
    v2 = LhdVariant(name="identical_2", algorithm_map={}, action_factory_map={}, description="copy2")
    lhd_config = LhdConfig(variants=[v1, v2])

    outdir = tmp_path / "runs"
    outdir.mkdir(parents=True, exist_ok=True)

    # run variants sequentially via run_variants (it returns list of NetworkModel objects)
    models = run_variants(
        lhd_config=lhd_config,
        cfg=cfg,
        graphdata=gd,
        output_dir=str(outdir),
        i=0,
        seed=seed,
        register_defaults=True,
        save_summary=False,
        save_incidence=False,
        save_prevalence=False,
        summary_metrics=None,
        overwrite=True
    )

    assert isinstance(models, list) and len(models) == 2, "Expected run_variants to return two models (one per variant)"

    m0, m1 = models[0], models[1]

    # Persist canonicalized records for offline debugging if needed
    try:
        dump_path = tmp_path / "instrumentation_records.json"
        with open(dump_path, "w") as fh:
            json.dump({
                str(k): canonicalize(v) for k, v in records.items()
            }, fh, indent=2)
    except Exception:
        pass

    ok, msg = first_mismatch_between_models(m0, m1, records)
    if not ok:
        # provide the records file location for debugging
        raise AssertionError(f"Variant runs diverged: {msg}\nInstrumentation saved to: {str(tmp_path)}")
    # final equality assertion on aggregated outputs too
    df0 = m0.results_to_df()
    df1 = m1.results_to_df()
    pd.testing.assert_frame_equal(df0.reset_index(drop=True), df1.reset_index(drop=True))