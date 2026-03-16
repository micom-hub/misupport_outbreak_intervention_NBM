#!/usr/bin/env python3
"""
scripts/diagnostics.py

This is an GPT-generated code script document that returns a bunch of information comparing two runs.

Calling it from CLI produces useful outputs at tmp/diagnostics/diagnostics for examining exactly where two runs diverge, for verifying stochasticity. 

Call:
python -m scripts.diagnostics --csv testLHS.csv --n_samples 1 --seed 0 --outdir tmp/diagnostics --model_index 0



"""
from __future__ import annotations
import argparse
import copy
import json
import hashlib
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Ensure repo root on sys.path for relative imports if needed
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Project imports (assumes this file lives in scripts/)
from scripts.variants.run_variants_funcs import prepare_run, sample_from_master_graphdata, build_graph_data
from scripts.variants.run_variants_funcs import validate_variant  # re-exported in that module
from scripts.utils.rng_utility import derive_seed_from_base
from scripts.lhd.lhdConfig import LhdConfig
from scripts.simulation.outbreak_model import NetworkModel
from scripts.graph.graph_utils import GraphData
from scripts.config import ModelConfig

# Helper utilities ---------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def hash_array(arr: Any) -> str:
    a = np.asarray([]) if arr is None else np.asarray(arr)
    try:
        # include dtype and shape for safety
        payload = a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()
    except Exception:
        payload = repr(a).encode()
    return sha256_bytes(payload)

def hash_dataframe(df: Optional[pd.DataFrame]) -> str:
    if df is None:
        return sha256_bytes(b"__none__")
    try:
        # stable ordering: sort by columns then rows (best-effort)
        cols = list(df.columns)
        if cols:
            df2 = df.sort_values(by=cols).reset_index(drop=True)
            b = df2.to_csv(index=False).encode()
        else:
            b = df.to_csv(index=False).encode()
    except Exception:
        b = repr(df).encode()
    return sha256_bytes(b)

def fingerprint_graphdata(gd: GraphData) -> Dict[str, Any]:
    """
    Produce a small fingerprint dictionary for GraphData to detect mutation.
    """
    try:
        el_h = hash_dataframe(gd.edge_list)
    except Exception:
        el_h = sha256_bytes(repr(getattr(gd, "edge_list", None)).encode())
    # csr_by_type hashes
    csr_dict = {}
    try:
        for ct, triple in getattr(gd, "csr_by_type", {}).items():
            indptr, indices, weights = triple
            h = hashlib.sha256()
            try:
                h.update(np.asarray(indptr).astype(np.int64).tobytes())
                h.update(np.asarray(indices).astype(np.int32).tobytes())
                h.update(np.asarray(weights).astype(np.float32).tobytes())
            except Exception:
                h.update(repr((indptr, indices, weights)).encode())
            csr_dict[str(ct)] = h.hexdigest()
    except Exception:
        csr_dict = {"error": "csr fingerprint failed"}
    # neighbor_map fingerprint (structural)
    try:
        nm_items = []
        for src in sorted(getattr(gd, "neighbor_map", {}).keys()):
            nbrs = gd.neighbor_map.get(src, [])
            sorted_nbrs = sorted([(int(t), float(w), str(ct)) for (t, w, ct) in nbrs])
            nm_items.append((int(src), tuple(sorted_nbrs)))
        nm_b = repr(nm_items).encode()
        nm_h = sha256_bytes(nm_b)
    except Exception:
        nm_h = sha256_bytes(repr(getattr(gd, "neighbor_map", None)).encode())
    return {"edge_list_hash": el_h, "csr_by_type": csr_dict, "neighbor_map_hash": nm_h}

def canonicalize_rng_state(state: Any) -> Any:
    """
    Convert numpy arrays in RNG state to python lists for stable JSON output and comparison.
    """
    if state is None:
        return None
    if isinstance(state, dict):
        out = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (list, tuple)):
                # recursively canonicalize dict elements inside lists if any
                out[k] = [canonicalize_rng_state(x) if isinstance(x, dict) else (x.tolist() if isinstance(x, np.ndarray) else x) for x in v]
            elif isinstance(v, dict):
                out[k] = canonicalize_rng_state(v)
            else:
                out[k] = v
        return out
    return repr(state)

def normalize_state_lists(states):
    """
    Normalize a timestep [S,E,I,R] into lists where each inner list is sorted (so ordering differences don't break equality).
    """
    if states is None:
        return None
    try:
        return [sorted(list(x)) if isinstance(x, (list, tuple, np.ndarray)) else x for x in states]
    except Exception:
        # fallback: attempt to coerce to lists and sort where possible
        try:
            return [sorted(list(x)) for x in states]
        except Exception:
            return states

def arrays_equal_sorted(a: Any, b: Any) -> bool:
    """Compare two arrays/lists treating them as unordered sets (sort each before compare)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    # If 1D integer-like arrays (node lists), compare sorted
    if a_arr.ndim == 1 and b_arr.ndim == 1:
        a_sorted = np.sort(a_arr)
        b_sorted = np.sort(b_arr)
        return np.array_equal(a_sorted, b_sorted)
    # otherwise exact array equality
    try:
        return np.array_equal(a_arr, b_arr)
    except Exception:
        return canonicalize(a) == canonicalize(b)

def canonicalize(x: Any) -> Any:
    """JSON-serializable canonicalization of common types used in compares."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [canonicalize(v) for v in x]
    if isinstance(x, dict):
        return {str(k): canonicalize(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return canonicalize(x.tolist())
    try:
        json.dumps(x)
        return x
    except Exception:
        return repr(x)

# Emulation & recording ---------------------------------------------------

def emulate_simulation_and_record(model: NetworkModel, results_folder: Path, variant_label: str, max_list_snapshot: int = 200) -> Dict[str, Any]:
    """
    Emulate the per-replicate simulation loop for a model object but capture
    detailed snapshots at initialization and each timestep (pre/post, LHD).
    Returns a nested dictionary with all captured records.
    """
    rec = {"variant": variant_label, "base_seed": int(getattr(model, "_base_seed", -1)), "replicates": []}
    run_dir = Path(results_folder)
    run_dir.mkdir(parents=True, exist_ok=True)

    n_reps = int(model.n_replicates)
    Tmax = int(model.Tmax)

    for run in range(n_reps):
        rep_rec = {"replicate_index": int(run), "init": None, "steps": []}
        try:
            # call model._initialize_states(run) to set replicate_seed, self.rng, and self.lhd.rng
            model._initialize_states(run)
        except Exception:
            # capture traceback and stop if init fails
            rep_rec["init"] = {"error": "initialize failed", "traceback": traceback.format_exc()}
            rec["replicates"].append(rep_rec)
            continue

        # capture initial RNG states and small samples
        rng_state = None
        lhd_rng_state = None
        try:
            rng_state = canonicalize_rng_state(model.rng.bit_generator.state if getattr(model, "rng", None) is not None else None)
        except Exception:
            rng_state = repr(getattr(model, "rng", None))
        try:
            lhd_rng_state = canonicalize_rng_state(model.lhd.rng.bit_generator.state if getattr(model.lhd, "rng", None) is not None else None)
        except Exception:
            lhd_rng_state = repr(getattr(model.lhd, "rng", None))

        init_snapshot = {
            "replicate_seed": int(getattr(model, "replicate_seed", -1)),
            "rng_state": rng_state,
            "rng_sample_first5": None,
            "lhd_rng_state": lhd_rng_state,
            "lhd_rng_sample_first5": None,
            "is_vaccinated_head": None,
            "I0": canonicalize(getattr(model, "I0", None)),
            "initial_states_over_time_0": canonicalize(normalize_state_lists(model.states_over_time[0]) if getattr(model, "states_over_time", None) else None),
            "numba_legacy_seed_base": int(getattr(model, "_numba_legacy_seed_base", -1))
        }
        try:
            init_snapshot["rng_sample_first5"] = model.rng.random(5).tolist() if getattr(model, "rng", None) is not None else None
        except Exception:
            init_snapshot["rng_sample_first5"] = None
        try:
            init_snapshot["lhd_rng_sample_first5"] = model.lhd.rng.random(5).tolist() if getattr(model.lhd, "rng", None) is not None else None
        except Exception:
            init_snapshot["lhd_rng_sample_first5"] = None
        try:
            iv = getattr(model, "is_vaccinated", None)
            if iv is not None:
                head = iv[:min(len(iv), max_list_snapshot)].tolist() if isinstance(iv, np.ndarray) else list(iv)[:max_list_snapshot]
                init_snapshot["is_vaccinated_head"] = canonicalize(head)
            else:
                init_snapshot["is_vaccinated_head"] = None
        except Exception:
            init_snapshot["is_vaccinated_head"] = repr(getattr(model, "is_vaccinated", None))

        rep_rec["init"] = init_snapshot

        # per-timestep loop
        for t in range(1, Tmax + 1):
            step_entry = {"time": int(t), "pre": None, "post": None, "lhd_after": None}
            try:
                model.current_time = int(t)
                # prepare recorder if the model wants exposure recording
                if model.config.sim.record_exposure_events:
                    recorder = model.recorder_template
                    recorder.reset()
                else:
                    recorder = None

                # PRE snapshot
                pre_snap = {}
                try:
                    pre_snap["state_before"] = canonicalize(model.state.copy() if getattr(model, "state", None) is not None else None)
                except Exception:
                    pre_snap["state_before"] = repr(getattr(model, "state", None))
                try:
                    pre_snap["rng_before"] = canonicalize_rng_state(model.rng.bit_generator.state if getattr(model, "rng", None) is not None else None)
                except Exception:
                    pre_snap["rng_before"] = repr(getattr(model, "rng", None))
                try:
                    pre_snap["lhd_rng_before"] = canonicalize_rng_state(model.lhd.rng.bit_generator.state if getattr(model.lhd, "rng", None) is not None else None)
                except Exception:
                    pre_snap["lhd_rng_before"] = repr(getattr(model.lhd, "rng", None))
                try:
                    pre_snap["lhd_action_log_before_len"] = len(list(model.lhd.action_log)) if getattr(model, "lhd", None) is not None else None
                except Exception:
                    pre_snap["lhd_action_log_before_len"] = repr(getattr(model.lhd, "action_log", None))

                # update multiplier matrices (pre-step)
                try:
                    # a small sample of multipliers for debug
                    mm = {}
                    for ct in model.contact_types[:3]:
                        mm[f"in_{ct}_head"] = (model.in_multiplier[ct][:10].tolist() if model.in_multiplier[ct] is not None else None)
                        mm[f"out_{ct}_head"] = (model.out_multiplier[ct][:10].tolist() if model.out_multiplier[ct] is not None else None)
                    pre_snap["multiplier_sample"] = canonicalize(mm)
                except Exception:
                    pre_snap["multiplier_sample"] = "error"

                step_entry["pre"] = pre_snap

                # call model.step which updates state and internal arrays
                model.step(recorder)

                # POST snapshot (immediately after model.step)
                post_snap = {}
                try:
                    post_snap["state_after"] = canonicalize(model.state.copy() if getattr(model, "state", None) is not None else None)
                except Exception:
                    post_snap["state_after"] = repr(getattr(model, "state", None))
                try:
                    # the most recent new_exposures/new_infections are appended to new_exposures
                    post_snap["new_exposures"] = canonicalize(model.new_exposures[-1] if getattr(model, "new_exposures", None) else None)
                    post_snap["new_infections"] = canonicalize(model.new_infections[-1] if getattr(model, "new_infections", None) else None)
                except Exception:
                    post_snap["new_exposures"] = repr(getattr(model, "new_exposures", None))
                    post_snap["new_infections"] = repr(getattr(model, "new_infections", None))

                try:
                    post_snap["rng_after"] = canonicalize_rng_state(model.rng.bit_generator.state if getattr(model, "rng", None) is not None else None)
                except Exception:
                    post_snap["rng_after"] = repr(getattr(model, "rng", None))
                try:
                    post_snap["numba_legacy_seed_used"] = int(getattr(model, "_numba_legacy_seed_base", -1))  # base; time-specific seeds derived elsewhere
                except Exception:
                    post_snap["numba_legacy_seed_used"] = None

                step_entry["post"] = post_snap

                # create recorder snapshot to pass into LHD (duplicate simulate's behavior)
                if recorder is not None:
                    snapshot = recorder.snapshot_compact(copy=True)
                else:
                    snapshot = {
                        'event_time': np.empty(0, dtype=np.int32),
                        'event_source': np.empty(0, dtype=np.int32),
                        'event_type': np.empty(0, dtype=np.int16),
                        'event_nodes_start': np.empty(0, dtype=np.int64),
                        'event_nodes_len': np.empty(0, dtype=np.int32),
                        'nodes': np.empty(0, dtype=np.int32),
                        'infections': np.empty(0, dtype = bool)
                    }

                # Now call LHD step
                try:
                    model.lhd.step(model.current_time, snapshot)
                except Exception:
                    # capture LHD exception and attach
                    step_entry["lhd_after"] = {"error": "lhd.step failed", "traceback": traceback.format_exc()}
                    rep_rec["steps"].append(step_entry)
                    # re-raise? for diagnostics we continue to record the failure
                    break

                # capture LHD post-state
                try:
                    lhd_post = {
                        "lhd_rng_after": canonicalize_rng_state(model.lhd.rng.bit_generator.state if getattr(model.lhd, "rng", None) is not None else None),
                        "lhd_action_log_len": len(list(model.lhd.action_log)),
                        "lhd_recent_actions": canonicalize(model.lhd.action_log[-5:]) if getattr(model.lhd, "action_log", None) else None
                    }
                except Exception:
                    lhd_post = {"error": "cannot read lhd poststate"}

                step_entry["lhd_after"] = lhd_post

                # append step_record
                rep_rec["steps"].append(step_entry)

                # exit early if epidemic died out to mimic simulate stopping
                S, E, I, R = model.states_over_time[-1]
                if not E and not I:
                    # simulation ended early for this replicate
                    break

            except Exception:
                # record exception and break the time loop
                step_entry["error"] = traceback.format_exc()
                rep_rec["steps"].append(step_entry)
                break

        # store replicate-level recorded model action log
        try:
            rep_rec["final_lhd_action_log"] = canonicalize(list(model.lhd.action_log))
        except Exception:
            rep_rec["final_lhd_action_log"] = repr(getattr(model.lhd, "action_log", None))

        # save replicate record
        rec["replicates"].append(rep_rec)

    # write record to disk for inspection
    out_file = run_dir / f"diagnostics_variant_{variant_label}.json"
    try:
        with open(out_file, "w") as fh:
            json.dump(canonicalize(rec), fh, indent=2)
    except Exception:
        pass

    return rec

# Comparison engine -------------------------------------------------------

def find_first_mismatch(recA: Dict[str, Any], recB: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Compare two simulation-record dictionaries produced by emulate_simulation_and_record.
    Returns (are_equal, message). If not equal, message describes first mismatch.
    """
    # quick checks
    if recA.get("base_seed") != recB.get("base_seed"):
        return False, f"base_seed differs: {recA.get('base_seed')} vs {recB.get('base_seed')}"

    nA = len(recA.get("replicates", []))
    nB = len(recB.get("replicates", []))
    if nA != nB:
        return False, f"number of replicates differ: {nA} vs {nB}"

    for ridx in range(nA):
        a_rep = recA["replicates"][ridx]
        b_rep = recB["replicates"][ridx]

        # check init errors first
        if isinstance(a_rep.get("init"), dict) and a_rep["init"].get("error"):
            return False, f"variant A replicate {ridx} init error: {a_rep['init'].get('traceback')}"
        if isinstance(b_rep.get("init"), dict) and b_rep["init"].get("error"):
            return False, f"variant B replicate {ridx} init error: {b_rep['init'].get('traceback')}"

        # compare replicate seeds
        a_seed = a_rep["init"].get("replicate_seed")
        b_seed = b_rep["init"].get("replicate_seed")
        if a_seed != b_seed:
            return False, f"replicate_seed differs for replicate {ridx}: {a_seed} vs {b_seed}"

        # compare initial RNG samples if present
        a_rng_sample = a_rep["init"].get("rng_sample_first5")
        b_rng_sample = b_rep["init"].get("rng_sample_first5")
        if a_rng_sample != b_rng_sample:
            return False, f"rng_sample_first5 differ at replicate {ridx}: A={a_rng_sample} B={b_rng_sample}"

        a_lhd_sample = a_rep["init"].get("lhd_rng_sample_first5")
        b_lhd_sample = b_rep["init"].get("lhd_rng_sample_first5")
        if a_lhd_sample != b_lhd_sample:
            return False, f"lhd_rng samples differ at replicate {ridx}: A={a_lhd_sample} B={b_lhd_sample}"

        # compare is_vaccinated head
        a_iv = a_rep["init"].get("is_vaccinated_head")
        b_iv = b_rep["init"].get("is_vaccinated_head")
        if a_iv != b_iv:
            return False, f"is_vaccinated differs at replicate {ridx}: A head={a_iv} B head={b_iv}"

        # compare I0
        if canonicalize(a_rep["init"].get("I0")) != canonicalize(b_rep["init"].get("I0")):
            return False, f"I0 differs at replicate {ridx}: A={a_rep['init'].get('I0')} B={b_rep['init'].get('I0')}"

        # Now compare steps
        stepsA = a_rep.get("steps", [])
        stepsB = b_rep.get("steps", [])
        tmax = max(len(stepsA), len(stepsB))
        for sidx in range(tmax):
            if sidx >= len(stepsA):
                return False, f"Missing step {sidx} in variant A for replicate {ridx}"
            if sidx >= len(stepsB):
                return False, f"Missing step {sidx} in variant B for replicate {ridx}"
            sa = stepsA[sidx]
            sb = stepsB[sidx]
            # pre-state compare
            pre_a = sa.get("pre", {})
            pre_b = sb.get("pre", {})
            if not arrays_equal_sorted(pre_a.get("state_before"), pre_b.get("state_before")):
                return False, f"PRE state mismatch at replicate {ridx} time {sa.get('time')}: diff sample A={pre_a.get('state_before')[:20] if pre_a.get('state_before') else pre_a.get('state_before')} B={pre_b.get('state_before')[:20] if pre_b.get('state_before') else pre_b.get('state_before')}"
            # compare new_exposures after step
            post_a = sa.get("post", {})
            post_b = sb.get("post", {})
            if not arrays_equal_sorted(post_a.get("new_exposures"), post_b.get("new_exposures")):
                return False, f"new_exposures mismatch at replicate {ridx} time {sa.get('time')}: A={post_a.get('new_exposures')} B={post_b.get('new_exposures')}"
            # compare post-state
            if not arrays_equal_sorted(post_a.get("state_after"), post_b.get("state_after")):
                return False, f"POST state mismatch at replicate {ridx} time {sa.get('time')}: A_sample={str(post_a.get('state_after')[:20]) if post_a.get('state_after') else post_a.get('state_after')} B_sample={str(post_b.get('state_after')[:20]) if post_b.get('state_after') else post_b.get('state_after')}"
            # compare LHD action logs lengths and recent actions
            la = sa.get("lhd_after", {})
            lb = sb.get("lhd_after", {})
            if la is None and lb is None:
                continue
            if la is None or lb is None:
                return False, f"LHD post-state mismatch presence at replicate {ridx} time {sa.get('time')}: A={la} B={lb}"
            if la.get("lhd_action_log_len") != lb.get("lhd_action_log_len"):
                return False, f"LHD action log length mismatch at replicate {ridx} time {sa.get('time')}: A={la.get('lhd_action_log_len')} B={lb.get('lhd_action_log_len')}"
            if canonicalize(la.get("lhd_recent_actions")) != canonicalize(lb.get("lhd_recent_actions")):
                return False, f"LHD recent actions differ at replicate {ridx} time {sa.get('time')}: A={la.get('lhd_recent_actions')} B={lb.get('lhd_recent_actions')}"

    # If we reach here, treat as identical (for recorded items)
    return True, "No mismatches found"

# Orchestrator -------------------------------------------------------------

def run_diagnostics(csv_path: str,
                    n_samples: int,
                    lhd_config: LhdConfig,
                    output_dir: str,
                    base_cfg: Optional[ModelConfig] = None,
                    data_dir: str = "data",
                    seed: Optional[int] = None,
                    model_index: int = 0,
                    use_graphdata_copy: bool = False) -> int:
    """
    Run diagnostics for the given parameter set index and variants (first two by default),
    return 0 on identical, 1 on mismatch, 2 on error.
    """
    out_base = Path(output_dir).expanduser().resolve()
    diag_dir = out_base / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    print("[diagnostics] Preparing run data (prepare_run)...")
    contacts_df, configs_list, master_gd = prepare_run(
        csv_path=csv_path,
        n_samples=n_samples,
        output_dir=str(out_base),
        base_cfg=base_cfg,
        data_dir=data_dir,
        overwrite_files=True,
        save_files=False,
        seed=seed
    )
    if model_index >= len(configs_list):
        print(f"[diagnostics] ERROR: model_index {model_index} >= configs_list length {len(configs_list)}")
        return 2

    cfg = configs_list[int(model_index)]
    print(f"[diagnostics] Selected parameter-set index {model_index}; cfg id={id(cfg)}")

    # Compute run_seed and graphdata sampling seed similar to run_parameter_set
    base_seed = int(seed) if seed is not None else int(cfg.sim.seed)
    run_seed = base_seed + int(model_index)
    run_graphdata_seed = int(derive_seed_from_base(run_seed))

    print(f"[diagnostics] base_seed={base_seed} run_seed={run_seed} run_graphdata_seed={run_graphdata_seed}")

    # Sample/build run_graphdata (same as run_parameter_set)
    print("[diagnostics] Sampling edges from master graphdata...")
    sampled_edges_df = sample_from_master_graphdata(master_gd, cfg, seed=run_graphdata_seed)
    print(f"[diagnostics] Sampled edges: {len(sampled_edges_df)} rows")

    print("[diagnostics] Building run graphdata...")
    run_graphdata = build_graph_data(
        edge_list=sampled_edges_df,
        contacts_df=contacts_df,
        config=cfg,
        seed=run_graphdata_seed,
        N=int(contacts_df.shape[0])
    )
    # fingerprint of graphdata before any variant simulation
    gd_fp_before = fingerprint_graphdata(run_graphdata)
    with open(diag_dir / "graphdata_fingerprint_before.json", "w") as fh:
        json.dump(gd_fp_before, fh, indent=2)

    # Pick first two variants
    variants = list(lhd_config.variants)
    if len(variants) < 2:
        print("[diagnostics] ERROR: need at least two variants in LhdConfig to compare")
        return 2

    vA, vB = variants[0], variants[1]
    print(f"[diagnostics] Comparing variants: '{vA.name}' vs '{vB.name}'")

    # For parity with run_variants, create copies of alg/action maps for each variant
    import copy as _cpy
    algA = {k: _cpy.deepcopy(v) for k, v in getattr(vA, "algorithm_map", {}).items()}
    actA = {k: _cpy.deepcopy(v) for k, v in getattr(vA, "action_factory_map", {}).items()}
    algB = {k: _cpy.deepcopy(v) for k, v in getattr(vB, "algorithm_map", {}).items()}
    actB = {k: _cpy.deepcopy(v) for k, v in getattr(vB, "action_factory_map", {}).items()}

    # Create run_dir for diagnostics outputs (mirror driver structure)
    run_dir = Path(out_base) / f"model_{int(model_index):04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Decide whether to share run_graphdata or deep-copy for each variant (diagnose both if desired)
    print(f"[diagnostics] use_graphdata_copy={use_graphdata_copy}")

    # Build models and run manual emulate simulation & record
    recs = {}
    for label, alg_map, act_map in [("A", algA, actA), ("B", algB, actB)]:
        # choose graphdata instance
        if use_graphdata_copy:
            gd_for_variant = _cpy.deepcopy(run_graphdata)
        else:
            gd_for_variant = run_graphdata

        # make a per-variant config copy (safer)
        cfg_for_variant = _cpy.deepcopy(cfg)

        # validate variant (may instantiate algorithm instances)
        try:
            validate_variant(_cpy.deepcopy(vA if label == "A" else vB))
        except Exception:
            # continue even if validate fails (we still attempt to build models)
            pass

        # instantiate NetworkModel similar to run_variants
        try:
            model = NetworkModel(
                config=cfg_for_variant,
                graphdata=gd_for_variant,
                run_dir=str(run_dir),
                seed=run_seed,
                lhd_register_defaults=True,
                lhd_algorithm_map=dict(alg_map),
                lhd_action_factory_map=dict(act_map)
            )
        except Exception as exc:
            print(f"[diagnostics] ERROR instantiating model for variant {label}: {exc}")
            return 2

        # mark variant name for file naming/debugging
        setattr(model, "variant_name", (vA.name if label == "A" else vB.name))

        # run emulation recorder
        print(f"[diagnostics] Running emulated simulation for variant {label} ('{model.variant_name}') ...")
        try:
            rec = emulate_simulation_and_record(model, results_folder=diag_dir, variant_label=label)
        except Exception:
            rec = {"error": "emulation failed", "traceback": traceback.format_exc()}
        recs[label] = rec

    # fingerprint graphdata after both runs to detect mutation
    gd_fp_after = fingerprint_graphdata(run_graphdata)
    with open(diag_dir / "graphdata_fingerprint_after.json", "w") as fh:
        json.dump(gd_fp_after, fh, indent=2)

    # Save the two recs to disk
    with open(diag_dir / "recs_A.json", "w") as fh:
        json.dump(canonicalize(recs.get("A")), fh, indent=2)
    with open(diag_dir / "recs_B.json", "w") as fh:
        json.dump(canonicalize(recs.get("B")), fh, indent=2)

    # Compare records and report
    ok, message = find_first_mismatch(recs["A"], recs["B"])
    print("[diagnostics] Comparison result:", "IDENTICAL" if ok else "DIFFER", "-", message)
    print("[diagnostics] Detailed artifacts written to:", str(diag_dir))
    print("[diagnostics] graphdata fingerprint before:", gd_fp_before)
    print("[diagnostics] graphdata fingerprint after:", gd_fp_after)

    return 0 if ok else 1

# CLI ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Diagnostics: compare two identical variants run on the same parameter set step-by-step")
    p.add_argument("--csv", default="testLHS.csv", help="CSV path for LHS")
    p.add_argument("--n_samples", type=int, default=1, help="n_samples to generate in LHS (prepare_run)")
    p.add_argument("--outdir", default="diagnostics_out", help="Output directory to write diagnostics")
    p.add_argument("--seed", type=int, default=None, help="Base seed (passed to prepare_run). If omitted uses cfg.sim.seed")
    p.add_argument("--model_index", type=int, default=0, help="Index of parameter set to inspect")
    p.add_argument("--data_dir", default="data", help="Data directory for prepare_run")
    p.add_argument("--use_graphdata_copy", action="store_true", help="Use copies of run_graphdata per-variant (to test mutation hypothesis)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Use the default LHD configuration from your code (import may differ for your setup)
    try:
        from scripts.lhd.lhdConfig import LHD_CONFIGURATION as default_lhd_config
    except Exception:
        default_lhd_config = None

    # Provide a base config if available (ModelConfig default)
    try:
        base_cfg = ModelConfig()
    except Exception:
        base_cfg = None

    lhd_config = default_lhd_config or LhdConfig()
    rc = run_diagnostics(csv_path=args.csv,
                         n_samples=args.n_samples,
                         lhd_config=lhd_config,
                         output_dir=args.outdir,
                         base_cfg=base_cfg,
                         data_dir=args.data_dir,
                         seed=args.seed,
                         model_index=args.model_index,
                         use_graphdata_copy=args.use_graphdata_copy)
    if rc != 0:
        print(f"[diagnostics] Completed with status {rc} (differences or error).")
        raise SystemExit(rc)
    else:
        print("[diagnostics] Completed: no differences found for recorded items.")