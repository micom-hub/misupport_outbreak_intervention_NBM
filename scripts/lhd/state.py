#scripts/lhd/state.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Iterable
from collections import defaultdict

import numpy as np

try:
    # optional import; used only for type hints / token storage
    from scripts.lhd.actions import ActionToken
except Exception:  # pragma: no cover
    ActionToken = object

class LHDState:
    """
    LHDState represents continuously updated partial knowledge of outbreak dynamics for the LHD.

    LHDState contains:
    - known cases (time reported, infectious stage, known attributes)
    - known edges (contact edges, currently treated as "fact")
    - known_graph (incrementally built network view)
    - interventions/pending_requests: placeholder for now

    Future iteration will aim to change "known truths" to "strength of evidence", which will enable more information-gain analyses
    
    """
    def __init__(self, N: int):
        self.N = int(N)

        #1) Establish case registry

        #bool, whether each index is a known case
        self.known_case = np.zeros(self.N, dtype = np.bool_)
        #list of known cases
        self.known_case_list = []
        #time a case was reported, -1 if never reported
        self.case_report_time = np.full(self.N, -1, dtype = np.int32)
        #knowledge of case stage - 0/unknown, 1/pre-infectious, 2/infectious
        self.case_stage = np.zeros(self.N, dtype=np.int8) 

        #attributes as known when/if reported
        self.case_age = np.full(self.N, -1, dtype = np.int32)
        self.case_is_vax_known = np.zeros(self.N, dtype = np.bool_)
        self.case_is_vax = np.zeros(self.N, dtype = np.bool_)


        #2) Known Contact Structure

        #undirected edge key for (i, j, ct)
        self._edge_seen = set()

        #edge log, (i, j, ct_id, discovered_time, information_source)
        self.known_edges: List[tuple[int, int, int, int, str]] = []

        #adjacency i -> list of (j, ct_id, discovered_time, information_source)
        self.known_adj: Dict[int, List[Tuple[int, int, int, str]]] = defaultdict(list)


        #3) Tracking interventions

        #separating sick people
        self.isolated_until = np.full(self.N, -1, dtype=np.int32)
        self.quarantined_until = np.full(self.N, -1, dtype = np.int32)

        #Pending requests
        self.pending_tests = defaultdict(list)
        self.pending_traces = defaultdict(list)

    # Surveillance Batches -> Updating LHD Knowledge
    def process_batch(self, batch: Dict[str, np.ndarray]) -> None:
        """
        Update LHD knowledge from a SurveillanceBatch.
        Expects batch dict with keys:
          - t
          - reported_cases, report_time, reported_stage, age, is_vax
          - trace_src, trace_tgt, trace_ct (can be empty)
        """
        t = int(batch.get("t", -1))

        cases = np.asarray(batch.get("reported_cases", np.empty(0, np.int32)), dtype=np.int32)
        if cases.size:
            stage = np.asarray(batch.get("reported_stage", np.zeros(cases.size, np.int8)), dtype=np.int8)
            rtime = np.asarray(batch.get("report_time", np.full(cases.size, t, np.int32)), dtype=np.int32)
            age = np.asarray(batch.get("age", np.full(cases.size, -1, np.int32)), dtype=np.int32)
            is_vax = np.asarray(batch.get("is_vax", np.zeros(cases.size, np.bool_)), dtype=np.bool_)

            # For reported cases, only keep first time reported (if mult)
            _, first_ind = np.unique(cases, return_index=True)
            first_ind.sort()
            cases = cases[first_ind]
            stage = stage[first_ind] if stage.size else np.zeros(cases.size, np.int8)
            rtime = rtime[first_ind] if rtime.size else np.full(cases.size, np.int32(t), np.int32)
            age = age[first_ind] if age.size else np.full(cases.size, -1, np.int32)
            is_vax = is_vax[first_ind] if is_vax.size else np.zeros(cases.size, np.bool_)
            
            #If the case hasn't been reported before, update info for that case
            new_mask = ~self.known_case[cases]
            if new_mask.any():
                new_cases = cases[new_mask]
                self.known_case[new_cases] = True
                self.case_report_time[new_cases] = rtime[new_mask]
                self.case_stage[new_cases] = stage[new_mask]
                self.case_age[new_cases] = age[new_mask]
                self.case_is_vax[new_cases] = is_vax[new_mask]
                self.case_is_vax_known[new_cases] = True
                self.known_case_list.extend(new_cases.tolist())

        # Use contact-tracing to assemble edges
        self._process_trace_edges(batch, t=t)

    def _process_trace_edges(self, batch: Dict[str, np.ndarray], *, t: int) -> None:
        src = np.asarray(batch.get("trace_src", np.empty(0, np.int32)), dtype=np.int32)
        if src.size == 0:
            return
        tgt = np.asarray(batch.get("trace_tgt", np.empty(0, np.int32)), dtype=np.int32)
        ct = np.asarray(batch.get("trace_ct", np.empty(0, np.int16)), dtype=np.int16)

        m = min(src.size, tgt.size, ct.size)
        for i in range(m):
            u = int(src[i]); v = int(tgt[i]); c = int(ct[i])
            if u == v or u < 0 or v < 0 or u >= self.N or v >= self.N:
                continue
            a, b = (u, v) if u < v else (v, u)

            #code edges as individual1, individual2, contact_type
            key = (a, b, c)
            if key in self._edge_seen:
                continue
            self._edge_seen.add(key)

            self.known_edges.append((a, b, c, int(t), "trace"))
            # adjacency in both directions
            self.known_adj[a].append((b, c, int(t), "trace"))
            self.known_adj[b].append((a, c, int(t), "trace"))

   
   #Helper function to get known neighbors of a node. 
    def neighbors(self, node: int) -> List[Tuple[int, int, int, str]]:
        """Return known neighbors of a node in the known graph."""
        return self.known_adj.get(int(node), [])
    

    #Helper function to reset LHDState between runs
    def reset_for_run(self) -> None:
        self.known_case.fill(False)
        self.case_report_time.fill(-1)
        self.case_stage.fill(0)
        self.case_age.fill(-1)
        self.case_is_vax_known.fill(False)
        self.case_is_vax.fill(False)
        self.known_case_list.clear()

        self._edge_seen.clear()
        self.known_edges.clear()
        self.known_adj.clear()

        self.isolated_until.fill(-1)
        self.quarantined_until.fill(-1)

        self.pending_tests.clear()
        self.pending_traces.clear()



