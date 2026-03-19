#scripts/lhd/surveillance.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

import numpy as np

from scripts.utils.rng_utility import u01_for_nodes, derive_seed_from_base

#integer constants to represent disease stage
STAGE_PRE = np.int8(1) #pre-infectious (E)
STAGE_INF = np.int8(2) #Infectious (I)



class SurveillanceModel:
    """
    SurveillanceModel, how LHD gathers limited information about the outbreak
    F(Truth (epi_state) + Information Actions) -> SurveillanceBatch

    Baseline Behavior:
    - Detects some cases on transition from E -> I, with reporting delay
    - Pre-infectious cases are only discovered with active case-finding

    Information Actions:
    - order_test schedules test results, positives are queued as case reports
    - order_trace: schedule contact tracing, adds known edges to batch


    Behavior:
    - When individuals make the transition from E -> I, there is a baseline discovery probability
    - Pre-infectious individuals can't be detected without active case-finding
    - Fixed reporting delays (for now)

    Internally:
    - case_status: 0 = never detected, 1 = queued to report, 2 = reported
    """
    def __init__(
        self,
        *,
        neighbor_map: Dict[int, List[Tuple[int, float, Any]]],
        ct_to_id = Dict[str, int],
        seed: int,
        N: int,
        ages: np.ndarray,
        is_vax: np.ndarray,
        p_detect_inf: float,
        report_delay_days: int = 0
    ):




        self.seed = int(seed)
        self.N = int(N)

        self.ages = np.asarray(ages, dtype = np.int32)
        self.is_vax = np.asarray(is_vax, dtype = np.bool_)

        self.p_detect_inf = float(p_detect_inf)
        self.p_detect_pre = 0.0 #can't detect pre-infectious for now

        self.delay = int(report_delay_days)
        if self.delay < 0:
            raise ValueError("report_delay_days must be >= 0")

        #case report queue lag = delay + 1
        self._L = self.delay + 1
        self._queue_nodes: List[List[np.ndarray]] = [[] for _ in range(self._L)]
        self._queue_stage: List[List[np.ndarray]] = [[] for _ in range(self._L)]
        self._trace_orders = defaultdict(list) #due_day -> [cases_array, params]
        self._test_orders = defaultdict(list) #due_day -> [node_array, params]

        self.case_status = np.zeros(self.N, dtype=np.uint8)

        #True contact structure for contact tracing 
        self.neighbor_map = neighbor_map
        self.ct_to_id = ct_to_id



    #Methods for passing around information
    def _deliver_due(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export case reports queued for 'today' to LHD
        """
        #check for queued events today
        b = int(t % self._L)
        if not self._queue_nodes[b]:
            return np.empty(0, np.int32), np.empty(0, np.int8)


        nodes = np.concatenate(self._queue_nodes[b]).astype(np.int32, copy=False)
        stages = np.concatenate(self._queue_stage[b]).astype(np.int8, copy=False)
        self._queue_nodes[b].clear()
        self._queue_stage[b].clear()

        # mark case status reported (what LHD knows)
        self.case_status[nodes] = 2
        return nodes, stages

    def _queue(self, due_t: int, nodes: np.ndarray, stage_code: np.int8) -> None:
        """
        Add cases to report queue
        """
        if nodes.size == 0:
            return
        b = int(due_t % self._L)
        self._queue_nodes[b].append(nodes.astype(np.int32, copy=False))
        self._queue_stage[b].append(np.full(nodes.size, stage_code, dtype=np.int8))


    #Methods for information-requesting actions
    def order_trace(self, *, t: int, cases: np.ndarray, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Schedules a contact trace for a case (or array of cases)
        """
        params = params or {}
        delay = int(params.get("delay_days", 0))
        due = int(t + delay)
        self._trace_orders[due].append((np.asarray(cases, dtype=np.int32), dict(params)))

    def order_test(self, *, t: int, nodes: np.ndarray, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Schedules a test for an individual node (or array of nodes)
        """
        #params may contain test sens/spec
        params = params or {}

        delay = int(params.get("delay_days", 0))
        due = int(t + delay)
        self._test_orders[due].append((np.asarray(nodes, dtype=np.int32), dict(params)))

    def _process_due_tests(self, t: int, epi_state: Dict[str, np.ndarray]) -> None:
        """
        1) Pop test orders that are happening today
        2) "Test" Node
        3) Queue case reports for positive cases 
        """
        #1) pop test orders
        orders = self._test_orders.pop(int(t), [])
        if not orders:
            return

        #2) Determine test results
        pre_ids = np.asarray(epi_state.get("pre_ids", np.empty(0, np.int32)), dtype=np.int32)
        inf_ids = np.asarray(epi_state.get("inf_ids", np.empty(0, np.int32)), dtype=np.int32)

        for nodes, params in orders:
            nodes = np.asarray(nodes, dtype=np.int32)
            if nodes.size == 0:
                continue

            nodes = np.unique(nodes)
            nodes = nodes[self.case_status[nodes] == 0]
            if nodes.size == 0:
                continue

            # determine true stage membership via searchsorted on sorted arrays
            is_pre = self._isin_sorted(nodes, pre_ids)
            is_inf = self._isin_sorted(nodes, inf_ids)

            sens_pre = float(params.get("sens_pre", params.get("sens", 0.5)))
            sens_inf = float(params.get("sens_inf", params.get("sens", 0.99)))
            spec = float(params.get("spec", 1.0))  # default: no false positives
            spec = min(max(spec, 0.0), 1.0)

            # per-node probability of positive result
            #false-positives
            p_pos = np.full(nodes.size, 1.0 - spec, dtype=np.float64)
            #probability of picking up pre-infectious
            p_pos[is_pre] = sens_pre
            #probability of picking up post-infectious
            p_pos[is_inf] = sens_inf
            
            seed_test = int(derive_seed_from_base(self.seed, t, params.get("tag", '')))
            u = u01_for_nodes(seed_test, int(t), nodes, np.int8(3))
            pos_mask = u < p_pos
            if not pos_mask.any():
                continue

            pos_nodes = nodes[pos_mask]
            pos_is_pre = is_pre[pos_mask] 
            pos_is_inf = is_inf[pos_mask]

            #3) Send cases for reporting (with lag)

            # Filter out positive results that are already known cases
            pos_nodes = pos_nodes[self.case_status[pos_nodes] == 0]
            if pos_nodes.size == 0:
                continue
            self.case_status[pos_nodes] = 1

            report_delay = int(params.get("report_delay_days", self.delay))
            due_report = int(t + report_delay)


            det_pre = pos_nodes[pos_is_pre[:pos_nodes.size]] if pos_is_pre.size == pos_nodes.size else pos_nodes[0:0]
            det_inf = pos_nodes[pos_is_inf[:pos_nodes.size]] if pos_is_inf.size == pos_nodes.size else pos_nodes[0:0]
            det_unk = pos_nodes[(~pos_is_pre) & (~pos_is_inf)] if pos_is_pre.size == pos_nodes.size else pos_nodes

            if det_pre.size:
                self._enqueue(due_report, det_pre, STAGE_PRE)
            if det_inf.size:
                self._enqueue(due_report, det_inf, STAGE_INF)
            if det_unk.size:
                self._enqueue(due_report, det_unk, np.int8(0))  # unknown/false-positive stage

    def _process_due_traces(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pop trace orders due today; return trace_src/trace_tgt/trace_ct arrays for the batch.

        1) Pop trace orders due today
        2) Identify contacts with probability
        3) Return src/tgt/ct arrays
        """
        orders = self._trace_orders.pop(int(t), [])
        if not orders:
            return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int16))

        src_parts: List[np.ndarray] = []
        tgt_parts: List[np.ndarray] = []
        ct_parts: List[np.ndarray] = []

        for cases, params in orders:
            cases = np.asarray(cases, dtype=np.int32)
            if cases.size == 0:
                continue

            cases = np.unique(cases)

            recall_prob = float(params.get("recall_prob", 0.25))
            recall_prob = min(max(recall_prob, 0.0), 1.0)
            max_per_case = int(params.get("max_per_case", 25))

            # contact type whitelist
            raw_cts = params.get("contact_types", None)
            if raw_cts is None:
                raw_cts = ["hh", "sch", "wp"]  #tracing focuses on non-casual contacts
            whitelist_ids = set()
            for ct in raw_cts:
                cid = self._ctid(ct)
                if cid >= 0:
                    whitelist_ids.add(cid)

            for src in cases.tolist():
                neigh = self.neighbor_map.get(int(src), [])
                if not neigh:
                    continue

                nbrs_list = []
                ctids_list = []

                for (nbr, _w, ct) in neigh:
                    cid = self._ctid(ct)
                    if cid < 0:
                        continue
                    if whitelist_ids and cid not in whitelist_ids:
                        continue
                    nbrs_list.append(int(nbr))
                    ctids_list.append(cid)

                if not nbrs_list:
                    continue

                nbrs = np.asarray(nbrs_list, dtype=np.int32)
                ctids = np.asarray(ctids_list, dtype=np.int16)

                # Determine who was recalled by traced source
                seed_recall = int(derive_seed_from_base(self.seed, t, src, "recall"))
                u = u01_for_nodes(seed_recall, int(t), nbrs, STAGE_INF)
                keep = u < recall_prob
                if not keep.any():
                    continue

                nbrs = nbrs[keep]
                ctids = ctids[keep]

                # cap number of contacts traced per person (reasonably)
                if max_per_case > 0 and nbrs.size > max_per_case:
                    seed_rank = int(derive_seed_from_base(self.seed, t, src, "rank"))
                    r = u01_for_nodes(seed_rank, int(t), nbrs, STAGE_PRE)
                    order = np.argsort(r)[:max_per_case]
                    nbrs = nbrs[order]
                    ctids = ctids[order]

                src_parts.append(np.full(nbrs.size, int(src), dtype=np.int32))
                tgt_parts.append(nbrs.astype(np.int32, copy=False))
                ct_parts.append(ctids.astype(np.int16, copy=False))

        if not src_parts:
            return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int16))

        return (
            np.concatenate(src_parts).astype(np.int32, copy=False),
            np.concatenate(tgt_parts).astype(np.int32, copy=False),
            np.concatenate(ct_parts).astype(np.int16, copy=False),
        )


    def step(
        self, *,
        t: int, epi_state: Dict[str, np.ndarray], 
        scheduled_actions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Returns a daily SurveillanceBatch dict with:
        - reported_cases (and relevant attributes)
        - trace_src/trace_tgt/trace_ct if contact tracing done
        """
        t = int(t)

        #1) Check if testing/contact tracing were scheduled
        if scheduled_actions:
            if "trace_cases" in scheduled_actions:
                self.order_trace(t=t, cases=scheduled_actions["trace_cases"], params=scheduled_actions.get("trace_params"))
            if "test_nodes" in scheduled_actions:
                self.order_test(t=t, nodes=scheduled_actions["test_nodes"], params=scheduled_actions.get("test_params"))

        #2) Baseline detection -- detect some cases on transition from E -> I


        #Determine which cases are detected, and schedule them to be reported
        new_inf = np.asarray(epi_state.get("new_inf_ids", np.empty(0, np.int32)), dtype = np.int32)
        if new_inf.size > 0 and self.p_detect_inf > 0.0:
            cand = new_inf[self.case_status[new_inf] == 0]
            if cand.size > 0:
                #splitmix pseudo-random draw for determinism
                u = u01_for_nodes(self.seed, t, cand, STAGE_INF)
                det = cand[u < self.p_detect_inf]
                if det.size > 0:
                    self.case_status[det] = 1 #Queue discovered events
                    self._queue(t + self.delay, det, STAGE_INF)

        #3) Handle active case-finding 
        self._process_due_tests(t, epi_state)
        trace_src, trace_tgt, trace_ct = self._process_due_traces(t)

        #4) Deliver reports due today
        reported_nodes, reported_stage = self._deliver_due(t)
        if reported_nodes.size == 0:
            batch = _empty_batch(t)
        else:
            batch = {
                "t": np.int32(t),
                "reported_cases": reported_nodes,
                "report_time": np.full(reported_nodes.size, np.int32(t), dtype=np.int32),
                "reported_stage": reported_stage,
                "age": self.ages[reported_nodes].astype(np.int32, copy=False),
                "is_vax": self.is_vax[reported_nodes].astype(np.bool_, copy=False),
                "trace_src": np.empty(0, dtype=np.int32),
                "trace_tgt": np.empty(0, dtype=np.int32),
                "trace_ct": np.empty(0, dtype=np.int16),
            }

            
        return batch


    def reset_for_run(self, *, 
    seed: Optional[int] = None, is_vax: Optional[np.ndarray] = None) -> None:
        """
        Reset surveillance object for a new run
        """
        if seed is not None:
            self.seed = int(seed)
        if is_vax is not None:
            self.is_vax = np.asarray(is_vax, dtype=np.bool_)
        self.case_status.fill(0)
        for b in range(self._L):
            self._queue_nodes[b].clear()
            self._queue_stage[b].clear()

        self._trace_orders.clear()
        self._test_orders.clear()

    ### INTERNAL HELPERS
    @staticmethod
    def _isin_sorted(a: np.ndarray, sorted_b: np.ndarray) -> np.ndarray:
        """
        Check that a belongs to the sorted list b
        """
        if a.size == 0 or sorted_b.size == 0:
            return np.zeros(a.size, dtype=bool)
        idx = np.searchsorted(sorted_b, a)
        ok = (idx < sorted_b.size) & (sorted_b[idx] == a)
        return ok


    def _ctid(self, ct: Any) -> int:
        """
        Convert string contact types to ct_ids
        """
        if isinstance(ct, (int, np.integer)):
            return int(ct)
        return int(self.ct_to_id.get(str(ct), -1))




def _empty_batch(t: int) -> Dict[str, np.ndarray]:
    #Helper to quickly return an empty batch if no events are discoverred
    return {
        "t": np.int32(t),
        "reported_cases": np.empty(0, dtype=np.int32),
        "report_time": np.empty(0, dtype=np.int32),
        "reported_stage": np.empty(0, dtype=np.int8),
        "age": np.empty(0, dtype=np.int32),
        "is_vax": np.empty(0, dtype=np.bool_),
        
        #TODO Implement contact tracing
        "trace_src": np.empty(0, dtype=np.int32),
        "trace_tgt": np.empty(0, dtype=np.int32),
        "trace_ct": np.empty(0, dtype=np.int16),
    }