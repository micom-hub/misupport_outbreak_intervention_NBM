# scripts/lhd/surveillance.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.utils.rng_utility import derive_seed_from_base, u01_for_nodes

# integer constants to represent disease stage
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
        report_delay_days: int = 0,
        p_detect_imported_daily: float = 0.0,
        imported_detection_start_day: int = 1,
    ):

        self.seed = int(seed)
        self.N = int(N)

        self.ages = np.asarray(ages, dtype=np.int32)
        self.is_vax = np.asarray(is_vax, dtype=np.bool_)

        self.p_detect_inf = float(p_detect_inf)
        self.p_detect_pre = 0.0  # can't detect pre-infectious for now
        self.p_detect_imported_daily = float(p_detect_imported_daily)
        self.imported_detection_start_day = int(imported_detection_start_day)

        self.delay = int(report_delay_days)
        if self.delay < 0:
            raise ValueError("report_delay_days must be >= 0")

        # case report queue lag = delay + 1
        self._L = self.delay + 1
        self._queue_nodes: List[List[np.ndarray]] = [[] for _ in range(self._L)]
        self._queue_stage: List[List[np.ndarray]] = [[] for _ in range(self._L)]
        self._trace_orders = defaultdict(list)  # due_day -> [cases_array, params]
        self._queue_source: List[List[np.ndarray]] = [[] for _ in range(self._L)]
        self._test_orders = defaultdict(list)  # due_day -> [node_array, params]

        self.case_status = np.zeros(self.N, dtype=np.uint8)

        # True contact structure for contact tracing
        self.neighbor_map = neighbor_map
        self.ct_to_id = ct_to_id

    # Methods for passing around information
    def _deliver_due(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Export case reports queued for 'today' to LHD
        """
        # check for queued events today
        b = int(t % self._L)
        if not self._queue_nodes[b]:
            return np.empty(0, np.int32), np.empty(0, np.int8), np.empty(0, dtype=object)

        nodes = np.concatenate(self._queue_nodes[b]).astype(np.int32, copy=False)
        stages = np.concatenate(self._queue_stage[b]).astype(np.int8, copy=False)
        sources = np.concatenate(self._queue_source[b]).astype(object, copy=False)
        self._queue_nodes[b].clear()
        self._queue_stage[b].clear()
        self._queue_source[b].clear()
        # mark case status reported (what LHD knows)
        self.case_status[nodes] = 2
        return nodes, stages, sources

    def _queue(self, due_t: int, nodes: np.ndarray, stage_code: np.int8, report_source: str) -> None:
        """
        Add cases to report queue
        """
        if nodes.size == 0:
            return
        b = int(due_t % self._L)
        self._queue_nodes[b].append(nodes.astype(np.int32, copy=False))
        self._queue_stage[b].append(np.full(nodes.size, stage_code, dtype=np.int8))
        self._queue_source[b].append(np.full(nodes.size, report_source, dtype=object))

    # Methods for information-requesting actions
    def order_trace(
        self, *, t: int, cases: np.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Schedules a contact trace for a case (or array of cases)
        """
        params = params or {}
        delay = max(1, int(params.get("delay_days", 1)))
        params["delay_days"] = delay
        due = int(t + delay)
        self._trace_orders[due].append(
            (np.asarray(cases, dtype=np.int32), dict(params))
        )

    def order_test(
        self, *, t: int, nodes: np.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Schedules a test for an individual node (or array of nodes)
        """
        # params may contain test sens/spec
        params = params or {}

        delay = max(1, int(params.get("delay_days", 1)))
        params["delay_days"] = delay
        due = int(t + delay)
        self._test_orders[due].append((np.asarray(nodes, dtype=np.int32), dict(params)))

    def _process_due_tests(
        self,
        t: int,
        epi_state: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Process all test orders due on or before day t.

        Returns the unique nodes whose test orders completed, regardless of result.
        """
        due_keys = [k for k in self._test_orders.keys() if k <= t]
        orders = []
        for k in sorted(due_keys):
            orders.extend(self._test_orders.pop(k))

        if not orders:
            return np.empty(0, dtype=np.int32)

        completed_parts = []

        pre_ids = np.sort(
            np.asarray(
                epi_state.get("pre_ids", np.empty(0, np.int32)),
                dtype=np.int32,
            )
        )
        inf_ids = np.sort(
            np.asarray(
                epi_state.get("inf_ids", np.empty(0, np.int32)),
                dtype=np.int32,
            )
        )

        for nodes, params in orders:
            nodes = np.unique(np.asarray(nodes, dtype=np.int32))
            if nodes.size == 0:
                continue

            completed_parts.append(nodes.copy())

            eligible_nodes = nodes[self.case_status[nodes] == 0]
            if eligible_nodes.size == 0:
                continue

            is_pre = self._isin_sorted(eligible_nodes, pre_ids)
            is_inf = self._isin_sorted(eligible_nodes, inf_ids)

            sens_pre = float(params.get("sens_pre", params.get("sens", 0.5)))
            sens_inf = float(params.get("sens_inf", params.get("sens", 0.99)))
            spec = float(np.clip(params.get("spec", 1.0), 0.0, 1.0))

            p_pos = np.full(
                eligible_nodes.size,
                1.0 - spec,
                dtype=np.float64,
            )
            p_pos[is_pre] = sens_pre
            p_pos[is_inf] = sens_inf

            seed_test = int(
                derive_seed_from_base(
                    self.seed,
                    t,
                    params.get("tag", ""),
                    "test",
                )
            )
            u = u01_for_nodes(
                seed_test,
                int(t),
                eligible_nodes,
                np.int8(3),
            )
            pos_mask = u < p_pos

            if not pos_mask.any():
                continue

            pos_nodes = eligible_nodes[pos_mask]
            pos_is_pre = is_pre[pos_mask]
            pos_is_inf = is_inf[pos_mask]

            self.case_status[pos_nodes] = 1

            report_delay = max(
                0,
                int(params.get("report_delay_days", self.delay)),
            )
            due_report = int(t + report_delay)

            det_pre = pos_nodes[pos_is_pre]
            det_inf = pos_nodes[pos_is_inf]
            det_unk = pos_nodes[(~pos_is_pre) & (~pos_is_inf)]

            if det_pre.size:
                self._queue(
                    due_report,
                    det_pre,
                    STAGE_PRE,
                    "tested_positive",
                )
            if det_inf.size:
                self._queue(
                    due_report,
                    det_inf,
                    STAGE_INF,
                    "tested_positive",
                )
            if det_unk.size:
                self._queue(
                    due_report,
                    det_unk,
                    np.int8(0),
                    "tested_positive",
                )

        if not completed_parts:
            return np.empty(0, dtype=np.int32)

        return np.unique(np.concatenate(completed_parts)).astype(
            np.int32,
            copy=False,
        )

    def _process_due_traces(
        self,
        t: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pop trace orders due today; return trace_src/trace_tgt/trace_ct arrays for the batch.

        1) Pop trace orders due today
        2) Identify contacts with probability
        3) Return src/tgt/ct arrays
        """
        # Collect all orders due now or in the past
        due_keys = [k for k in self._trace_orders.keys() if k <= t]

        # gather all tracing orders
        orders = []
        for k in sorted(due_keys):
            orders.extend(self._trace_orders.pop(k))
        # return early if nothing to do
        if not orders:
            return (
                np.empty(0, np.int32),
                np.empty(0, np.int32),
                np.empty(0, np.int16),
                np.empty(0, np.int32),
            )

        src_parts: List[np.ndarray] = []
        tgt_parts: List[np.ndarray] = []
        ct_parts: List[np.ndarray] = []
        completed_cases_parts: List[np.ndarray] = []


        for cases, params in orders:
            cases = np.asarray(cases, dtype=np.int32)
            if cases.size == 0:
                continue

            cases = np.unique(cases)
            completed_cases_parts.append(cases.copy())

            recall_prob = float(params.get("recall_prob", 0.25))
            recall_prob = min(max(recall_prob, 0.0), 1.0)
            max_per_case = int(params.get("max_per_case", 25))

            # contact type whitelist
            raw_cts = params.get("contact_types", None)
            if raw_cts is None:
                raw_cts = ["hh", "sch", "wp"]  # tracing focuses on non-casual contacts
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

                for nbr, _w, ct in neigh:
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
        completed_cases = (
        np.unique(np.concatenate(completed_cases_parts)).astype(
            np.int32,
            copy=False,
        )
        if completed_cases_parts
        else np.empty(0, np.int32)
    )

        if not src_parts:
            return (
                np.empty(0, np.int32),
                np.empty(0, np.int32),
                np.empty(0, np.int16),
                completed_cases,
            )

        # otherwise, return the findings
        return (
            np.concatenate(src_parts).astype(np.int32, copy=False),
            np.concatenate(tgt_parts).astype(np.int32, copy=False),
            np.concatenate(ct_parts).astype(np.int16, copy=False),
            completed_cases
        )
    def _process_imported_detection(
        self,
        t: int,
        epi_state: Dict[str, np.ndarray],
    ) -> None:
        """
        Apply a distinct daily ascertainment process to imported infectious cases.

        Imported cases are not automatically reported. Beginning on
        imported_detection_start_day, each imported case that is still infectious
        and has never been detected receives one deterministic daily ascertainment
        draw. Detected imported cases then enter the normal reporting queue.
        """
        if t < self.imported_detection_start_day:
            return
        if self.p_detect_imported_daily <= 0.0:
            return

        imported_ids = np.asarray(
            epi_state.get("imported_inf_ids", np.empty(0, np.int32)),
            dtype=np.int32,
        )
        inf_ids = np.sort(
            np.asarray(
                epi_state.get("inf_ids", np.empty(0, np.int32)),
                dtype=np.int32,
            )
        )

        if imported_ids.size == 0 or inf_ids.size == 0:
            return

        imported_ids = np.unique(imported_ids)
        still_infectious = self._isin_sorted(imported_ids, inf_ids)
        candidates = imported_ids[still_infectious]
        candidates = candidates[self.case_status[candidates] == 0]

        if candidates.size == 0:
            return

        p_detect = float(np.clip(self.p_detect_imported_daily, 0.0, 1.0))
        imported_seed = int(
            derive_seed_from_base(self.seed, "imported_detection")
        )
        u = u01_for_nodes(
            imported_seed,
            int(t),
            candidates,
            np.int8(4),
        )
        detected = candidates[u < p_detect]

        if detected.size:
            self.case_status[detected] = 1
            self._queue(
                int(t + self.delay),
                detected,
                STAGE_INF,
                "imported_passive",
            )

    def step(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
        scheduled_actions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Returns a daily SurveillanceBatch dict with:
        - reported_cases (and relevant attributes)
        - trace_src/trace_tgt/trace_ct if contact tracing done
        """
        t = int(t)

        # 1) Check if testing/contact tracing were scheduled
        if scheduled_actions:
            if "trace_cases" in scheduled_actions:
                self.order_trace(
                    t=t,
                    cases=scheduled_actions["trace_cases"],
                    params=scheduled_actions.get("trace_params"),
                )
            if "test_nodes" in scheduled_actions:
                self.order_test(
                    t=t,
                    nodes=scheduled_actions["test_nodes"],
                    params=scheduled_actions.get("test_params"),
                )

        # 2) Baseline detection -- detect some cases on transition from E -> I

        # Determine which cases are detected, and schedule them to be reported
        new_inf = np.asarray(
            epi_state.get("new_inf_ids", np.empty(0, np.int32)), dtype=np.int32
        )
        if new_inf.size > 0 and self.p_detect_inf > 0.0:
            cand = new_inf[self.case_status[new_inf] == 0]
            if cand.size > 0:
                # splitmix pseudo-random draw for determinism
                u = u01_for_nodes(self.seed, t, cand, STAGE_INF)
                det = cand[u < self.p_detect_inf]
                if det.size > 0:
                    self.case_status[det] = 1  # Queue discovered events
                    self._queue(t + self.delay, det, STAGE_INF, "passive")
        
        # Imported cases have a separate delayed daily discovery process to prevent instant die-out
        self._process_imported_detection(t, epi_state)
                

        # 3) Handle active case-finding with processing due tests/traces
        tests_completed = self._process_due_tests(t, epi_state)
        trace_src, trace_tgt, trace_ct, traces_completed = self._process_due_traces(t)

        # 4) Deliver reports due today
        reported_nodes, reported_stage, reported_sources = self._deliver_due(t)

        batch = {
            "t": np.int32(t),
            "reported_cases": reported_nodes,
            "report_time": (
                np.full(reported_nodes.size, np.int32(t), dtype=np.int32)
                if reported_nodes.size > 0
                else np.empty(0, dtype=np.int32)
            ),
            "reported_stage": (
                reported_stage
                if reported_nodes.size > 0
                else np.empty(0, dtype=np.int8)
            ),
            "age": (
                self.ages[reported_nodes].astype(np.int32, copy=False)
                if reported_nodes.size > 0
                else np.empty(0, dtype=np.int32)
            ),
            "is_vax": (
                self.is_vax[reported_nodes].astype(np.bool_, copy=False)
                if reported_nodes.size > 0
                else np.empty(0, dtype=np.bool_)
            ),
            "reported_sources": (
                reported_sources
                if reported_nodes.size > 0
                else np.empty(0, dtype=object)),
            "tests_completed": tests_completed,
            "traces_completed": traces_completed,
            "trace_src": trace_src,
            "trace_tgt": trace_tgt,
            "trace_ct": trace_ct,
        }

        return batch

    def reset_for_run(
        self, *, seed: Optional[int] = None, is_vax: Optional[np.ndarray] = None
    ) -> None:
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
            self._queue_source[b].clear()

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
        # Ensure indices are within bounds before accessing the array
        idx_clipped = np.clip(idx, 0, sorted_b.size - 1)
        ok = sorted_b[idx_clipped] == a
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

        "tests_completed": np.empty(0, dtype=np.int32),
        "traces_completed": np.empty(0, dtype=np.int32),
        
        "trace_src": np.empty(0, dtype=np.int32),
        "trace_tgt": np.empty(0, dtype=np.int32),
        "trace_ct": np.empty(0, dtype=np.int16),
    }
