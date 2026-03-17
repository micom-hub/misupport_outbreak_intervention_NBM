#scripts/lhd/surveillance.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from scripts.utils.rng_utility import u01_for_nodes

#integer constants to represent disease stage
STAGE_PRE = np.int8(1) #pre-infectious (E)
STAGE_INF = np.int8(2) #Infectious (I)



class SurveillanceModel:
    """
    SurveilanceModel represents the interface between the true outbreak network, and what the local health department is aware of by converting "truth" (updated E/I sets) into daily SurveillanceBatch objects for LHD

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

        self._L = self.delay + 1
        self._queue_nodes: List[List[np.ndarray]] = [[] for _ in range(self._L)]
        self._queue_stage: List[List[np.ndarray]] = [[] for _ in range(self._L)]

        self.case_status = np.zeros(self.N, dtype=np.uint8)


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

    def _deliver_due(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export case reports queued for 'today' to LHD
        """
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

    def _enqueue(self, due_t: int, nodes: np.ndarray, stage_code: np.int8) -> None:
        """
        Add nodes to queue
        """
        if nodes.size == 0:
            return
        b = int(due_t % self._L)
        self._queue_nodes[b].append(nodes.astype(np.int32, copy=False))
        self._queue_stage[b].append(np.full(nodes.size, stage_code, dtype=np.int8))

    def step(
        self, *,
        t: int, epi_state: Dict[str, np.ndarray], 
        scheduled_actions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Inputs (epi_state dict must include):
        - new_inf_ids: int array of nodes newly infectious this dt
        - new_pre_ids: int array of nodes newly exposed today
        - pre_ids: current pre-infectious set of nodes
        - inf_ids: current infectious set of nodes

        Output: SurveillanceBatch dict
        """
        t = int(t)



        #1) Baseline detection -- detect some cases on transition from E -> I


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
                    self._enqueue(t + self.delay, det, STAGE_INF)

        #2) Handle active casefinding and information gain events
        #To be done



        #3) "Deliver" reports scheduled for today
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