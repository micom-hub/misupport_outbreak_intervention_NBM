from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.utils.rng_utility import (
    derive_seed_from_base,
    u01_for_nodes,
)

STAGE_UNKNOWN = np.int8(0)
STAGE_PRE = np.int8(1)
STAGE_INF = np.int8(2)


class SurveillanceModel:
    """
    Surveillance and active case-finding model.

    Passive ascertainment:
      - evaluates every currently infectious, undetected person each day;
      - uses different daily hazards for symptomatic and non-symptomatic
        infectious people;
      - gives imported cases an optional additional ascertainment channel.

    Active testing:
      - completes after a configured turnaround;
      - produces a result event;
      - positive results are detected cases;
      - positive reports reach the LHD after a reporting delay.

    Detection and report receipt are separate events.
    """

    def __init__(
        self,
        *,
        neighbor_map: Dict[
            int,
            List[Tuple[int, float, Any]],
        ],
        ct_to_id: Dict[str, int],
        seed: int,
        N: int,
        ages: np.ndarray,
        is_vax: np.ndarray,
        p_detect_symptomatic_daily: float,
        p_detect_asymptomatic_daily: float,
        p_detect_imported_daily: float,
        imported_detection_start_day: int,
        passive_report_delay_days: int,
    ):
        self.seed = int(seed)
        self.N = int(N)

        self.ages = np.asarray(
            ages,
            dtype=np.int32,
        )
        self.is_vax = np.asarray(
            is_vax,
            dtype=np.bool_,
        )

        self.p_detect_symptomatic_daily = float(
            p_detect_symptomatic_daily
        )
        self.p_detect_asymptomatic_daily = float(
            p_detect_asymptomatic_daily
        )
        self.p_detect_imported_daily = float(
            p_detect_imported_daily
        )

        self.imported_detection_start_day = int(
            imported_detection_start_day
        )
        self.passive_report_delay_days = int(
            passive_report_delay_days
        )

        if self.passive_report_delay_days < 0:
            raise ValueError(
                "passive_report_delay_days must be >= 0"
            )

        # due day -> report records
        self._report_queue = defaultdict(list)

        # due day -> [(nodes, params)]
        self._test_orders = defaultdict(list)
        self._trace_orders = defaultdict(list)

        # 0 = never detected
        # 1 = detected, report pending
        # 2 = report received by LHD
        self.case_status = np.zeros(
            self.N,
            dtype=np.uint8,
        )

        self.neighbor_map = neighbor_map
        self.ct_to_id = ct_to_id

        self.model = None

    def attach_model(self, model) -> None:
        """
        Attach the NetworkModel event sink after initialization.
        """
        self.model = model

    def _emit(
        self,
        *,
        t: int,
        event_type: str,
        node_id: Optional[int] = None,
        source_node_id: Optional[int] = None,
        detection_source: Optional[str] = None,
        report_source: Optional[str] = None,
        disease_stage: Optional[str] = None,
        result: Optional[str] = None,
        contact_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.model is None:
            return

        self.model.record_event(
            day=int(t),
            event_type=event_type,
            node_id=node_id,
            source_node_id=source_node_id,
            detection_source=detection_source,
            report_source=report_source,
            disease_stage=disease_stage,
            result=result,
            contact_type=contact_type,
            metadata=metadata,
        )

    def _queue_report(
        self,
        *,
        detection_day: int,
        nodes: np.ndarray,
        stage_code: np.int8,
        report_source: str,
        report_delay_days: int,
    ) -> None:
        nodes = np.asarray(
            nodes,
            dtype=np.int32,
        )

        if nodes.size == 0:
            return

        delay = max(0, int(report_delay_days))
        due_day = int(detection_day + delay)

        stages = np.full(
            nodes.size,
            stage_code,
            dtype=np.int8,
        )
        sources = np.full(
            nodes.size,
            report_source,
            dtype=object,
        )
        detection_days = np.full(
            nodes.size,
            int(detection_day),
            dtype=np.int32,
        )

        self._report_queue[due_day].append(
            (
                nodes.copy(),
                stages,
                sources,
                detection_days,
            )
        )

        for node in nodes:
            self._emit(
                t=detection_day,
                event_type="report_scheduled",
                node_id=int(node),
                report_source=report_source,
                metadata={
                    "report_due_day": due_day,
                    "report_delay_days": delay,
                },
            )

    def _mark_detected(
        self,
        *,
        t: int,
        nodes: np.ndarray,
        stage_code: np.int8,
        detection_source: str,
        report_delay_days: int,
    ) -> np.ndarray:
        nodes = np.unique(
            np.asarray(nodes, dtype=np.int32)
        )

        if nodes.size == 0:
            return np.empty(0, dtype=np.int32)

        eligible = nodes[
            self.case_status[nodes] == 0
        ]

        if eligible.size == 0:
            return np.empty(0, dtype=np.int32)

        self.case_status[eligible] = 1

        stage_name = {
            int(STAGE_UNKNOWN): "unknown",
            int(STAGE_PRE): "exposed",
            int(STAGE_INF): "infectious",
        }.get(int(stage_code), "unknown")

        for node in eligible:
            self._emit(
                t=t,
                event_type="detection",
                node_id=int(node),
                detection_source=detection_source,
                report_source=detection_source,
                disease_stage=stage_name,
            )

        self._queue_report(
            detection_day=t,
            nodes=eligible,
            stage_code=stage_code,
            report_source=detection_source,
            report_delay_days=report_delay_days,
        )

        return eligible

    def _deliver_due_reports(
        self,
        t: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        due_days = [
            day
            for day in self._report_queue.keys()
            if day <= t
        ]

        records = []
        for day in sorted(due_days):
            records.extend(
                self._report_queue.pop(day)
            )

        if not records:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int8),
                np.empty(0, dtype=object),
                np.empty(0, dtype=np.int32),
            )

        nodes = np.concatenate(
            [record[0] for record in records]
        ).astype(np.int32, copy=False)
        stages = np.concatenate(
            [record[1] for record in records]
        ).astype(np.int8, copy=False)
        sources = np.concatenate(
            [record[2] for record in records]
        ).astype(object, copy=False)
        detection_days = np.concatenate(
            [record[3] for record in records]
        ).astype(np.int32, copy=False)

        # Defensive deduplication.
        _, first_indices = np.unique(
            nodes,
            return_index=True,
        )
        first_indices.sort()

        nodes = nodes[first_indices]
        stages = stages[first_indices]
        sources = sources[first_indices]
        detection_days = detection_days[first_indices]

        self.case_status[nodes] = 2

        for node, source, detection_day in zip(
            nodes,
            sources,
            detection_days,
        ):
            self._emit(
                t=t,
                event_type="report_received",
                node_id=int(node),
                report_source=str(source),
                metadata={
                    "detection_day": int(detection_day),
                    "detection_to_report_delay": (
                        int(t) - int(detection_day)
                    ),
                },
            )

        return (
            nodes,
            stages,
            sources,
            detection_days,
        )

    def _process_imported_ascertainment(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
    ) -> np.ndarray:
        if t < self.imported_detection_start_day:
            return np.empty(0, dtype=np.int32)

        probability = float(
            np.clip(
                self.p_detect_imported_daily,
                0.0,
                1.0,
            )
        )

        if probability <= 0.0:
            return np.empty(0, dtype=np.int32)

        imported_ids = np.asarray(
            epi_state.get(
                "imported_ids",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )
        infectious_ids = np.sort(
            np.asarray(
                epi_state.get(
                    "inf_ids",
                    np.empty(0, np.int32),
                ),
                dtype=np.int32,
            )
        )

        if (
            imported_ids.size == 0
            or infectious_ids.size == 0
        ):
            return np.empty(0, dtype=np.int32)

        currently_infectious = self._isin_sorted(
            imported_ids,
            infectious_ids,
        )
        candidates = imported_ids[
            currently_infectious
        ]
        candidates = candidates[
            self.case_status[candidates] == 0
        ]

        if candidates.size == 0:
            return np.empty(0, dtype=np.int32)

        seed = int(
            derive_seed_from_base(
                self.seed,
                "imported_ascertainment",
            )
        )
        draws = u01_for_nodes(
            seed,
            int(t),
            candidates,
            np.int8(10),
        )

        detected = candidates[draws < probability]

        return self._mark_detected(
            t=t,
            nodes=detected,
            stage_code=STAGE_INF,
            detection_source="imported_ascertainment",
            report_delay_days=(
                self.passive_report_delay_days
            ),
        )

    def _process_daily_passive_detection(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        infectious_ids = np.asarray(
            epi_state.get(
                "inf_ids",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )

        if infectious_ids.size == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=object),
            )

        candidates = infectious_ids[
            self.case_status[infectious_ids] == 0
        ]

        if candidates.size == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=object),
            )

        symptomatic_ids = np.sort(
            np.asarray(
                epi_state.get(
                    "symptomatic_ids",
                    np.empty(0, np.int32),
                ),
                dtype=np.int32,
            )
        )
        is_symptomatic = self._isin_sorted(
            candidates,
            symptomatic_ids,
        )

        probabilities = np.where(
            is_symptomatic,
            float(
                self.p_detect_symptomatic_daily
            ),
            float(
                self.p_detect_asymptomatic_daily
            ),
        )
        probabilities = np.clip(
            probabilities,
            0.0,
            1.0,
        )

        seed = int(
            derive_seed_from_base(
                self.seed,
                "daily_passive_detection",
            )
        )
        draws = u01_for_nodes(
            seed,
            int(t),
            candidates,
            np.int8(11),
        )

        detected_mask = draws < probabilities
        detected_nodes = candidates[detected_mask]
        detected_symptomatic = is_symptomatic[
            detected_mask
        ]

        if detected_nodes.size == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=object),
            )

        all_detected = []
        all_sources = []

        symptomatic_detected = detected_nodes[
            detected_symptomatic
        ]
        if symptomatic_detected.size:
            accepted = self._mark_detected(
                t=t,
                nodes=symptomatic_detected,
                stage_code=STAGE_INF,
                detection_source="passive_symptomatic",
                report_delay_days=(
                    self.passive_report_delay_days
                ),
            )
            all_detected.append(accepted)
            all_sources.extend(
                ["passive_symptomatic"]
                * accepted.size
            )

        asymptomatic_detected = detected_nodes[
            ~detected_symptomatic
        ]
        if asymptomatic_detected.size:
            accepted = self._mark_detected(
                t=t,
                nodes=asymptomatic_detected,
                stage_code=STAGE_INF,
                detection_source="passive_asymptomatic",
                report_delay_days=(
                    self.passive_report_delay_days
                ),
            )
            all_detected.append(accepted)
            all_sources.extend(
                ["passive_asymptomatic"]
                * accepted.size
            )

        if not all_detected:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=object),
            )

        return (
            np.concatenate(all_detected).astype(
                np.int32,
                copy=False,
            ),
            np.asarray(
                all_sources,
                dtype=object,
            ),
        )

    def order_test(
        self,
        *,
        t: int,
        nodes: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        params = dict(params or {})
        delay = max(
            1,
            int(params.get("delay_days", 1)),
        )
        params["delay_days"] = delay

        due_day = int(t + delay)
        nodes = np.unique(
            np.asarray(nodes, dtype=np.int32)
        )

        if nodes.size == 0:
            return

        self._test_orders[due_day].append(
            (nodes.copy(), params)
        )

        for node in nodes:
            self._emit(
                t=t,
                event_type="test_ordered",
                node_id=int(node),
                metadata={
                    "test_due_day": due_day,
                    "turnaround_days": delay,
                },
            )

    def _process_due_tests(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        due_days = [
            day
            for day in self._test_orders.keys()
            if day <= t
        ]

        orders = []
        for day in sorted(due_days):
            orders.extend(
                self._test_orders.pop(day)
            )

        if not orders:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=bool),
                np.empty(0, dtype=np.int32),
            )

        exposed_ids = np.sort(
            np.asarray(
                epi_state.get(
                    "pre_ids",
                    np.empty(0, np.int32),
                ),
                dtype=np.int32,
            )
        )
        infectious_ids = np.sort(
            np.asarray(
                epi_state.get(
                    "inf_ids",
                    np.empty(0, np.int32),
                ),
                dtype=np.int32,
            )
        )

        completed_parts = []
        result_parts = []
        detected_parts = []

        for nodes, params in orders:
            nodes = np.unique(
                np.asarray(nodes, dtype=np.int32)
            )
            if nodes.size == 0:
                continue

            completed_parts.append(nodes.copy())

            is_exposed = self._isin_sorted(
                nodes,
                exposed_ids,
            )
            is_infectious = self._isin_sorted(
                nodes,
                infectious_ids,
            )

            sensitivity_exposed = float(
                params.get(
                    "sensitivity_exposed",
                    0.50,
                )
            )
            sensitivity_infectious = float(
                params.get(
                    "sensitivity_infectious",
                    0.99,
                )
            )
            specificity = float(
                np.clip(
                    params.get(
                        "specificity",
                        1.0,
                    ),
                    0.0,
                    1.0,
                )
            )

            positive_probability = np.full(
                nodes.size,
                1.0 - specificity,
                dtype=np.float64,
            )
            positive_probability[is_exposed] = (
                sensitivity_exposed
            )
            positive_probability[is_infectious] = (
                sensitivity_infectious
            )

            seed = int(
                derive_seed_from_base(
                    self.seed,
                    "active_test",
                    t,
                    params.get("tag", ""),
                )
            )
            draws = u01_for_nodes(
                seed,
                int(t),
                nodes,
                np.int8(12),
            )
            positive = (
                draws < positive_probability
            )
            result_parts.append(positive.copy())

            for node, result in zip(
                nodes,
                positive,
            ):
                self._emit(
                    t=t,
                    event_type="test_completed",
                    node_id=int(node),
                    result=(
                        "positive"
                        if bool(result)
                        else "negative"
                    ),
                )

            positive_nodes = nodes[positive]
            if positive_nodes.size == 0:
                continue

            positive_exposed = is_exposed[positive]
            positive_infectious = (
                is_infectious[positive]
            )

            report_delay = max(
                0,
                int(
                    params.get(
                        "report_delay_days",
                        0,
                    )
                ),
            )

            exposed_positive_nodes = positive_nodes[
                positive_exposed
            ]
            infectious_positive_nodes = positive_nodes[
                positive_infectious
            ]
            unknown_positive_nodes = positive_nodes[
                (~positive_exposed)
                & (~positive_infectious)
            ]

            for group, stage in [
                (
                    exposed_positive_nodes,
                    STAGE_PRE,
                ),
                (
                    infectious_positive_nodes,
                    STAGE_INF,
                ),
                (
                    unknown_positive_nodes,
                    STAGE_UNKNOWN,
                ),
            ]:
                accepted = self._mark_detected(
                    t=t,
                    nodes=group,
                    stage_code=stage,
                    detection_source="active_test",
                    report_delay_days=report_delay,
                )
                if accepted.size:
                    detected_parts.append(accepted)

        completed = (
            np.concatenate(completed_parts)
            if completed_parts
            else np.empty(0, np.int32)
        ).astype(np.int32, copy=False)

        results = (
            np.concatenate(result_parts)
            if result_parts
            else np.empty(0, dtype=bool)
        ).astype(bool, copy=False)

        newly_detected = (
            np.concatenate(detected_parts)
            if detected_parts
            else np.empty(0, np.int32)
        ).astype(np.int32, copy=False)

        return completed, results, newly_detected

    def order_trace(
        self,
        *,
        t: int,
        cases: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        params = dict(params or {})
        delay = max(
            1,
            int(params.get("delay_days", 1)),
        )
        params["delay_days"] = delay

        due_day = int(t + delay)
        cases = np.unique(
            np.asarray(cases, dtype=np.int32)
        )

        if cases.size == 0:
            return

        self._trace_orders[due_day].append(
            (cases.copy(), params)
        )

    def _process_due_traces(
        self,
        t: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        due_days = [
            day
            for day in self._trace_orders.keys()
            if day <= t
        ]

        orders = []
        for day in sorted(due_days):
            orders.extend(
                self._trace_orders.pop(day)
            )

        if not orders:
            return (
                np.empty(0, np.int32),
                np.empty(0, np.int32),
                np.empty(0, np.int16),
                np.empty(0, np.int32),
            )

        source_parts = []
        target_parts = []
        contact_type_parts = []
        completed_cases_parts = []

        for cases, params in orders:
            cases = np.unique(
                np.asarray(cases, dtype=np.int32)
            )
            if cases.size == 0:
                continue

            completed_cases_parts.append(
                cases.copy()
            )

            recall_probability = float(
                np.clip(
                    params.get(
                        "recall_prob",
                        0.25,
                    ),
                    0.0,
                    1.0,
                )
            )
            maximum_per_case = int(
                params.get(
                    "max_per_case",
                    25,
                )
            )

            raw_contact_types = params.get(
                "contact_types",
                ["hh", "sch", "wp"],
            )
            allowed_type_ids = {
                self._ctid(contact_type)
                for contact_type
                in raw_contact_types
                if self._ctid(contact_type) >= 0
            }

            for source in cases.tolist():
                neighbors = self.neighbor_map.get(
                    int(source),
                    [],
                )
                if not neighbors:
                    continue

                target_list = []
                type_id_list = []

                for target, _weight, contact_type in neighbors:
                    type_id = self._ctid(
                        contact_type
                    )

                    if type_id < 0:
                        continue

                    if (
                        allowed_type_ids
                        and type_id
                        not in allowed_type_ids
                    ):
                        continue

                    target_list.append(
                        int(target)
                    )
                    type_id_list.append(
                        type_id
                    )

                if not target_list:
                    continue

                targets = np.asarray(
                    target_list,
                    dtype=np.int32,
                )
                type_ids = np.asarray(
                    type_id_list,
                    dtype=np.int16,
                )

                recall_seed = int(
                    derive_seed_from_base(
                        self.seed,
                        "trace_recall",
                        t,
                        source,
                    )
                )
                recall_draws = u01_for_nodes(
                    recall_seed,
                    int(t),
                    targets,
                    np.int8(13),
                )
                recalled = (
                    recall_draws
                    < recall_probability
                )

                targets = targets[recalled]
                type_ids = type_ids[recalled]

                if targets.size == 0:
                    continue

                if (
                    maximum_per_case > 0
                    and targets.size
                    > maximum_per_case
                ):
                    rank_seed = int(
                        derive_seed_from_base(
                            self.seed,
                            "trace_rank",
                            t,
                            source,
                        )
                    )
                    ranks = u01_for_nodes(
                        rank_seed,
                        int(t),
                        targets,
                        np.int8(14),
                    )
                    order = np.argsort(ranks)[
                        :maximum_per_case
                    ]
                    targets = targets[order]
                    type_ids = type_ids[order]

                source_parts.append(
                    np.full(
                        targets.size,
                        int(source),
                        dtype=np.int32,
                    )
                )
                target_parts.append(
                    targets.astype(
                        np.int32,
                        copy=False,
                    )
                )
                contact_type_parts.append(
                    type_ids.astype(
                        np.int16,
                        copy=False,
                    )
                )

        completed_cases = (
            np.unique(
                np.concatenate(
                    completed_cases_parts
                )
            )
            if completed_cases_parts
            else np.empty(0, np.int32)
        ).astype(np.int32, copy=False)

        if not source_parts:
            return (
                np.empty(0, np.int32),
                np.empty(0, np.int32),
                np.empty(0, np.int16),
                completed_cases,
            )

        return (
            np.concatenate(source_parts).astype(
                np.int32,
                copy=False,
            ),
            np.concatenate(target_parts).astype(
                np.int32,
                copy=False,
            ),
            np.concatenate(
                contact_type_parts
            ).astype(np.int16, copy=False),
            completed_cases,
        )

    def step(
        self,
        *,
        t: int,
        epi_state: Dict[str, np.ndarray],
        scheduled_actions: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, np.ndarray]:
        t = int(t)

        if scheduled_actions:
            if "trace_cases" in scheduled_actions:
                self.order_trace(
                    t=t,
                    cases=scheduled_actions[
                        "trace_cases"
                    ],
                    params=scheduled_actions.get(
                        "trace_params"
                    ),
                )

            if "test_nodes" in scheduled_actions:
                self.order_test(
                    t=t,
                    nodes=scheduled_actions[
                        "test_nodes"
                    ],
                    params=scheduled_actions.get(
                        "test_params"
                    ),
                )

        detected_nodes_parts = []
        detection_sources_parts = []

        # Additional imported-case ascertainment occurs first.
        imported_detected = (
            self._process_imported_ascertainment(
                t=t,
                epi_state=epi_state,
            )
        )
        if imported_detected.size:
            detected_nodes_parts.append(
                imported_detected
            )
            detection_sources_parts.append(
                np.full(
                    imported_detected.size,
                    "imported_ascertainment",
                    dtype=object,
                )
            )

        # All remaining infectious and undetected people are eligible for
        # symptom-dependent passive ascertainment.
        passive_detected, passive_sources = (
            self._process_daily_passive_detection(
                t=t,
                epi_state=epi_state,
            )
        )
        if passive_detected.size:
            detected_nodes_parts.append(
                passive_detected
            )
            detection_sources_parts.append(
                passive_sources
            )

        tests_completed, test_results, test_detected = (
            self._process_due_tests(
                t=t,
                epi_state=epi_state,
            )
        )
        if test_detected.size:
            detected_nodes_parts.append(
                test_detected
            )
            detection_sources_parts.append(
                np.full(
                    test_detected.size,
                    "active_test",
                    dtype=object,
                )
            )

        (
            trace_src,
            trace_tgt,
            trace_ct,
            traces_completed,
        ) = self._process_due_traces(t)

        (
            reported_nodes,
            reported_stage,
            reported_sources,
            detection_days,
        ) = self._deliver_due_reports(t)

        detected_cases = (
            np.concatenate(detected_nodes_parts)
            if detected_nodes_parts
            else np.empty(0, np.int32)
        ).astype(np.int32, copy=False)

        detection_sources = (
            np.concatenate(
                detection_sources_parts
            )
            if detection_sources_parts
            else np.empty(0, dtype=object)
        ).astype(object, copy=False)

        batch = {
            "t": np.int32(t),
            "detected_cases": detected_cases,
            "detection_sources": detection_sources,
            "reported_cases": reported_nodes,
            "report_time": np.full(
                reported_nodes.size,
                t,
                dtype=np.int32,
            ),
            "detection_day": detection_days,
            "reported_stage": reported_stage,
            "age": (
                self.ages[reported_nodes]
                if reported_nodes.size
                else np.empty(0, np.int32)
            ),
            "is_vax": (
                self.is_vax[reported_nodes]
                if reported_nodes.size
                else np.empty(0, dtype=bool)
            ),
            "reported_sources": reported_sources,
            "tests_completed": tests_completed,
            "test_results_positive": test_results,
            "traces_completed": traces_completed,
            "trace_src": trace_src,
            "trace_tgt": trace_tgt,
            "trace_ct": trace_ct,
        }

        return batch

    # Helpers to assess surveillance state
    def has_pending_reports(self) -> bool:
        return any(
            bool(records)
            for records in self._report_queue.values()
        )


    def has_pending_tests(self) -> bool:
        return any(
            bool(orders)
            for orders in self._test_orders.values()
        )


    def has_pending_traces(self) -> bool:
        return any(
            bool(orders)
            for orders in self._trace_orders.values()
        )

    def reset_for_run(
        self,
        *,
        seed: Optional[int] = None,
        is_vax: Optional[np.ndarray] = None,
    ) -> None:
        if seed is not None:
            self.seed = int(seed)

        if is_vax is not None:
            self.is_vax = np.asarray(
                is_vax,
                dtype=np.bool_,
            )

        self.case_status.fill(0)
        self._report_queue.clear()
        self._test_orders.clear()
        self._trace_orders.clear()

    @staticmethod
    def _isin_sorted(
        values: np.ndarray,
        sorted_reference: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values)
        sorted_reference = np.asarray(
            sorted_reference
        )

        if (
            values.size == 0
            or sorted_reference.size == 0
        ):
            return np.zeros(
                values.size,
                dtype=bool,
            )

        indices = np.searchsorted(
            sorted_reference,
            values,
        )
        clipped = np.clip(
            indices,
            0,
            sorted_reference.size - 1,
        )
        return (
            sorted_reference[clipped]
            == values
        )

    def _ctid(self, contact_type: Any) -> int:
        if isinstance(
            contact_type,
            (int, np.integer),
        ):
            return int(contact_type)

        return int(
            self.ct_to_id.get(
                str(contact_type),
                -1,
            )
        )