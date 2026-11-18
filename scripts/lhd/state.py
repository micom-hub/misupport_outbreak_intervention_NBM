from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.lhd.tasks import TaskQueue


class LHDState:
    """
    Partial information and operational state available to the LHD.

    This class stores:
      - reported cases and known attributes;
      - known undirected contact edges;
      - active isolation state;
      - pending surveillance orders;
      - persistent operational tasks.
    """

    def __init__(self, N: int):
        self.N = int(N)

        self.last_t = -1
        self.new_cases_today = np.empty(
            0,
            dtype=np.int32,
        )
        self.new_edges_today = 0

        # Reported-case registry
        self.known_case = np.zeros(
            self.N,
            dtype=np.bool_,
        )
        self.known_case_list: List[int] = []

        self.case_detection_day = np.full(
            self.N,
            -1,
            dtype=np.int32,
        )
        self.case_report_time = np.full(
            self.N,
            -1,
            dtype=np.int32,
        )
        self.case_stage = np.zeros(
            self.N,
            dtype=np.int8,
        )
        self.case_age = np.full(
            self.N,
            -1,
            dtype=np.int32,
        )
        self.case_is_vax_known = np.zeros(
            self.N,
            dtype=np.bool_,
        )
        self.case_is_vax = np.zeros(
            self.N,
            dtype=np.bool_,
        )
        self.case_report_source = np.full(
            self.N,
            "",
            dtype=object,
        )

        # Undirected known-contact network
        self._edge_seen = set()
        self.known_edges: List[
            Tuple[int, int, int, int, str]
        ] = []
        self.known_adj: Dict[
            int,
            List[Tuple[int, int, int, str]],
        ] = defaultdict(list)

        # Intervention state
        self.isolated_until = np.full(
            self.N,
            -1,
            dtype=np.int32,
        )
        self.quarantined_until = np.full(
            self.N,
            -1,
            dtype=np.int32,
        )

        # Surveillance orders
        self.pending_tests: Dict[int, int] = {}
        self.pending_traces: Dict[int, int] = {}
        self.traced_nodes = set()

        # Persistent operational queue
        self.task_queue = TaskQueue(run_number=0)

    def process_batch(
        self,
        batch: Dict[str, np.ndarray],
    ) -> None:
        t = int(batch.get("t", -1))
        self.last_t = t
        self.new_cases_today = np.empty(
            0,
            dtype=np.int32,
        )
        self.new_edges_today = 0

        tests_completed = np.asarray(
            batch.get(
                "tests_completed",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )
        for node in tests_completed:
            self.pending_tests.pop(int(node), None)

        traces_completed = np.asarray(
            batch.get(
                "traces_completed",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )
        for node in traces_completed:
            node = int(node)
            self.pending_traces.pop(node, None)
            self.traced_nodes.add(node)

        cases = np.asarray(
            batch.get(
                "reported_cases",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )

        if cases.size:
            stages = np.asarray(
                batch.get(
                    "reported_stage",
                    np.zeros(cases.size, np.int8),
                ),
                dtype=np.int8,
            )
            report_times = np.asarray(
                batch.get(
                    "report_time",
                    np.full(cases.size, t, np.int32),
                ),
                dtype=np.int32,
            )
            detection_days = np.asarray(
                batch.get(
                    "detection_day",
                    np.full(cases.size, -1, np.int32),
                ),
                dtype=np.int32,
            )
            ages = np.asarray(
                batch.get(
                    "age",
                    np.full(cases.size, -1, np.int32),
                ),
                dtype=np.int32,
            )
            vaccination = np.asarray(
                batch.get(
                    "is_vax",
                    np.zeros(cases.size, np.bool_),
                ),
                dtype=np.bool_,
            )
            sources = np.asarray(
                batch.get(
                    "reported_sources",
                    np.full(cases.size, "", object),
                ),
                dtype=object,
            )

            _, first_indices = np.unique(
                cases,
                return_index=True,
            )
            first_indices.sort()

            cases = cases[first_indices]
            stages = stages[first_indices]
            report_times = report_times[first_indices]
            detection_days = detection_days[first_indices]
            ages = ages[first_indices]
            vaccination = vaccination[first_indices]
            sources = sources[first_indices]

            new_mask = ~self.known_case[cases]

            if new_mask.any():
                new_cases = cases[new_mask]
                self.new_cases_today = (
                    new_cases.astype(
                        np.int32,
                        copy=True,
                    )
                )

                self.known_case[new_cases] = True
                self.case_detection_day[new_cases] = (
                    detection_days[new_mask]
                )
                self.case_report_time[new_cases] = (
                    report_times[new_mask]
                )
                self.case_stage[new_cases] = (
                    stages[new_mask]
                )
                self.case_age[new_cases] = ages[new_mask]
                self.case_is_vax[new_cases] = (
                    vaccination[new_mask]
                )
                self.case_is_vax_known[new_cases] = True
                self.case_report_source[new_cases] = (
                    sources[new_mask]
                )
                self.known_case_list.extend(
                    new_cases.tolist()
                )

        self.new_edges_today = int(
            self._process_trace_edges(batch, t=t)
        )

    def _process_trace_edges(
        self,
        batch: Dict[str, np.ndarray],
        *,
        t: int,
    ) -> int:
        source = np.asarray(
            batch.get(
                "trace_src",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )
        target = np.asarray(
            batch.get(
                "trace_tgt",
                np.empty(0, np.int32),
            ),
            dtype=np.int32,
        )
        contact_type = np.asarray(
            batch.get(
                "trace_ct",
                np.empty(0, np.int16),
            ),
            dtype=np.int16,
        )

        n = min(
            source.size,
            target.size,
            contact_type.size,
        )
        added = 0

        for index in range(n):
            u = int(source[index])
            v = int(target[index])
            ct = int(contact_type[index])

            if (
                u == v
                or u < 0
                or v < 0
                or u >= self.N
                or v >= self.N
            ):
                continue

            a, b = (
                (u, v)
                if u < v
                else (v, u)
            )
            key = (a, b, ct)

            if key in self._edge_seen:
                continue

            self._edge_seen.add(key)
            self.known_edges.append(
                (a, b, ct, int(t), "trace")
            )
            self.known_adj[a].append(
                (b, ct, int(t), "trace")
            )
            self.known_adj[b].append(
                (a, ct, int(t), "trace")
            )
            added += 1

        return added

    def neighbors(
        self,
        node: int,
    ) -> List[Tuple[int, int, int, str]]:
        return self.known_adj.get(int(node), [])

    def mark_test_pending(
        self,
        node: int,
        due_day: int,
    ) -> None:
        self.pending_tests[int(node)] = int(due_day)

    def mark_trace_pending(
        self,
        node: int,
        due_day: int,
    ) -> None:
        self.pending_traces[int(node)] = int(due_day)

    def is_test_pending(self, node: int) -> bool:
        return int(node) in self.pending_tests

    def is_trace_pending(self, node: int) -> bool:
        return int(node) in self.pending_traces

    def has_been_traced(self, node: int) -> bool:
        return int(node) in self.traced_nodes

    def is_isolated(
        self,
        node: int,
        day: Optional[int] = None,
    ) -> bool:
        if day is None:
            day = self.last_t

        return (
            int(self.isolated_until[int(node)])
            > int(day)
        )

    def get_investigation_components(
        self,
    ) -> List[Dict[str, Any]]:
        if not self.known_edges:
            return []

        adjacency = defaultdict(set)
        nodes = set()

        for u, v, _ct, _day, _source in self.known_edges:
            adjacency[int(u)].add(int(v))
            adjacency[int(v)].add(int(u))
            nodes.add(int(u))
            nodes.add(int(v))

        components = []
        visited = set()

        for root in sorted(nodes):
            if root in visited:
                continue

            stack = [root]
            visited.add(root)
            members = []

            while stack:
                current = stack.pop()
                members.append(current)

                for neighbor in adjacency.get(
                    current,
                    set(),
                ):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            components.append(
                {
                    "index_case": min(members),
                    "members": sorted(members),
                }
            )

        return components

    def reset_for_run(
        self,
        run_number: int = 0,
    ) -> None:
        self.last_t = -1
        self.new_cases_today = np.empty(
            0,
            dtype=np.int32,
        )
        self.new_edges_today = 0

        self.known_case.fill(False)
        self.known_case_list.clear()
        self.case_detection_day.fill(-1)
        self.case_report_time.fill(-1)
        self.case_stage.fill(0)
        self.case_age.fill(-1)
        self.case_is_vax_known.fill(False)
        self.case_is_vax.fill(False)
        self.case_report_source.fill("")

        self._edge_seen.clear()
        self.known_edges.clear()
        self.known_adj.clear()

        self.isolated_until.fill(-1)
        self.quarantined_until.fill(-1)

        self.pending_tests.clear()
        self.pending_traces.clear()
        self.traced_nodes.clear()

        self.task_queue.reset(
            run_number=int(run_number)
        )