from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "event_id",
    "run_number",
    "day",
    "event_type",
    "node_id",
    "source_node_id",
    "action_id",
    "task_id",
    "policy_name",
    "policy_phase",
    "infection_origin",
    "infection_generation",
    "disease_stage",
    "symptomatic",
    "detection_source",
    "report_source",
    "contact_type",
    "result",
    "value",
    "metadata_json",
]


class EventLedger:
    """
    Append-only event ledger for epidemiological, surveillance, and
    intervention events.

    The ledger is intended for auditing and post-run analysis. Core model
    state should continue to use efficient NumPy arrays.
    """

    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []
        self._next_event_id = 0

    def append(
        self,
        *,
        run_number: int,
        day: int,
        event_type: str,
        node_id: Optional[int] = None,
        source_node_id: Optional[int] = None,
        action_id: Optional[str] = None,
        task_id: Optional[str] = None,
        policy_name: Optional[str] = None,
        policy_phase: Optional[int] = None,
        infection_origin: Optional[str] = None,
        infection_generation: Optional[int] = None,
        disease_stage: Optional[str] = None,
        symptomatic: Optional[bool] = None,
        detection_source: Optional[str] = None,
        report_source: Optional[str] = None,
        contact_type: Optional[str] = None,
        result: Optional[str] = None,
        value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        event_id = int(self._next_event_id)
        self._next_event_id += 1

        if metadata is None:
            metadata_json = None
        else:
            metadata_json = json.dumps(
                self._json_safe(metadata),
                sort_keys=True,
            )

        row = {
            "event_id": event_id,
            "run_number": int(run_number),
            "day": int(day),
            "event_type": str(event_type),
            "node_id": self._optional_int(node_id),
            "source_node_id": self._optional_int(source_node_id),
            "action_id": action_id,
            "task_id": task_id,
            "policy_name": policy_name,
            "policy_phase": self._optional_int(policy_phase),
            "infection_origin": infection_origin,
            "infection_generation": self._optional_int(
                infection_generation
            ),
            "disease_stage": disease_stage,
            "symptomatic": (
                None if symptomatic is None else bool(symptomatic)
            ),
            "detection_source": detection_source,
            "report_source": report_source,
            "contact_type": contact_type,
            "result": result,
            "value": (
                None if value is None else float(value)
            ),
            "metadata_json": metadata_json,
        }

        self._rows.append(row)
        return event_id

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=EVENT_COLUMNS)

        df = pd.DataFrame(self._rows)

        for column in EVENT_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA

        df = df[EVENT_COLUMNS].copy()

        nullable_integer_columns = [
            "event_id",
            "run_number",
            "day",
            "node_id",
            "source_node_id",
            "policy_phase",
            "infection_generation",
        ]

        for column in nullable_integer_columns:
            df[column] = pd.array(df[column], dtype="Int64")

        df["symptomatic"] = pd.array(
            df["symptomatic"],
            dtype="boolean",
        )

        return df

    def clear(self) -> None:
        self._rows.clear()
        self._next_event_id = 0

    def __len__(self) -> int:
        return len(self._rows)

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        return int(value)

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            return [cls._json_safe(v) for v in value.tolist()]

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, dict):
            return {
                str(key): cls._json_safe(val)
                for key, val in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(v) for v in value]

        if isinstance(value, (str, int, float, bool)):
            return value

        return str(value)