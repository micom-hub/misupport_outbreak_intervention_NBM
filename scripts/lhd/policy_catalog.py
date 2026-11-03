from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


POLICIES: Dict[str, Dict[str, Any]] = {
    "observe_only": {
        "description": (
            "Surveillance and reporting occur, but the LHD creates no "
            "operational response tasks."
        ),
        "activate_on_first_report": False,
        "investigate_cases": False,
        "isolate_cases": False,
        "trace_cases": False,
        "notify_contacts": False,
        "test_contacts": False,
        "automated_notification_in_phase2": False,
        "immediate_escalation": False,
        "adaptive_escalation": False,
        "perfect_information": False,
    },

    "case_isolation_only": {
        "description": (
            "Investigate reported cases and provide isolation support. "
            "Do not elicit, notify, or test contacts."
        ),
        "activate_on_first_report": True,
        "investigate_cases": True,
        "isolate_cases": True,
        "trace_cases": False,
        "notify_contacts": False,
        "test_contacts": False,
        "automated_notification_in_phase2": False,
        "immediate_escalation": False,
        "adaptive_escalation": False,
        "perfect_information": False,
    },

    "trace_centered_response": {
        "description": (
            "Investigate reported cases, isolate cases, elicit and notify "
            "contacts, and refer notified contacts for testing."
        ),
        "activate_on_first_report": True,
        "investigate_cases": True,
        "isolate_cases": True,
        "trace_cases": True,
        "notify_contacts": True,
        "test_contacts": True,
        "automated_notification_in_phase2": False,
        "immediate_escalation": False,
        "adaptive_escalation": False,
        "perfect_information": False,
    },

    "immediate_scalable_mitigation": {
        "description": (
            "At the first LHD-received report, order scalable community "
            "mitigation while continuing case investigation and isolation. "
            "Direct exhaustive tracing is replaced by scalable notification."
        ),
        "activate_on_first_report": True,
        "investigate_cases": True,
        "isolate_cases": True,
        "trace_cases": False,
        "notify_contacts": False,
        "test_contacts": False,
        "automated_notification_in_phase2": True,
        "immediate_escalation": True,
        "adaptive_escalation": False,
        "perfect_information": False,
    },

    "adaptive_escalation": {
        "description": (
            "Begin with trace-centered response. Escalate to scalable "
            "community mitigation when an LHD-observable trigger is met."
        ),
        "activate_on_first_report": True,
        "investigate_cases": True,
        "isolate_cases": True,
        "trace_cases": True,
        "notify_contacts": True,
        "test_contacts": True,
        "automated_notification_in_phase2": True,
        "immediate_escalation": False,
        "adaptive_escalation": True,
        "perfect_information": False,
    },

    "perfect_information": {
        "description": (
            "Upper-bound benchmark with complete knowledge of the true "
            "epidemic state and true contact graph, unlimited operational "
            "capacity, and immediate targeted isolation of every exposed or "
            "infectious person before each day's transmission."
        ),
        "activate_on_first_report": False,
        "investigate_cases": False,
        "isolate_cases": False,
        "trace_cases": False,
        "notify_contacts": False,
        "test_contacts": False,
        "automated_notification_in_phase2": False,
        "immediate_escalation": False,
        "adaptive_escalation": False,
        "perfect_information": True,
    },
}


def get_policy_spec(policy_name: str) -> Dict[str, Any]:
    name = str(policy_name)

    if name not in POLICIES:
        raise ValueError(
            f"Unknown policy '{name}'. "
            f"Available policies: {sorted(POLICIES)}"
        )

    return deepcopy(POLICIES[name])