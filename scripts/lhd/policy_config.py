#scripts/lhd/policy_config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

from scripts.lhd.policy_catalog import POLICIES as _POLICY_CATALOG



@dataclass
class PolicyVariant:
    """
    Policy variants detail how LHD's will function, and contain:
    - name (external name to print)
    -policy_name (internal name in policy catalog)
    - lhd_overrides: optional dict to change cfg.lhd params
    - description: optional
    """
    name: str
    policy_name: str
    lhd_overrides: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

@dataclass
class PolicyConfig:
    """
    Container for configurations to be used and compared in a model run
    """
    variants: List[PolicyVariant] = field(default_factory=list)

    def __post_init__(self):
        for v in self.variants:
            validate_variant(v)

    def add_variant(self, variant: PolicyVariant) -> None:
        validate_variant(variant)
        self.variants.append(variant)

    def get_variant(self, name: str) -> PolicyVariant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"Variant not found: {name}")


def validate_variant(v: PolicyVariant) -> None:
    if not isinstance(v.name, str) or not v.name.strip():
        raise ValueError("Variant.name must be a non-empty string")
    if not isinstance(v.policy_name, str) or not v.policy_name.strip():
        raise ValueError("Variant.policy_name must be a non-empty string")

    if _POLICY_CATALOG is not None and v.policy_name not in _POLICY_CATALOG:
        raise ValueError(f"Unknown policy_name '{v.policy_name}' (not in policy_catalog.POLICIES)")

    if not isinstance(v.lhd_overrides, dict):
        raise ValueError("Variant.lhd_overrides must be a dict")

#Example configuration
# Example configuration: edit this list to compare policies by name
POLICY_CONFIGURATION = PolicyConfig(variants=[
    PolicyVariant(name="observe_only", policy_name="observe_only"),
    PolicyVariant(name="isolate_only", policy_name="isolate_only"),
    PolicyVariant(name="trace_only", policy_name="trace_only"),
    PolicyVariant(name="trace_then_isolate", policy_name="trace_then_isolate"),
])