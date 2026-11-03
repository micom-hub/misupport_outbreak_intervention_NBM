#scripts/lhd/tokens.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

"""
Dataclass for tracking actions that have been taken, and scheduling their reversal
"""
@dataclass(frozen=True)
class MultiplierToken:
    expires_at: int
    nodes: np.ndarray           
    contact_types: Tuple[str, ...]
    in_factor: float
    out_factor: float
    action: str = "multiplier"

    def revert(self, model) -> None:
        nodes = np.asarray(self.nodes, dtype=np.int32)
        if nodes.size == 0:
            return
        for ct in self.contact_types:
            if self.in_factor != 1.0:
                model.in_multiplier[ct][nodes] /= self.in_factor
            if self.out_factor != 1.0:
                model.out_multiplier[ct][nodes] /= self.out_factor

# Universal multiplers applied to a full contact type/factor
@dataclass(frozen=True)
class GlobalMultiplierToken:
    expires_at: int
    contact_factors: Dict[str, float]
    action: str = "community_mitigation"

    def revert(self, model) -> None:
        for contact_type, factor in self.contact_factors.items():
            if contact_type not in model.in_multiplier:
                continue

            safe_factor = max(float(factor), 1e-6)

            model.in_multiplier[contact_type] /= safe_factor
            model.out_multiplier[contact_type] /= safe_factor