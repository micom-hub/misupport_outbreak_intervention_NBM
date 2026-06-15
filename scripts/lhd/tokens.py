#scripts/lhd/tokens.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
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