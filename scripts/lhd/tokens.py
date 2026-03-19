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