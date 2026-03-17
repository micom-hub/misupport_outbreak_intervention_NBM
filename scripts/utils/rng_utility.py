import hashlib
import numpy as np
from typing import Any

#Helper function for reproducibility
def derive_seed_from_base(base_seed: int, *tags: Any) -> int:
    #Deterministically get an integer seed from a base seed
    h = hashlib.sha256()
    h.update(str(int(base_seed)).encode("utf-8"))
    for t in tags:
        h.update(b"\x00")
        h.update(str(t).encode("utf-8"))

    return int.from_bytes(h.digest()[:4], "big")


#Pseudo-random work-arounds for keeping counter-factuals consistent

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
def _splitmix64(x: np.ndarray) -> np.ndarray:
    """
    Vectorized splitmix64 for pseudorandom number generation 
    """
    z = (x + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & _MASK64
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & _MASK64
    z = z ^ (z >> np.uint64(31))
    return z & _MASK64

def u01_for_nodes(seed: int, t: int, nodes: np.ndarray, stage: int) -> np.ndarray:
    """
    Deterministic U(0,1) for each node, keyed by (seed, t, node, stage).
    Returns float64 array in [0,1).
    """
    nodes_u = nodes.astype(np.uint64, copy=False)
    x = np.uint64(seed) ^ (np.uint64(t) * np.uint64(0xD2B74407B1CE6E93)) ^ (nodes_u * np.uint64(0xCA5A826395121157)) ^ (np.uint64(stage) * np.uint64(0x9E3779B97F4A7C15))
    z = _splitmix64(x)
    # Use top 53 bits to make a float in [0,1)
    return ((z >> np.uint64(11)).astype(np.float64)) * (1.0 / (2.0**53))