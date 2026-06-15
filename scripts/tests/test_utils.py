import json
import hashlib
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_array(arr: Any) -> str:
    a = np.asarray([]) if arr is None else np.asarray(arr)
    try:
        # include dtype and shape for safety
        payload = a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()
    except Exception:
        payload = repr(a).encode()
    return sha256_bytes(payload)


def hash_dataframe(df: Optional[pd.DataFrame]) -> str:
    if df is None:
        return sha256_bytes(b"__none__")
    try:
        # stable ordering: sort by columns then rows (best-effort)
        cols = list(df.columns)
        if cols:
            df2 = df.sort_values(by=cols).reset_index(drop=True)
            b = df2.to_csv(index=False).encode()
        else:
            b = df.to_csv(index=False).encode()
    except Exception:
        b = repr(df).encode()
    return sha256_bytes(b)


def fingerprint_graphdata(
    gd: Any,
) -> Dict[str, Any]:  # Using Any for GraphData to avoid circular import
    """
    Produce a small fingerprint dictionary for GraphData to detect mutation.
    """
    try:
        el_h = hash_dataframe(gd.edge_list)
    except Exception:
        el_h = sha256_bytes(repr(getattr(gd, "edge_list", None)).encode())
    # csr_by_type hashes
    csr_dict = {}
    try:
        for ct, triple in getattr(gd, "csr_by_type", {}).items():
            indptr, indices, weights = triple
            h = hashlib.sha256()
            try:
                h.update(np.asarray(indptr).astype(np.int64).tobytes())
                h.update(np.asarray(indices).astype(np.int32).tobytes())
                h.update(np.asarray(weights).astype(np.float32).tobytes())
            except Exception:
                h.update(repr((indptr, indices, weights)).encode())
            csr_dict[str(ct)] = h.hexdigest()
    except Exception:
        csr_dict = {"error": "csr fingerprint failed"}
    # neighbor_map fingerprint (structural)
    try:
        nm_items = []
        for src in sorted(getattr(gd, "neighbor_map", {}).keys()):
            nbrs = gd.neighbor_map.get(src, [])
            sorted_nbrs = sorted([(int(t), float(w), str(ct)) for (t, w, ct) in nbrs])
            nm_items.append((int(src), tuple(sorted_nbrs)))
        nm_b = repr(nm_items).encode()
        nm_h = sha256_bytes(nm_b)
    except Exception:
        nm_h = sha256_bytes(repr(getattr(gd, "neighbor_map", None)).encode())
    return {"edge_list_hash": el_h, "csr_by_type": csr_dict, "neighbor_map_hash": nm_h}


def canonicalize_rng_state(state: Any) -> Any:
    """Convert numpy arrays in RNG state to python lists for stable JSON output and comparison."""
    if state is None:
        return None
    if isinstance(state, dict):
        out = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (list, tuple)):
                out[k] = [
                    (
                        canonicalize_rng_state(x)
                        if isinstance(x, dict)
                        else (x.tolist() if isinstance(x, np.ndarray) else x)
                    )
                    for x in v
                ]
            elif isinstance(v, dict):
                out[k] = canonicalize_rng_state(v)
            else:
                out[k] = v
        return out
    return repr(state)


def normalize_state_lists(states: Any) -> Any:
    """Normalize a timestep [S,E,I,R] into lists where each inner list is sorted (so ordering differences don't break equality)."""
    if states is None:
        return None
    try:
        return [
            sorted(list(x)) if isinstance(x, (list, tuple, np.ndarray)) else x
            for x in states
        ]
    except Exception:
        try:
            return [sorted(list(x)) for x in states]
        except Exception:
            return states


def arrays_equal_sorted(a: Any, b: Any) -> bool:
    """Compare two arrays/lists treating them as unordered sets (sort each before compare)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.ndim == 1 and b_arr.ndim == 1:
        a_sorted = np.sort(a_arr)
        b_sorted = np.sort(b_arr)
        return np.array_equal(a_sorted, b_sorted)
    try:
        return np.array_equal(a_arr, b_arr)
    except Exception:
        return canonicalize(a) == canonicalize(b)


def canonicalize(x: Any) -> Any:
    """JSON-serializable canonicalization of common types used in compares."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [canonicalize(v) for v in x]
    if isinstance(x, dict):
        return {str(k): canonicalize(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return canonicalize(x.tolist())
    try:
        json.dumps(x)
        return x
    except Exception:
        return repr(x)


def dicts_equal(d0: Any, d1: Any) -> bool:
    """Compare two dictionaries after canonicalizing their contents."""
    return canonicalize(d0) == canonicalize(d1)
