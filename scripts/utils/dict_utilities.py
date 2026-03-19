#scripts/utils/dict_utilities.py

def freeze_for_key(x):
    """Convert nested dict/list/set structures into hashable tuples."""
    if x is None:
        return None
    if isinstance(x, dict):
        return tuple((k, freeze_for_key(v)) for k, v in sorted(x.items(), key=lambda kv: str(kv[0])))
    if isinstance(x, (list, tuple)):
        return tuple(freeze_for_key(v) for v in x)
    if isinstance(x, set):
        return tuple(sorted((freeze_for_key(v) for v in x), key=str))
    # numpy scalars -> python scalars
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    return x  # assumes hashable (int/float/str/bool/etc.)

def params_key(params):
    return freeze_for_key(params or {})