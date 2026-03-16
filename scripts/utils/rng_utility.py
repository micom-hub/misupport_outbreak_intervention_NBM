import hashlib
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
