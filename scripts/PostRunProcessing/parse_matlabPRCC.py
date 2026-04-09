from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import re
import itertools


import numpy as np
import pandas as pd


_MAT_STEM_RE = re.compile(r"^prcc-input-(?P<policy>.+)-stat-cols-(?P<numDataVar>\d+)$")


def parse_matlab(
    run_dir: Union[str, Path],
    *,
    structured_dir: str = "StructuredPRCCs",
    mat_glob: str = "**/*.mat",
    out_path: Optional[Union[str, Path]] = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Parse all PRCC .mat files under <run_dir>/<structured_dir>/ into a tidy long table
    and write prcc_results_long.parquet.

    Output schema (long):
      mat_path, kind, policy, numDataVar,
      time_index, timepoint,
      output, parameter,
      method_id, method,
      prcc, p_value, significant, alpha_used

    Methods correspond to your plotPRCC convention:
      1: uncorrected (p = uncorrectedSignificance, alpha_used = prccResult.alpha)
      2: ztest      (p = uncorrectedPccZtestSignificance, significant = uncorrectedPccZtest==1, alpha_used = 0.05)
      3: bonferroni (p = bonferroniSignificance, alpha_used = prccResult.alpha)
      4: bhfdr      (p = bhfdrSignificance, alpha_used = prccResult.alpha)
    """
    run_dir = Path(run_dir).expanduser().resolve()
    structured_root = run_dir / structured_dir
    if not structured_root.exists():
        raise FileNotFoundError(f"Structured PRCC directory not found: {structured_root}")

    mat_paths = sorted(structured_root.glob(mat_glob))
    if not mat_paths:
        raise FileNotFoundError(f"No .mat files found under: {structured_root} (glob={mat_glob})")

    frames: List[pd.DataFrame] = []

    for mp in mat_paths:
        # Only parse the files produced by your pipeline (optional but safer)
        stem = mp.stem
        m = _MAT_STEM_RE.match(stem)
        if m is None:
            if strict:
                raise ValueError(f"Unexpected .mat filename (expected prcc-input-...-stat-cols-N): {mp.name}")
            continue

        policy = m.group("policy")
        numDataVar = int(m.group("numDataVar"))

        kind = _infer_kind(mp, structured_dir_name=structured_dir)

        prcc = _load_prcc_mat(mp)

        df_one = _prccresult_to_long(
            prcc,
            mat_path=str(mp),
            kind=kind,
            policy=policy,
            numDataVar=numDataVar,
        )
        frames.append(df_one)

    if not frames:
        raise ValueError(f"Found .mat files under {structured_root}, but none matched expected naming pattern.")

    df = pd.concat(frames, ignore_index=True)

    if out_path is None:
        out_path = run_dir / "results" / "prcc_results_long.parquet"
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write parquet
    try:
        df.to_parquet(out_path, index=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write parquet to {out_path}. "
            f"Install a parquet engine (recommended: pip install pyarrow). Original error: {exc}"
        ) from exc

    return df


# ------------------------
# Internals
# ------------------------

def _infer_kind(mat_path: Path, *, structured_dir_name: str) -> str:
    """
    Infer kind like 'summary'/'incidence'/'prevalence' from:
      .../<run_dir>/StructuredPRCCs/<kind>/...
    """
    parts = mat_path.parts
    try:
        idx = parts.index(structured_dir_name)
        if idx + 1 < len(parts):
            return str(parts[idx + 1])
    except ValueError:
        pass
    return mat_path.parent.name


def _load_prcc_mat(path: Path) -> Dict[str, Any]:
    """
    Load prccResult struct from a MATLAB .mat file.

    Supports MATLAB v7 files via scipy.io.loadmat.
    If you ever save as v7.3 (HDF5), you'll need mat73/h5py (not implemented here).
    """
    try:
        from scipy.io import loadmat  # type: ignore
    except Exception as exc:
        raise ImportError("scipy is required to read .mat files (pip install scipy).") from exc

    d = loadmat(str(path), simplify_cells=True)
    if "prccResult" in d:
        prcc = d["prccResult"]
    else:
        # fallback: sometimes saved under a different variable name
        keys = [k for k in d.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No variables found in {path}")
        prcc = d[keys[0]]

    if not isinstance(prcc, dict):
        raise TypeError(f"Expected prccResult to be a dict after loadmat(simplify_cells=True); got {type(prcc)}")
    return prcc


def _prccresult_to_long(
    prcc: Dict[str, Any],
    *,
    mat_path: str,
    kind: str,
    policy: str,
    numDataVar: int,
) -> pd.DataFrame:
    # ---- 1) Infer expected TP/O/P from metadata FIRST ----
    tp_raw = np.asarray(prcc.get("analysisTimePoints", []), dtype=float).reshape(-1)
    if tp_raw.size == 0:
        raise ValueError(f"No analysisTimePoints found in {mat_path}; cannot infer TP.")
    TP_exp = int(tp_raw.size)

    outputs = np.array([_clean_name(s) for s in _to_str_list(prcc.get("modelOutputNames"))], dtype=object)
    params  = np.array([_clean_name(s) for s in _to_str_list(prcc.get("paramNames"))], dtype=object)

    if outputs.size == 0:
        # common for some MATLAB saves; treat as a single output
        outputs = np.array(["output"], dtype=object)
    if params.size == 0:
        raise ValueError(f"No paramNames found in {mat_path}; cannot infer P.")

    O_exp = int(outputs.size)
    P_exp = int(params.size)

    # ---- 2) Coerce all tensors to shape (TP, O, P) ----
    prcc_arr = _as_T_O_P(prcc["uncorrectedPrcc"], TP_exp, O_exp, P_exp,
                         name="uncorrectedPrcc", mat_path=mat_path)
    p_unc    = _as_T_O_P(prcc["uncorrectedSignificance"], TP_exp, O_exp, P_exp,
                         name="uncorrectedSignificance", mat_path=mat_path)
    p_z      = _as_T_O_P(prcc["uncorrectedPccZtestSignificance"], TP_exp, O_exp, P_exp,
                         name="uncorrectedPccZtestSignificance", mat_path=mat_path)
    p_bon    = _as_T_O_P(prcc["bonferroniSignificance"], TP_exp, O_exp, P_exp,
                         name="bonferroniSignificance", mat_path=mat_path)
    p_bhfdr  = _as_T_O_P(prcc["bhfdrSignificance"], TP_exp, O_exp, P_exp,
                         name="bhfdrSignificance", mat_path=mat_path)

    # z-test reject indicator: treat NaN as 0, then binarize -> int8
    z_rej = _as_T_O_P(prcc["uncorrectedPccZtest"], TP_exp, O_exp, P_exp,
                      name="uncorrectedPccZtest", mat_path=mat_path)
    z_rej = np.nan_to_num(z_rej, nan=0.0)
    z_rej = (z_rej > 0.5).astype(np.int8)

    TP, O, P = prcc_arr.shape  # should equal TP_exp/O_exp/P_exp

    # ---- 3) Safety checks on names ----
    if outputs.size != O:
        warnings.warn(
            f"modelOutputNames length ({outputs.size}) != O ({O}) in {mat_path}. Using generic output names.",
            RuntimeWarning,
        )
        outputs = np.array([f"output_{i}" for i in range(O)], dtype=object)

    if params.size != P:
        warnings.warn(
            f"paramNames length ({params.size}) != P ({P}) in {mat_path}. Using generic parameter names.",
            RuntimeWarning,
        )
        params = np.array([f"param_{i}" for i in range(P)], dtype=object)

    # alpha stored in prccResult (used for uncorrected/bonferroni/bhfdr)
    alpha_main = float(np.asarray(prcc.get("alpha", 0.05), dtype=float).reshape(-1)[0])

    # ---- 4) Vectorized long-table construction ----
    ti = np.repeat(np.arange(TP), O * P)
    oi = np.tile(np.repeat(np.arange(O), P), TP)
    pi = np.tile(np.arange(P), TP * O)

    base = pd.DataFrame({
        "mat_path": mat_path,
        "kind": kind,
        "policy": policy,
        "numDataVar": int(numDataVar),

        "time_index": ti.astype(np.int32),
        "timepoint": tp_raw[ti],

        "output": outputs[oi],
        "parameter": params[pi],

        "prcc": prcc_arr.reshape(-1),
        "p_uncorrected": p_unc.reshape(-1),
        "p_bonferroni": p_bon.reshape(-1),
        "p_bhfdr": p_bhfdr.reshape(-1),
        "ztest_reject": z_rej.reshape(-1).astype(np.int8),
        "p_ztest": p_z.reshape(-1),
        "alpha_main": alpha_main,
    })

    frames = []

    # 1) uncorrected
    df1 = base[["mat_path","kind","policy","numDataVar","time_index","timepoint","output","parameter","prcc"]].copy()
    df1["method_id"] = 1
    df1["method"] = "uncorrected"
    df1["p_value"] = base["p_uncorrected"]
    df1["alpha_used"] = alpha_main
    df1["significant"] = df1["p_value"] < alpha_main
    frames.append(df1)

    # 2) z-test
    df2 = base[["mat_path","kind","policy","numDataVar","time_index","timepoint","output","parameter","prcc"]].copy()
    df2["method_id"] = 2
    df2["method"] = "ztest"
    df2["p_value"] = base["p_ztest"]
    df2["alpha_used"] = 0.05
    df2["significant"] = base["ztest_reject"].astype(bool)
    frames.append(df2)

    # 3) bonferroni
    df3 = base[["mat_path","kind","policy","numDataVar","time_index","timepoint","output","parameter","prcc"]].copy()
    df3["method_id"] = 3
    df3["method"] = "bonferroni"
    df3["p_value"] = base["p_bonferroni"]
    df3["alpha_used"] = alpha_main
    df3["significant"] = df3["p_value"] < alpha_main
    frames.append(df3)

    # 4) bhfdr
    df4 = base[["mat_path","kind","policy","numDataVar","time_index","timepoint","output","parameter","prcc"]].copy()
    df4["method_id"] = 4
    df4["method"] = "bhfdr"
    df4["p_value"] = base["p_bhfdr"]
    df4["alpha_used"] = alpha_main
    df4["significant"] = df4["p_value"] < alpha_main
    frames.append(df4)

    return pd.concat(frames, ignore_index=True)

def _ensure_3d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3:
        return a
    if a.ndim == 2:
        # assume TP=1
        return a[None, :, :]
    if a.ndim == 1:
        # extremely degenerate; treat as TP=1,O=1,P=len
        return a[None, None, :]
    raise ValueError(f"Expected 1D/2D/3D array; got shape={a.shape}")


def _to_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return [str(_stringify_cell(e)) for e in x]

    arr = np.asarray(x)

    # Cell arrays typically become object arrays
    if arr.dtype == object:
        return [str(_stringify_cell(e)) for e in arr.reshape(-1)]

    # String arrays (dtype 'U'/'S') vs char arrays (1-char elements)
    if arr.dtype.kind in {"U", "S"}:
        flat = arr.reshape(-1)

        if flat.size == 0:
            return []

        # If every element is length 1, it's a char array -> join
        lengths = [len(str(v)) for v in flat.tolist()]
        if all(L == 1 for L in lengths):
            if arr.ndim == 2:
                # join each row into a string
                return ["".join(row.tolist()).strip() for row in arr]
            return ["".join(flat.tolist()).strip()]

        # Otherwise it's an array of full strings -> keep as list
        return [str(v) for v in flat.tolist()]

    # Fallback
    return [str(v) for v in arr.reshape(-1)]


def _stringify_cell(e: Any) -> str:
    if e is None:
        return ""
    if isinstance(e, str):
        return e
    arr = np.asarray(e)
    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.reshape(-1).tolist()).strip()
    if arr.dtype == object and arr.size == 1:
        return _stringify_cell(arr.reshape(-1)[0])
    return str(e)


def _clean_name(s: str) -> str:
    s = str(s).strip()
    # strip outer quotes/apostrophes like "'peakPrev'" or '"peakPrev"'
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        s = s[1:-1].strip()
    return s
def _as_T_O_P(a: Any, TP: int, O: int, P: int, *, name: str, mat_path: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)

    # Handle common squeeze cases (when O==1 or TP==1 or P==1)
    if arr.ndim == 2:
        if O == 1 and arr.shape == (TP, P):
            return arr[:, None, :]
        if O == 1 and arr.shape == (P, TP):
            return arr.T[:, None, :]
        if TP == 1 and arr.shape == (O, P):
            return arr[None, :, :]
        if TP == 1 and arr.shape == (P, O):
            return arr.T[None, :, :]
        if P == 1 and arr.shape == (TP, O):
            return arr[:, :, None]
        if P == 1 and arr.shape == (O, TP):
            return arr.T[:, :, None]
        raise ValueError(f"{name} in {mat_path} has 2D shape {arr.shape}, cannot coerce to ({TP},{O},{P}).")

    if arr.ndim == 3:
        # Try all axis permutations to match expected shape
        for perm in itertools.permutations((0, 1, 2)):
            t = arr.transpose(perm)
            if t.shape == (TP, O, P):
                return t
        raise ValueError(f"{name} in {mat_path} has 3D shape {arr.shape}, cannot permute to ({TP},{O},{P}).")

    if arr.ndim == 1:
        if TP == 1 and O == 1 and arr.size == P:
            return arr[None, None, :]
        if TP == 1 and P == 1 and arr.size == O:
            return arr[None, :, None]
        if O == 1 and P == 1 and arr.size == TP:
            return arr[:, None, None]
        raise ValueError(f"{name} in {mat_path} has 1D shape {arr.shape}, cannot coerce to ({TP},{O},{P}).")

    raise ValueError(f"{name} in {mat_path} has unexpected ndim={arr.ndim}, shape={arr.shape}.")



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Run directory containing StructuredPRCCs/")
    ap.add_argument("--out", default=None, help="Output parquet path (optional)")
    args = ap.parse_args()

    df = parse_matlab(args.run_dir, out_path=args.out)

    out_path = Path(args.out).expanduser().resolve() if args.out else (
        Path(args.run_dir).expanduser().resolve() / "results" / "prcc_results_long.parquet"
    )

    print(
        f"[parse_matlabPRCC] Success: wrote {len(df):,} row(s) to {out_path}",
        flush=True,
    )