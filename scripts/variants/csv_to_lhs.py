#scripts/variants/csv_to_lhs.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any, List
from pathlib import Path

from scripts.config import ModelConfig


def csv_to_LHS(
    csv_path: str, 
    n_samples: int, 
    output_dir: Optional[str] = None,
    seed: Optional[int] = None) -> pd.DataFrame:
    """
    Reads a CSV containing a column for:
         Parameters (named after ModelConfig, ex) epi.base_transmission_prob)
         Minimum (Minimum value to sample from)
         Maximum (Maximum value to sample from)
         Integer (binary whether samples should be coerced to int)

    If output_dir is provided, saves LHS to output_dir

    Returns:
    samples_df: pandas.DataFrame with a row for each sample and a column for each parameter
    """

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    N = int(n_samples)

    #Read in csv, normalize column names, and check
    df_raw = pd.read_csv(p)
    df_raw.columns = df_raw.columns.str.lower()

    expected_columns = ["parameter", "minimum", "maximum", "integer"]
    actual_columns = list(df_raw.columns)
    if not (actual_columns == expected_columns):
        raise ValueError(f"Error in csv for LHS, expected columns: '{expected_columns}', got: '{actual_columns}'")
    else:
        df = df_raw

    #Handle parameter names
    df["parameter"] = df["parameter"].astype(str).str.strip()
    if df["parameter"].duplicated().any():
        duped = df["parameter"][df["parameter"].duplicated()].unique().tolist()
        raise ValueError(f"Duplicate Parameters listed for LHS: {duped}")
    
    mins = pd.to_numeric(df["minimum"]).astype(float).to_numpy()
    maxs = pd.to_numeric(df["maximum"]).astype(float).to_numpy()

    if np.any(mins > maxs):
        bad_ind = np.where(mins > maxs)[0]
        bad_param = df["parameter"].iloc[bad_ind].tolist()
        raise ValueError(f"Minimum > Maximum for parameters {bad_param}")

    params = df["parameter"].tolist()

    #Check that parameters actually exist in modelconfig
    cfg_dict = ModelConfig().to_dict()
    missing = [p for p in params if not _dotted_key_exists(cfg_dict, p)]
    if missing:
        raise ValueError(f"The following parameters are not present in ModelConfig: {missing}")

    integers = pd.to_numeric(df["integer"]).fillna(0).to_numpy()
    integer_flags = (integers != 0)


    ##### Now do sampling

    rng = np.random.default_rng(int(seed) if seed is not None else 2026)

    sampled = {}
    param_types: Dict[str, str] = {}

    for j, name in enumerate(params):
        lo = float(mins[j])
        hi = float(maxs[j])
        is_int = bool(integer_flags[j])

        if is_int:
            lo_i = int(np.ceil(lo))
            hi_i = int(np.floor(hi))
        
            m = hi_i - lo_i + 1
            if m == 1:
                vals = np.full(N, lo_i, dtype = np.int64)
            else:
                perm = rng.permutation(N)
                jitter = rng.random(N)
                u = (perm + jitter) / N # in [0, 1)
                ind = np.floor(u*m).astype(int) #0 through m-1
                vals = (lo_i + ind).astype(np.int64)
        else:
            if hi == lo:
                vals = np.full(N, float(lo), dtype = np.float64)
            else:
                perm = rng.permutation(N)
                jitter = rng.random(N)
                u = (perm + jitter) / N
                vals = (lo + u * (hi - lo)).astype(np.float64)

        sampled[name] = vals
        param_types[name] = "int" if is_int else "float"

    lhs_df = pd.DataFrame({name: sampled[name] for name in params})

    for name in params:
        if param_types[name] == "int":
            lhs_df[name] = lhs_df[name].astype(np.int64)
        else:
            lhs_df[name] = lhs_df[name].astype(np.float64)

    #Save LHS to output_dir if appropriate
    if output_dir is not None:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok = True)

        lhs_df.to_csv(str(out / "LHS.csv"), index=False)
        
    return lhs_df



def LHS_to_cfg(
    lhs_df: pd.DataFrame,
    base_cfg: Optional[ModelConfig] = None
) -> List[ModelConfig]:
    """
    Takes a dataframe of LHS samples , and a base Model Configuration (base_cfg), and creates variants of the base configuration with assigned LHS values

    Args: 
        lhs_df: pandas dataframe produced by csv_to_LHS
        base_cfg: ModelConfig as base to vary from (defaults to EL_CONFIG)


    -Assumes lhs_df columns are dotted paths to ModelConfig (epi.vax_uptake)


    """

    if base_cfg is None:
        base_cfg = ModelConfig()

    if not isinstance(lhs_df, pd.DataFrame):
        raise TypeError("lhs_df must be a pandas DataFrame")

    base_dict = base_cfg.to_dict()

    for col in lhs_df.columns:
        dotted = str(col).strip()
        if not dotted:
            raise ValueError("Empty column name in lhs_df")
        if not _dotted_key_exists(base_dict, dotted):
            raise KeyError(f"Parameter '{dotted}' not found in base ModelConfig")

    #Set up overrides to be used in copy_with
    configs: List[ModelConfig] = []
    for _, row in lhs_df.iterrows():
        overrides: Dict[str, Dict[str, Any]] = {}
        for col in lhs_df.columns:
            dotted = str(col).strip()
            val = row[col]
            _set_nested(overrides, dotted, val)
        cfg_new = base_cfg.copy_with(overrides = overrides)
        configs.append(cfg_new)

    return configs



def _set_nested(overrides: Dict[str, Dict[str, Any]], dotted: str, value: Any):
    """Add value to override at a nested path from dotted """
    parts = dotted.split(".")
    if len(parts) == 1:
        # top-level replacement (rare for your config)
        overrides[parts[0]] = value
        return
    top = parts[0]
    cur = overrides.setdefault(top, {})
    for p in parts[1:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value




def _dotted_key_exists(cfg_dict: dict, dotted: str) -> bool:
    """
    Helper function to check if the dotted key exists in model config data structure
    """
    node = cfg_dict
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True

def csv_to_cfg(
    csv_path: str, 
    N: int, 
    output_dir: Optional[str, None],
    default_config: Optional[ModelConfig],
    seed: Optional[int] = None) -> List[ModelConfig]:

    lhs_df = csv_to_LHS(
        csv_path = csv_path,
        n_samples = N,
        output_dir = output_dir,
        seed = seed)

    ModelConfigList = LHS_to_cfg(lhs_df = lhs_df, base_cfg = default_config)

    return ModelConfigList

