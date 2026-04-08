#scripts/PostRunProcessing/PRCC.py
"""
Utils for converting a results CSV file into the format expected by lhsPrccFromCsv.m to conduct 

Expected inputs corresponding to CSV
TP - timepoints
R - number of runs (parameter sets times stochastic reps)
P - number of varied parameters or initial conditions
O - number of data columns (model outputs) excluding run number and timestep
"""
from __future__ import annotations
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

#Functions to reformat aggregated results files into expected format for matlab PRCC analysis

_STAT_COLS_RE = re.compile(r"stat-cols-([0-9]+)")




#Main function to convert run results and LHS files to .csv expected by matlab
def export_prcc_inputs_for_matlab(
    results_path: Union[str, Path, pd.DataFrame],
    lhs_path: Union[str, Path, pd.DataFrame],
    *,
    kind: str,  # "summary" or "timeseries"
    out_dir: Optional[Union[str, Path]] = None,
    policy_col: Optional[str] = None,
    output_cols: Optional[Sequence[str]] = None, # for summary
    timeseries_output_name: str = "output", # for timeseries (single output column)
    aggregate_replicates: bool = False,
    replicate_agg: str = "median",  # "mean" or "median"
    dropna: bool = True,
    file_prefix: str = "prcc-input",
    exp_id_base: int = 1,     # MATLAB-friendly indexing (1-based)
    validate_written: bool = True,
) -> Dict[str, str]:
    """
    Writes an appropriately formatted results csv for use by lhsPrccFromCsv.m

    Vital Inputs:
    - results_path: Either a filepath to a results file, or the pandas df
    - lhs_path: Either a filepath to the LHS that generated the results, or df
    - kind: Specify result type as either "summary" or "timeseries" 

    Inputs that default appropriately:
    - out_dir: Where to write files, defaults to directory of results
    - policy_col: name of policy column, defaults to autodetect
    - output_cols: which output columns to include (if None, infers)
    - timeseries_output_name: name of output for timeseries export
    - aggregate_replicates: if true, aggregates stochastic replicates within (model_index, policy)
    - replicate_agg: (mean or median if aggregating)
    - file_prefix: str = "prcc-input"
    - validate_written: run validator on written files


    """
    #Normalize input
    kind = str(kind).strip().lower()
    if kind not in ("summary", "timeseries"):
        raise ValueError("kind must be 'summary' or 'timeseries'")

    results = _read_table(results_path)
    lhs = _ensure_model_index(_read_table(lhs_path))

    pol_col = _detect_policy_col(results, policy_col)

    if "model_index"not in results.columns:
        raise ValueError("results must include model_index column")

    #Identify output directory
    if out_dir is None:
        if isinstance(results_path, (str, Path)):
            out_dir = Path(results_path).expanduser().resolve().parent
        else:
            out_dir = Path.cwd()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents = True, exist_ok=True)

    #Identify parameter columns from LHS
    param_cols = _numeric_param_cols(
        lhs, 
        drop_cols=("model_index",), 
        drop_constant=True)
    if not param_cols:
        raise ValueError("No numeric (non-constant) parameter columns found in LHS.csv")


    #identify aggregation function
    if replicate_agg == "mean":
        agg_func = "mean"
    elif replicate_agg == "median":
        agg_func = "median"
    else:
        raise ValueError("replicate_agg must be 'mean' or 'median'")

    
    out_files: Dict[str, str] = {}

    #Split results by policy for individual PRCCs

    for policy_value, df_pol in results.groupby(pol_col, sort=False):
        safe_policy = _safe_name(policy_value)
        df_pol = df_pol.copy()

        #Handle run summary outputs
        if kind == "summary":
            #infer output cols as numeric/boolean cols, ignoring known cols
            if output_cols is None:
                ignore = {"model_index", "run_number", pol_col, "success", "error", "error_trace"}
                cand = []
                for c in df_pol.columns:
                    if c in ignore:
                        continue
                    if pd.api.types.is_bool_dtype(df_pol[c]) or pd.api.types.is_numeric_dtype(df_pol[c]):
                        cand.append(c)
                output_cols_use = cand
            else:
                output_cols_use = list(output_cols)
            
            if not output_cols_use:
                raise ValueError(f"No output columns selected for summary policy={policy_value}")
            
            #Aggregate if requested
            if aggregate_replicates and "run_number" in df_pol.columns:
                df_pol = df_pol.groupby("model_index", as_index=False)[output_cols_use].agg(agg_func)
            else:
                pass

            #Single timestep for summaries
            df_pol["timestep"] = 0

            #adjust row indices for matlab
            df_pol["experiment"] = df_pol["model_index"].astype(np.int64) + int(exp_id_base)

            #build the final dataframe by merging parameters 
            merged = df_pol.merge(lhs[["model_index"] + param_cols], on="model_index", how="inner", validate="many_to_one")
            final = merged[["experiment", "timestep"] + output_cols_use + param_cols].copy()

        #Handle timeseries data
        else: 
            tcols = _time_cols(df_pol)
            if not tcols:
                raise ValueError(f"No time columns t_* found for timeseries policy={policy_value}")

            #aggregate if requested
            warnings.warn("Aggregation requested for timeseries data, verify that this is intended", UserWarning)
            if aggregate_replicates and "run_number" in df_pol.columns:
                df_pol = df_pol.groupby("model_index", as_index=False)[tcols].agg(agg_func)

            #convert data to long as expected by matlab script

            long = df_pol.melt(id_vars=["model_index"], value_vars=tcols, var_name="_tcol", value_name=timeseries_output_name)
            long["timestep"] = long["_tcol"].str.split("_", n=1, expand=True)[1].astype(np.int32)
            long.drop(columns=["_tcol"], inplace=True)

            #re-index and add parameters
            long["experiment"] = long["model_index"].astype(np.int64) + int(exp_id_base)

            merged = long.merge(lhs[["model_index"] + param_cols], on="model_index", how="inner", validate="many_to_one")

            final = merged[["experiment", "timestep", timeseries_output_name] + param_cols].copy()

        #Ensure data is numeric-only for MATLAB csv reader
        for c in final.columns:
            if pd.api.types.is_bool_dtype(final[c]):
                final[c] = final[c].astype(np.int32)
            elif not pd.api.types.is_numeric_dtype(final[c]):
                final[c] = pd.to_numeric(final[c], errors="raise")

        #drop NAs if requested
        if dropna:
            final = final.dropna(axis = 0, how = "any")

        #sort columns into the order expected by matlab script
        final = final.sort_values(["experiment", "timestep"], kind="mergesort").reset_index(drop=True)

        #create filename that contains the number of stat cols
        
        if kind == "summary":
            O = len(output_cols_use) #output for each statistic
        else:
            O = 1 #prev/incidence as only output

        numDataVar = 2 + O #ensure an appropriate number of columns
        if numDataVar < 3:
            raise ValueError("PRCC requires numDataVar >= 3 (needs at least one output column)")
        out_name = f"{file_prefix}-{safe_policy}-stat-cols-{numDataVar}.csv"
        out_path = out_dir / out_name
        final.to_csv(out_path, index = False)

        if validate_written:
            validate_prcc_csv_for_matlab(out_path, kind = kind)

        out_files[str(policy_value)] = str(out_path)

    return out_files



#Validator function to ensure everything is appropriate from prcc outfile
def validate_prcc_csv_for_matlab(
    csv_path: Union[str, Path],
    *,
    kind: Optional[str] = None,  # "summary" or "timeseries" (optional)
    require_stat_cols_in_name: bool = True,
    require_sorted: bool = True,
    require_complete_time_grid: bool = True,
) -> Dict[str, Any]:
    """
    Validate a PRCC CSV for compatibility with lhsPrccFromCsv.m.
    Returns a dict of inferred info (numDataVar, O, P, R, TP, etc.)
    Raises ValueError if invalid.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    m = _STAT_COLS_RE.search(path.name)
    if require_stat_cols_in_name and not m:
        raise ValueError(f"Filename must contain 'stat-cols-<N>' (got {path.name})")
    numDataVar = int(m.group(1)) if m else None

    df = pd.read_csv(path)
    if df.shape[0] == 0:
        raise ValueError("CSV has no data rows")

    # check numeric
    try:
        arr = df.to_numpy(dtype=float)
    except Exception as exc:
        raise ValueError(f"CSV contains non-numeric values (Matlab csvread will fail): {exc}") from exc

    numCols = df.shape[1]
    if numDataVar is not None:
        if numDataVar > numCols:
            raise ValueError(f"stat-cols-{numDataVar} exceeds actual column count {numCols}")
        if numDataVar < 3:
            raise ValueError("stat-cols value must be >=3 (2 id cols + >=1 output col)")

        O = numDataVar - 2
        P = numCols - numDataVar
        if O < 1:
            raise ValueError("No model output columns (O < 1)")
        if P < 1:
            raise ValueError("No parameter columns (P < 1)")
    else:
        O = None
        P = None

    exp = df.iloc[:, 0].to_numpy(dtype=np.int64)
    t = df.iloc[:, 1].to_numpy(dtype=np.int64)

    # check sorted
    if require_sorted:
        if not np.all((exp[1:] > exp[:-1]) | ((exp[1:] == exp[:-1]) & (t[1:] >= t[:-1]))):
            raise ValueError("Rows are not sorted by (experiment, timestep). MATLAB reformatData will be wrong.")

    timepoints = np.unique(t)
    TP = int(timepoints.size)
    runs = np.unique(exp)
    R = int(runs.size)

    # check time grid size
    if require_complete_time_grid and TP > 1:
        tp_set = set(timepoints.tolist())
        for e in runs:
            tt = t[exp == e]
            if set(np.unique(tt).tolist()) != tp_set:
                raise ValueError(f"Experiment {e} does not have a complete set of timepoints")

    # check params per timestep are the same on timeseries
    if TP > 1 and numDataVar is not None:
        param_block = df.iloc[:, numDataVar:].to_numpy(dtype=float)
        for e in runs:
            idx = np.where(exp == e)[0]
            if idx.size <= 1:
                continue
            ref = param_block[idx[0], :]
            if not np.allclose(param_block[idx, :], ref[None, :], equal_nan=True):
                raise ValueError(f"Parameters vary across timesteps for experiment {e}")

    # check kind is appropriate
    if kind is not None:
        kind = str(kind).lower()
        if kind == "summary" and TP != 1:
            raise ValueError(f"kind='summary' but TP={TP} unique timepoints found")
        if kind == "timeseries" and TP < 2:
            raise ValueError("kind='timeseries' but <2 unique timepoints found")

    return {
        "path": str(path),
        "numCols": int(numCols),
        "numDataVar": int(numDataVar) if numDataVar is not None else None,
        "O": int(O) if O is not None else None,
        "P": int(P) if P is not None else None,
        "R": int(R),
        "TP": int(TP),
        "timepoints": timepoints.tolist(),
    }



#Helper to handle finding the file or df object, and return a pandas df
def _read_table(path_or_df: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df.copy()
    path = Path(path_or_df).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (expected .parquet or .csv)")


#Helpers to check for appropriate columns and dtypes
def _ensure_model_index(lhs: pd.DataFrame) -> pd.DataFrame:
    lhs = lhs.copy()
    if "model_index" not in lhs.columns:
        lhs.insert(0, "model_index", np.arange(lhs.shape[0], dtype=np.int32))
    return lhs

def _numeric_param_cols(lhs: pd.DataFrame, *, drop_cols=("model_index",), drop_constant=True) -> List[str]:
    cols = []
    for c in lhs.columns:
        if c in drop_cols:
            continue
        if pd.api.types.is_bool_dtype(lhs[c]) or pd.api.types.is_numeric_dtype(lhs[c]):
            cols.append(c)
    if drop_constant:
        keep = []
        for c in cols:
            # drop constant columns (PRCC will break / be meaningless)
            if lhs[c].nunique(dropna=False) > 1:
                keep.append(c)
        cols = keep
    return cols

def _detect_policy_col(df: pd.DataFrame, policy_col: Optional[str]) -> str:
    if policy_col and policy_col in df.columns:
        return policy_col
    for cand in ("variant_name", "policy_name", "variant", "policy"):
        if cand in df.columns:
            return cand
    raise ValueError("Could not infer policy column. Provide policy_col explicitly.")

def _time_cols(df: pd.DataFrame) -> List[str]:
    # t_0, t_1, ...
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith("t_")]
    def key(x):
        try:
            return int(x.split("_", 1)[1])
        except Exception:
            return 10**18
    return sorted(cols, key=key)

def _safe_name(x: Any) -> str:
    s = str(x)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")
    return s or "policy"



