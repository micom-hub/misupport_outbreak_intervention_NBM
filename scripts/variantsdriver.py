#scripts/variantsdriver.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import json
import os
import traceback
import concurrent.futures
import multiprocessing
import time

import numpy as np
import pandas as pd

from scripts.variants.csv_to_lhs import csv_to_cfg
from scripts.variants.runvariants import run_variants
from scripts.graph.graph_utils import (
    build_minimal_graphdata_from_edge_list,
    build_graph_data,
    sample_from_master_graphdata,
)
from scripts.driver import prepare_contacts, read_or_build_master
from scripts.config import ModelConfig, DEFAULT_MODEL_CONFIG
from scripts.lhd.lhdConfig import LhdConfig, validate_variant



#Pseudo Code


def prepare_run(
    csv_path: str,
    n_samples: int,
    output_dir: str,
    *,
    base_cfg: Optional[ModelConfig] = None,
    data_dir: str = "data",
    overwrite_files: bool = True,
    save_files: bool = True,
    seed: Optional[int] = None
) -> Tuple[List[ModelConfig], pd.DataFrame]:

    """
    Prepares all necessary data structures for an experiment
        -Ensures synthetic population is loaded and formatted
        - Loads/builds master edge list for experiment

    Args:
        - csv_path to a formatted csv for LHS sweep
        - base_config: a default model config object to be varied on

    Returns: (configs list, master_edge_df)
    """
    base_cfg = base_cfg or DEFAULT_MODEL_CONFIG

    #Create output directory
    out_base = Path(output_dir).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok = True)

    #1) Ensure contact data exists
    contacts_df = prepare_contacts(
        county = base_cfg.sim.county,
        state = base_cfg.sim.state,
        data_dir = data_dir,
        overwrite_files = overwrite_files,
        save_files = save_files
    )

    #2) Load or build master edge list
    if seed is None:
        seed = base_cfg.sim.seed
    base_seed = int(seed)
    rng_master = np.random.default_rng(int(base_seed))

    master_df = read_or_build_master(contacts_df = contacts_df, cfg = base_cfg, 
    run_dir = str(out_base), rng = rng_master, variant = True) 

    #3) Sample LHS Object 
    configs_list = csv_to_cfg(
        csv_path = csv_path, 
        N = n_samples, 
        output_dir = str(out_base),
        default_config = base_cfg,
        seed = base_seed
        )

    return (configs_list, master_df)


