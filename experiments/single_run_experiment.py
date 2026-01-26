"""
one_run_experiment.py

Example experiment of a single run, mostly for testing
"""
from __future__ import annotations
import os
import json
import datetime 
import numpy as np # noqa: F401
import pandas as pd # noqa: F401
import matplotlib # noqa: F401
import matplotlib.pyplot as plt # noqa: F401

from scripts.model_driver import prepare_contacts, run_single_model, run_variants  # noqa: F401
from scripts.network_model import DefaultModelParams, EqualPriority, RandomPriority, PrioritizeElders, CallIndividualsAction  # noqa: F401

# --------------------------
#       INTRODUCTION
# --------------------------
#Brief Description of experiment
EXPERIMENT_NAME = "single_run"
DESCRIPTION = """
This is a test of a single run through the experiment workflow
"""

# --------------------------
#          METHODS
# --------------------------
#Set the base parameters to be used for runs
#Update BASE_PARAMS with values to be non-default
BASE_PARAMS = DefaultModelParams.copy()
BASE_PARAMS.update({
    "n_runs": 100,
    "base_transmission_prob": 1,
    "simulation_duration": 20,
    "seed": 2026,
    "record_exposure_events": True,
    "overwrite_edge_list": False,
    "save_data_files": True,
    "county": "Alcona",
    "I0": [50]
})

#Results root path for this experiment
EXPERIMENT_DIR = os.path.join("experiments", "results", EXPERIMENT_NAME)
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%dT%H%MEST")
RUN_DIR = os.path.join(EXPERIMENT_DIR, RUN_TIMESTAMP)
os.makedirs(RUN_DIR, exist_ok = True)

COUNTY = BASE_PARAMS.get("county")
STATE = BASE_PARAMS.get("state")

#Define experiment variants
#each variant gets a name, a dict of params to override with matching keys to base_params, a mapping of LHD action types to algorithms, a mapping of LHD action types to action factories, and a save_exposures option

def call_action_factory(nodes, contact_type, prio, cost, params = None):
    nodes_arr = np.asarray(nodes, dtype=np.int32)
    contact_types = [contact_type] if contact_type is not None else ['cas', 'sch', 'wp']

    reduction = (
        params.get('reduction', BASE_PARAMS.get('lhd_default_int_reduction', 0.8))
        if params else BASE_PARAMS.get('lhd_default_int_reduction', 0.8)
    )
    duration = int(
        params.get('duration', BASE_PARAMS.get('lhd_default_int_duration', 10))
        if params else BASE_PARAMS.get('lhd_default_int_duration', 10)
    )
    call_cost = float(cost) if cost is not None else float(BASE_PARAMS.get('lhd_default_call_duration', 0.1))

    return CallIndividualsAction(
        nodes=nodes_arr,
        contact_types=contact_types,
        reduction=float(reduction),
        duration=int(duration),
        call_cost=float(call_cost),
        min_factor=1e-6
        )

VARIANTS = [
    {
        "name": "equal_call",
        "param_overrides": {},
        "algorithm_map": {"call": EqualPriority()},
        "factory_map": {"call": call_action_factory},
        "save_exposures": True,
    }
]


#Save run configuration next to results
with open(os.path.join(RUN_DIR, "experiment_config.json"), "w") as f:
    json.dump({
        "name": EXPERIMENT_NAME,
        "description": DESCRIPTION,
        "base_params": BASE_PARAMS,
        "variants": VARIANTS,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat() 
    }, f, indent = 4, default = str)


# --------------------------
#         RUN MODEL
# --------------------------
#run model based on methods specifications

print(f"Preparing contacts for county: {COUNTY}")
contacts_df = prepare_contacts(COUNTY, STATE, data_dir = "data", save_files = BASE_PARAMS)

print(f"Running Experiment {EXPERIMENT_NAME}")
variant_results = run_variants(
    contacts_df = contacts_df,
    base_params = BASE_PARAMS,
    variants = VARIANTS,
    run_dir = RUN_DIR,
    base_seed = BASE_PARAMS.get("seed", 2026)
)

print("Run finished!")
# --------------------------
#          RESULTS
# --------------------------
#statistical analysis, plotting, creating results files, etc.
