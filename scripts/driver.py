"""
Driver to structure data, configure, and run NetworkModel objects

The script:
1) Ensures data for the desired county exists
2) Checks for a master edge list, or builds it and a GraphData object
3) Samples an edge list from master edge list per model object
4) Builds GraphData object and caches it
5) Instantiates NetworkModel with ModelConfig and GraphData objects
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
from line_profiler import profile


from scripts.config import ModelConfig
from scripts.graph.graph_utils import (
    build_minimal_graphdata_from_edge_list,
    build_graph_data,
    sample_from_master_graphdata,
)

from scripts.simulation.outbreak_model import NetworkModel
from scripts.utils.synth_data_processing import synthetic_data_process, build_edge_list
from scripts.utils.fred_fetch import downloadPopData


# Prepare and structure contact data


def prepare_contacts(
    county: str,
    state: str,
    data_dir: str = "data",
    overwrite_files: bool = True,
    save_files: bool = False,
):
    """
    Handles data retrieval for FRED synthetic population, processes data, and returns a dataframe of all contact data

    Always returns contacts_df or raises error
    """

    county_name = county.lower().capitalize()
    state_name = state.lower().capitalize()

    os.makedirs(data_dir, exist_ok=True)
    # find any entries in data_dir that start with county_name
    entries = [f for f in os.listdir(data_dir) if f.startswith(county_name)]
    if not entries:
        downloadPopData(state=state_name, county=county_name)
        # re-list
        entries = [f for f in os.listdir(data_dir) if f.startswith(county_name)]
        if not entries:
            raise RuntimeError(
                f"downloadPopData did not produce any data files for {county_name} in {data_dir}"
            )

    # prefer a directory if present, else take zip or folder
    dir_entries = [e for e in entries if os.path.isdir(os.path.join(data_dir, e))]
    if dir_entries:
        countyfoldername = dir_entries[0]
    else:
        countyfoldername = entries[0]

    countyfolder = os.path.join(data_dir, countyfoldername)

    # If data is a zipfile, extract data and delete zip
    if os.path.isfile(countyfolder) and countyfolder.lower().endswith(".zip"):
        try:
            contacts_df = synthetic_data_process(county)
        except Exception as e:
            raise RuntimeError(
                f"Failed to process synthetic data from zip for {county_name}: {e}"
            ) from e

        if save_files and os.path.exists(countyfolder):
            try:
                os.remove(countyfolder)
            except OSError:
                pass
        _validate_contacts_df(contacts_df)
        return contacts_df

    # check county folder for contacts.parquet, and read it. if doesn't exist, build
    if os.path.isdir(countyfolder):
        parquet_path = os.path.join(countyfolder, "contacts.parquet")
        if os.path.exists(parquet_path) and not overwrite_files:
            return pd.read_parquet(parquet_path)
        else:
            try:
                contacts_df = synthetic_data_process(county_name, save_files=save_files)
                _validate_contacts_df(contacts_df)
                return contacts_df
            except Exception as e:
                raise RuntimeError(
                    f"Could not find contacts.parquet in {countyfolder} and synthetic_data_process failed: {e}"
                ) from e

    raise RuntimeError(f"Unhandled data entry for county {county_name}: {countyfolder}")


# Read or build master edge list for county population
@profile
def read_or_build_master(
    contacts_df: pd.DataFrame,
    cfg: ModelConfig,
    run_dir: str,
    seed: Optional[int] = None,
    variant: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Trys to read in a master edge list if it exists in the run directory, otherwise builds one

    Saves config to a json

    """
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents = True, exist_ok = True)

    master_path = run_dir / "MasterEdgelist.parquet"
    config_path = run_dir / "ModelConfig.json"
    if not variant:
        try:
            cfg.to_json(str(config_path))
        except Exception:
            pass

    #If master already exists and we aren't overwriting, read in
    if master_path.exists() and not bool(cfg.sim.overwrite_master):
        return pd.read_parquet(str(master_path))

    master_df = build_edge_list(
        contacts_df=contacts_df,
        config=cfg,
        seed= cfg.sim.seed,
        save=False,
        county=cfg.sim.county,
        master_casual_contacts=int(cfg.sim.master_casual_candidates)
)
    if cfg.sim.save_master:
        try:
            master_df.to_parquet(str(master_path), index = False)
        except Exception:
            pass

    return master_df
        
# Single model run
def run_single_model(
    contacts_src: Union[pd.DataFrame, str],
    cfg: Union[ModelConfig, str, Path],
    output_dir: Optional[str] = None,
    *,
    seed: Optional[int] = None
    ) -> NetworkModel:
    """
    Driver function to run a single model
    
    Args:
        contacts_src: as a contacts_df object, a filepath, or county name
        cfg: ModelConfig object for run or absolute filepath to a ModelConfig.json saved by ModelConfig.to_json
        output_dir: base directory for model outputs, defaults to "model_runs"
        seed: optional seed to overwrite cfg.sim.seed
    """

    #If cfg is a filepath, build ModelConfig
    if isinstance(cfg, (str, Path)):
        cfg_path = Path(cfg).expanduser()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"ModelConfig JSON not found at {cfg_path}")
        cfg = ModelConfig.from_json(str(cfg_path))
    
    try:
        cfg.validate()
    except Exception:
        raise
    
    #Overwrite seed if one is provided 
    if seed is not None:
        cfg = cfg.copy_with({"sim": {"seed": int(seed)}})


    #Figure out what contacts_src is, and normalize
    if isinstance(contacts_src, pd.DataFrame):
        contacts_df = contacts_src.reset_index(drop=True)
    elif isinstance(contacts_src, str):
        #Try as a filepath
        if os.path.exists(contacts_src) and contacts_src.endswith(".parquet"):
            contacts_df = pd.read_parquet(contacts_src).reset_index(drop=True)
        else:
            #Try as a county
            contacts_df = prepare_contacts(
                contacts_src, 
                cfg.sim.state, 
                data_dir = "data", 
                overwrite_files = cfg.sim.overwrite_master,
                save_files = cfg.sim.save_data_files
            )
    else:
        raise TypeError("contacts_src must be a DataFrame or a path/county string")

    #Determine if output_dir is relative or absolute and save
    if output_dir is None:
        output_dir = "model_runs"
    base_output = Path(output_dir).expanduser()
    if not base_output.is_absolute():
        base_output = Path.cwd() / base_output
    run_name = cfg.sim.run_name
    run_dir = (base_output / run_name).resolve()

    if cfg.sim.save_data_files:
        run_dir.mkdir(parents = True, exist_ok = True)

    # Build or load master
    master_df = read_or_build_master(
        contacts_df,
        cfg, 
        run_dir = run_dir,
        seed = cfg.sim.seed
        )

    # Build minimal graphdata from master that can be sampled from 
    minimal_graphdata = build_minimal_graphdata_from_edge_list(master_df, N=int(contacts_df.shape[0]))



    sampled_edges_df = sample_from_master_graphdata(
        minimal_graphdata, 
        cfg, 
        seed = cfg.sim.seed
    )

    # Build full run GraphData (this is the heavy step)
    run_graphdata = build_graph_data(
        edge_list=sampled_edges_df, 
        contacts_df=contacts_df, 
        config = cfg, 
        seed = cfg.sim.seed, 
        N=int(contacts_df.shape[0])
    )

    # Instantiate NetworkModel with provided prebuilt GraphData
    model = NetworkModel(
        config = cfg,
        graphdata = run_graphdata, 
        run_dir = str(run_dir),
        seed = cfg.sim.seed,
    )

    # Run simulation
    model.simulate()

    return model

#Helper function to make sure contacts_df looks right
def _validate_contacts_df(df: pd.DataFrame) -> None:
    """
    Throws error if contacts_df doesn't have necessary columns
    """
    
    required = {"PID", "hh_id", "wp_id", "sch_id", "gq_id", "age", "sex", "gq"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Contacts DataFrame missing required columns: {missing}")


if __name__ == "__main__":
    cfg = ModelConfig().copy_with(
        {
            "sim": {"county": "Keweenaw", "display_plots": True ,
            "I0":3 }
        }
    )
    contacts = prepare_contacts(cfg.sim.county, cfg.sim.state, save_files = True)
    model = run_single_model(contacts, cfg, seed = 13 )
    model.results_to_df().to_csv("testingresults.csv", index=False)
    
    
