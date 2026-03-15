import os
import zipfile
import pandas as pd
import numpy as np
from itertools import combinations
from numba import njit
from scripts.config import ModelConfig
from line_profiler import profile

#@profile
def synthetic_data_process(county, save_files=True):
    print("Processing FRED synthetic population data...")
    """
    Processes FRED synthetic population zip files, cleans and merges demographic/contact/location info.
    Returns contacts and locations pandas DataFrames.
    
    Args:
        county (str): Name of the county (must match zip filename and directory).
        save_files (bool): If True, saves contacts and locations DataFrames as .parquet files.

    Returns:
        contacts_pruned (pd.DataFrame): Individual-level demographic and contact info.
        locations (pd.DataFrame): Location-level info (households, schools, workplaces, group quarters).
    """
    # Data Directory and Extraction 
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            "Attempted to handle data before downloading FRED synthpop sets."
        )

    zip_path = os.path.join(data_dir, f"{county}.zip")
    county_dat = os.path.join(data_dir, county)

    if os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        # Optionally, remove the zip file with os.remove(zip_path)
    elif not os.path.isdir(county_dat):
        raise FileNotFoundError(
            f"Data file for {county} County not found at {data_dir}"
        )

    # Load all .txt Files as DataFrames ----
    dfs = {}
    for filename in os.listdir(county_dat):
        # Only load txt files; skip hospitals, metadata, or parquet files
        if (
            filename not in ["hospitals.txt", "METADATA.txt"]
            and not filename.endswith(".parquet")
            and filename.endswith(".txt")
        ):
            table_name = filename.replace(".txt", "")
            filepath = os.path.join(county_dat, filename)
            dfs[table_name] = pd.read_csv(filepath, sep="\t")

    people = dfs.get("people")
    gq_people = dfs.get("gq_people")

    #Handle Demographics and Contact Info ----
    contacts = people.rename(
        columns={
            "sp_id": "PID",
            "sp_hh_id": "hh_id",
            "school_id": "sch_id",
            "work_id": "wp_id",
        }
    ).copy()

    contacts["gq_id"] = np.nan
    contacts["sch_id"] = contacts["sch_id"].replace("X", np.nan)
    contacts["wp_id"] = contacts["wp_id"].replace("X", np.nan)
    contacts["hh_id"] = contacts["hh_id"].astype(str)
    contacts["PID"] = contacts["PID"].astype(str)
    contacts["gq"] = False

    # Add group quarters individuals
    gq_contacts = gq_people.rename(
        columns={
            "sp_id": "PID",
            "sp_gq_id": "gq_id",
        }
    ).copy()
    gq_contacts["gq"] = True
    gq_contacts["PID"] = gq_contacts["PID"].astype(str)
    # Add missing fields with NaN for consistency
    for col in ["hh_id", "sch_id", "wp_id", "relate", "race"]:
        gq_contacts[col] = np.nan
    # Concatenate
    contacts = pd.concat([gq_contacts, contacts], ignore_index=True)

    # ---- List PIDs for each location type 
    def members_grouped(df, colname, label):
        valid = df[~df[colname].isna()]
        mems = valid.groupby(colname)["PID"].apply(list).reset_index()
        mems.columns = [colname, f"{label}_members"]
        return mems

    sch_members = members_grouped(contacts, "sch_id", "sch")
    wp_members = members_grouped(contacts, "wp_id", "wp")
    gq_members = members_grouped(contacts[contacts["gq"] == True], "gq_id", "gq")  # noqa: E712
    hh_members = members_grouped(contacts, "hh_id", "hh")

    contacts = contacts.merge(hh_members, on="hh_id", how="left")
    contacts = contacts.merge(gq_members, on="gq_id", how="left")
    contacts = contacts.merge(wp_members, on="wp_id", how="left")
    contacts = contacts.merge(sch_members, on="sch_id", how="left")

    # build a locations df
    def build_location_df(members_df, id_col, mem_col, type_name):
        out = pd.DataFrame({
            "id": members_df[id_col].astype(str),
            "members": members_df[mem_col],
            "type": type_name,
        })
        out["size"] = out["members"].apply(len)
        return out

    school_locations = build_location_df(sch_members, "sch_id", "sch_members", "School")
    work_locations   = build_location_df(wp_members, "wp_id", "wp_members", "Workplace")
    gq_locations     = build_location_df(gq_members, "gq_id", "gq_members", "Group Quarters")
    hh_locations     = build_location_df(hh_members, "hh_id", "hh_members", "Household")

    locations = pd.concat(
        [school_locations, work_locations, gq_locations, hh_locations],
        ignore_index=True,
    )

    # ---- pare down and save to parquet
    columns_interest = [
        "PID", "hh_id", "wp_id", "sch_id", "gq_id", "age", "sex", "race", "relate", "gq"
    ]
    contacts_pruned = contacts[columns_interest].copy()

    print("Data Wrangled!")

    if save_files:
        print("Saving to parquet")
        contacts_pruned.to_parquet(os.path.join(county_dat, "contacts.parquet"))
        locations.to_parquet(os.path.join(county_dat, "locations.parquet"))
        print(f"Contact and Location Files Saved to {county_dat}")

    return contacts_pruned #, locations ## Right now not using locations df, could be useful later

#@profile
def build_individual_lookup(contact_df):
    """
    Returns a Dataframe that is indexed from 0-(N-1) to be referred to by the network model to correspond with the NxN matrix

        Args: contact_df, an integrated pandas dataframe of FRED synthetic population data (produced by SyntheticDataProcess)
    """
    lookup_df = contact_df.reset_index(drop = True)
    lookup_df = lookup_df[["age", "race", "sex"]]

    return lookup_df

@profile
def build_edge_list(
    contacts_df: pd.DataFrame,
    config: ModelConfig,
    rng: np.random.Generator = None,
    save: bool = False, 
    county: str = None,
    master_casual_contacts: int = 100
):
    """
    Generate a master edge list of all possible contact combinations, to be sampled from in graph construction

    Args:
        contacts_df (pd.DataFrame): Output from synthetic_data_process
        config: a ModelConfig object
        rng (np.rng): a seeded rng for reproducibility
        save (Bool): if true, will save edge list to parquet
        county (str): if data is to be saved, direct to proper data file

    Returns:
        edges_df (pd.DataFrame): DataFrame with ['source', 'target','weight]
    """

    print("Building edge list...")
    #Re-index contacts_df
    contacts_df = contacts_df.reset_index(drop = True)

    cfg = config


    if rng is None:
        rng = np.random.default_rng(int(cfg.sim.seed))

    edge_list = []


    #helper function to add an unordered pair to edge list, deduplicated
    def _add_pair(i,j,w,ct):
        if i == j:
            return
        s, t = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
        edge_list.append((s, t, float(w), ct))

    #add all hh, wp, sch, gq pairs
    hh_weight = cfg.population.hh_weight
    hh_groups = contacts_df.groupby('hh_id').groups
    for hh_id, ind in hh_groups.items():
        if hh_id == 'nan' or hh_id == '':
            continue
        ind_list = list(ind)
        if len(ind_list) < 2:
            continue
        for i, j in combinations(ind_list, 2):
            _add_pair(i, j, hh_weight, "hh")

    wp_weight = cfg.population.wp_weight
    wp_groups = contacts_df.groupby('wp_id').groups
    for wp_id, ind in wp_groups.items():
        if wp_id == 'nan' or wp_id == '':
            continue
        ind_list = list(ind)
        if len(ind_list) < 2:
            continue
        for i, j in combinations(ind_list, 2):
            _add_pair(i, j, wp_weight, "wp")

    sch_weight = cfg.population.sch_weight
    sch_groups = contacts_df.groupby('sch_id').groups
    for sch_id, ind in sch_groups.items():
        if sch_id == 'nan' or sch_id == '':
            continue
        ind_list = list(ind)
        if len(ind_list) < 2:
            continue
        for i, j in combinations(ind_list, 2):
            _add_pair(i, j, sch_weight, "sch")

    gq_weight = cfg.population.gq_weight
    gq_groups = contacts_df.groupby('gq_id').groups
    for gq_id, ind in gq_groups.items():
        if gq_id == 'nan' or gq_id == '':
            continue
        ind_list = list(ind)
        if len(ind_list) < 2:
            continue
        for i, j in combinations(ind_list, 2):
            _add_pair(i, j, gq_weight, "gq")

    #Casual contacts - sampled up to master_casual_contacts
    cas_weight = cfg.population.cas_weight
    non_gq_mask = ~contacts_df["gq"].astype(bool).to_numpy()
    non_gq_indices = contacts_df.index[non_gq_mask].to_numpy()
    N_non_gq = non_gq_indices.size

    #build a set of pairs not allowed (share hh/wp/sch)
    forbidden = set()
    for (i, j, w, ct) in edge_list:
        if i in non_gq_indices and j in non_gq_indices:
            forbidden.add((min(int(i), int(j)), max(int(i), int(j))))
        
    #sample casual candidates
    for i in non_gq_indices:
        candidates = []
        for j in non_gq_indices:
            if int(i) == int(j):
                continue
            a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            if (a,b) in forbidden:
                continue
            candidates.append(int(j))
        if not candidates:
            continue
        k = min(master_casual_contacts, len(candidates))
        sampled = rng.choice(candidates, size = k, replace = False)
        for j in sampled:
            _add_pair(i, j, cas_weight, "cas")

    #deduplicate any identical tuples (same s, t, ct)
    unique = {}
    for s, t, w, ct in edge_list:
        key = (int(s), int(t), str(ct))
        if key not in unique:
            unique[key] = (int(s), int(t), float(w), str(ct))
    rows = list(unique.values())
    edges_df = pd.DataFrame(rows, columns=['source','target','weight','contact_type'])

    #aggregate any (s,t) pair with multiple ct, keeping the higher weighted ct
    cts = ["hh", "sch", "wp", "gq", "cas"]
    ct_ws = [hh_weight, sch_weight, wp_weight, gq_weight, cas_weight]
    combined = sorted(zip(ct_ws, cts), reverse = True)
    prioritized_cts = [x[1] for x in combined]
    priority_map = {ct: i for i, ct in enumerate(prioritized_cts)}

    #correct datatypes
    

#sort by weight, then drop duplicates and keep only the first 
    edges_df = edges_df.sort_values(
        by = ['source', 'target', 'weight', 'contact_type'],
        ascending = [True, True, False, True],
        kind = 'mergesort'
    )
    edges_unique = edges_df.drop_duplicates(subset=['source','target'],keep='first').copy().reset_index(drop=True)


    edges_df = edges_unique

    edges_df['source'] = edges_df['source'].astype(np.int32)
    edges_df['target'] = edges_df['target'].astype(np.int32)
    edges_df['weight'] = edges_df['weight'].astype(np.float32)
    edges_df['contact_type'] = edges_df['contact_type'].astype(str)




        
    if save:
        if not county:
            raise Exception("Save requested but no county provided")
        save_loc = os.path.join(os.getcwd(), "data", county)
        os.makedirs(save_loc, exist_ok=True)
        out_file = os.path.join(save_loc, (cfg.sim.run_name + "_master_edgeList.parquet"))
        edges_df.to_parquet(out_file, index=False)
        print(f"Master network edge list saved to {out_file}")

    return edges_df



