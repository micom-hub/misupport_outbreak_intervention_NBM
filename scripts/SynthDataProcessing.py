import os
import zipfile
import pandas as pd
import numpy as np

@profile
def synthetic_data_process(county="Chippewa", save_files=True):
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

@profile
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
    contacts_df: pd.df,
    params: dict,
    rng: np.rng = None,
    save: bool = False, 
    county: str = None
):
    """
    Take a contact dataframe output from synthetic_data_process, and produce an edge list with adjustible contact weights, sampling individual contacts from each location (school, workplace, group_quarters), and using all household contacts

    Args:
        contacts_df (pd.DataFrame): Output from synthetic_data_process
        params (dict): Parameters dictionary which contains parameters describing contact weights and number of contacts to sample from workplace and school
        rng (np.rng): a seeded rng for reproducibility
        save (Bool): if true, will save edge list to parquet
        county (str): if data is to be saved, direct to proper data file

    Returns:
        edges_df (pd.DataFrame): DataFrame with ['source', 'target','weight]
    """

    print("Building edge list...")
    #Re-index contacts_df
    contacts_df = contacts_df.reset_index(drop = True)

    if rng is None:
        rng = np.random.default_rng()

    edge_list = []

    #group lookup
    for col in ['hh_id', 'wp_id', 'sch_id', 'gq_id']:
        contacts_df[col] = contacts_df[col].astype(str)
    
    #Create edges between all individuals in a household
    household_groups = contacts_df.groupby('hh_id').groups

    for hh_id, indices in household_groups.items():
        if hh_id == 'nan' or hh_id == '':
            continue
        ind_list = list(indices)
        for i in range(len(ind_list)):
            for j in range(i+1, len(ind_list)):
                edge_list.append((ind_list[i], ind_list[j], params["hh_weight"], "hh"))
                edge_list.append((ind_list[j], ind_list[i], params["hh_weight"], "hh"))
        
    #Sample workplace contacts for each individual
    wp_groups = contacts_df.groupby('wp_id').groups
    for wp_id, indices in wp_groups.items():
        if wp_id == 'nan' or wp_id == '':
            continue
        ind_list = list(indices)
        for i in ind_list:
            contacts_to_sample = [ind for ind in ind_list if ind != i]
            k = min(params["wp_contacts"], len(contacts_to_sample))
            sampled = rng.choice(contacts_to_sample, size = k, replace = False) if k > 0 else []
            for j in sampled:
                edge_list.append((i, j, params['wp_weight'], "wp"))
                edge_list.append((j, i, params["wp_weight"], "wp"))

    #School sampled contacts
    sch_groups = contacts_df.groupby('sch_id').groups
    for sch_id, indices in sch_groups.items():
        if sch_id == 'nan' or sch_id == '':
            continue
        ind_list = list(indices)
        for i in ind_list:
            contacts_to_sample = [ind for ind in ind_list if ind != i]
            k = min(params["sch_contacts"], len(contacts_to_sample))
            sampled = rng.choice(contacts_to_sample, size = k, replace = False) 
            for j in sampled:
                edge_list.append((i, j, params['sch_weight'], "sch"))
                edge_list.append((j, i, params["sch_weight"], "sch"))

    #gq sampled contacts

    gq_groups = contacts_df.groupby('gq_id').groups
    for gq_id, indices in gq_groups.items():
        if gq_id == 'nan' or gq_id == '':
            continue
        ind_list = list(indices)
        for i in ind_list:
            contacts_to_sample = [ind for ind in ind_list if ind != i]
            k = min(params["gq_contacts"], len(contacts_to_sample))
            sampled = rng.choice(contacts_to_sample, size = k, replace = False) 
            for j in sampled:
                edge_list.append((i, j, params['gq_weight'], "gq"))
                edge_list.append((j, i, params["gq_weight"], "gq"))

    #Casual sampled contacts
    num_cas = params["cas_contacts"]
    cas_weight = params["cas_weight"]

    #gq individuals do not have casual contacts
    non_gq_mask = ~contacts_df["gq"].astype(bool)
    non_gq_indices = contacts_df.index[non_gq_mask].to_numpy()

    for i in non_gq_indices:
        possible_contacts = non_gq_indices[non_gq_indices != i]
        k = min(num_cas, len(possible_contacts))
        cas_contacts = rng.choice(possible_contacts, size = k, replace = False) if k > 0 else []
        for j in cas_contacts:
            edge_list.append((i, j, cas_weight, "cas"))
            edge_list.append((j, i, cas_weight, "cas"))


    #Combine and deduplicate edges
    edges_df = pd.DataFrame(edge_list, columns = ['source', 'target', 'weight', 'contact_type'])
    #If an individual shares a home and school/workplace as a contact, add weights
    edges_df = edges_df.groupby(["source", "target", "contact_type"], as_index = False)["weight"].sum()

    edges_df = edges_df[edges_df['source'] != edges_df['target']]

    if save:
        if not county:
            raise Exception("Save was indicated, but no county name provided")

        save_loc = os.path.join(os.getcwd(), "data", county)
        out_file = os.path.join(save_loc, (params["run_name"] + "_edgeList.parquet"))
        edges_df.to_parquet(out_file, index = False)
        print(f"Network edge list saved to {out_file}")

    return edges_df
    





