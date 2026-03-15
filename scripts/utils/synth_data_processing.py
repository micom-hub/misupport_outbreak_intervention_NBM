import os
import zipfile
import pandas as pd
import numpy as np
from itertools import combinations
from numba import njit
from scripts.config import ModelConfig
from line_profiler import profile
from typing import Dict, Any

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
    # ###TODO IMPLEMENT LATER IF GIS ANALYSIS USEFUL
    # def members_grouped(df, colname, label):
    #     valid = df[~df[colname].isna()]
    #     mems = valid.groupby(colname)["PID"].apply(list).reset_index()
    #     mems.columns = [colname, f"{label}_members"]
    #     return mems

    # sch_members = members_grouped(contacts, "sch_id", "sch")
    # wp_members = members_grouped(contacts, "wp_id", "wp")
    # gq_members = members_grouped(contacts[contacts["gq"] == True], "gq_id", "gq")  # noqa: E712
    # hh_members = members_grouped(contacts, "hh_id", "hh")

    # contacts = contacts.merge(hh_members, on="hh_id", how="left")
    # contacts = contacts.merge(gq_members, on="gq_id", how="left")
    # contacts = contacts.merge(wp_members, on="wp_id", how="left")
    # contacts = contacts.merge(sch_members, on="sch_id", how="left")

    # # build a locations df
    # def build_location_df(members_df, id_col, mem_col, type_name):
    #     out = pd.DataFrame({
    #         "id": members_df[id_col].astype(str),
    #         "members": members_df[mem_col],
    #         "type": type_name,
    #     })
    #     out["size"] = out["members"].apply(len)
    #     return out

    # school_locations = build_location_df(sch_members, "sch_id", "sch_members", "School")
    # work_locations   = build_location_df(wp_members, "wp_id", "wp_members", "Workplace")
    # gq_locations     = build_location_df(gq_members, "gq_id", "gq_members", "Group Quarters")
    # hh_locations     = build_location_df(hh_members, "hh_id", "hh_members", "Household")

    # locations = pd.concat(
    #     [school_locations, work_locations, gq_locations, hh_locations],
    #     ignore_index=True,
    # )

    # ---- pare down and save to parquet
    columns_interest = [
        "PID", "hh_id", "wp_id", "sch_id", "gq_id", "age", "sex", "race", "relate", "gq"
    ]
    contacts_pruned = contacts[columns_interest].copy()

    print("Data Wrangled!")

    if save_files:
        print("Saving to parquet")
        contacts_pruned.to_parquet(os.path.join(county_dat, "contacts.parquet"))
        # locations.to_parquet(os.path.join(county_dat, "locations.parquet"))
        print(f"Contact Files Saved to {county_dat}")

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
    seed: int = 2026,
    save: bool = False, 
    county: str = None,
    master_casual_contacts: int = 100
) -> pd.DataFrame:
    """
    Generate a master edge list of all possible contact combinations, to be sampled from in graph construction

    Args:
        contacts_df (pd.DataFrame): Output from synthetic_data_process
        config: a ModelConfig object
        rng (np.rng): a seeded rng for reproducibility
        save (Bool): if true, will save edge list to parquet
        county (str): if data is to be saved, direct to proper data file

    Returns:
        edges_df (pd.DataFrame): DataFrame with ['source', 'target','weight', 'ct]
    """

    #Re-index contacts_df
    contacts_df = contacts_df.reset_index(drop = True)
    cfg = config

    rng = np.random.default_rng(int(seed))

    #Gather weights
    hh_weight = float(cfg.population.hh_weight)
    wp_weight = float(cfg.population.wp_weight)
    sch_weight = float(cfg.population.sch_weight)
    gq_weight = float(cfg.population.gq_weight)
    cas_weight = float(cfg.population.cas_weight)

    # mapping for prioritizing contact types for duplicate contacts 
    ct_to_weight = {"hh": hh_weight, "sch": sch_weight, "wp": wp_weight, "gq": gq_weight, "cas": cas_weight}
    cts_sorted = sorted(ct_to_weight.items(), key=lambda kv: (-kv[1], kv[0]))
    ct_priority: Dict[str, int] = {ct: i for i, (ct, _) in enumerate(cts_sorted)}
    
    def _skip_gid(gid: Any) -> bool:
        #Helper to skip group if missing/na
        if gid is None or pd.isna(gid):
            return True
        if isinstance(gid, str) and gid.strip().lower() in ("", "nan", "none"):
            return True
        return False


##### Building structured contacts (HH, WP, SCH, GQ)
    edge_list = []

    def _add_pair(i,j,w,ct):
        if i == j:
            return
        s, t = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
        edge_list.append((s, t, float(w), ct))

    for gid, ind in contacts_df.groupby("hh_id", sort=False).groups.items():
        if _skip_gid(gid):
            continue
        idx = list(ind)
        if len(idx) >= 2:
            for i, j in combinations(idx, 2):
                _add_pair(i, j, hh_weight, "hh")

    for gid, ind in contacts_df.groupby("wp_id", sort=False).groups.items():
        if _skip_gid(gid):
            continue
        idx = list(ind)
        if len(idx) >= 2:
            for i, j in combinations(idx, 2):
                _add_pair(i, j, wp_weight, "wp")

    for gid, ind in contacts_df.groupby("sch_id", sort=False).groups.items():
        if _skip_gid(gid):
            continue
        idx = list(ind)
        if len(idx) >= 2:
            for i, j in combinations(idx, 2):
                _add_pair(i, j, sch_weight, "sch")

    for gid, ind in contacts_df.groupby("gq_id", sort=False).groups.items():
        if _skip_gid(gid):
            continue
        idx = list(ind)
        if len(idx) >= 2:
            for i, j in combinations(idx, 2):
                _add_pair(i, j, gq_weight, "gq")

    #Aggregate types into structured contacts DF 
    structured_df = pd.DataFrame(edge_list, columns=["source","target","weight","contact_type"])
    if not structured_df.empty:
        structured_df["source"] = structured_df["source"].astype(np.int32)
        structured_df["target"] = structured_df["target"].astype(np.int32)
        structured_df["weight"] = structured_df["weight"].astype(np.float32)
        structured_df["ct_priority"] = structured_df["contact_type"].map(ct_priority).astype(np.int16)

    #Keep highest weight structured contat per pair
    structured_df = structured_df.sort_values(
            by=["source", "target", "ct_priority"],
            ascending=[True, True, True],
            kind="mergesort",
        ).drop_duplicates(subset=["source", "target"], keep="first").drop(columns=["ct_priority"]).reset_index(drop=True)


####Unstructured / Casual Bulk sampling
    gq_flag = contacts_df["gq"].astype(bool).to_numpy()
    non_gq = np.flatnonzero(~gq_flag).astype(np.int32)

    hh_codes = _factorize_ids(contacts_df["hh_id"])
    wp_codes = _factorize_ids(contacts_df["wp_id"])
    sch_codes = _factorize_ids(contacts_df["sch_id"])

    src_cas, tgt_cas = sample_casual_edges_bulk(
        pool=non_gq,
        hh=hh_codes, wp=wp_codes, sch=sch_codes,
        k_min=int(master_casual_contacts),
        rng=rng)

    s = np.minimum(src_cas, tgt_cas).astype(np.int32, copy=False)
    t = np.maximum(src_cas, tgt_cas).astype(np.int32, copy=False)

    #build df without weights for now
    casual_df = pd.DataFrame({
            "source": s.astype(np.int32, copy=False),
            "target": t.astype(np.int32, copy=False),
            "contact_type": "cas",
            })

    #Combine dfs, deduplicating by ct_priority
    if structured_df.empty:
        edges_df = casual_df
    elif casual_df.empty:
        edges_df = structured_df[["source", "target", "contact_type"]]
    else:
        edges_df = pd.concat(
            [structured_df[["source", "target", "contact_type"]], casual_df],
            ignore_index=True,
        )

        #Prioritize contact types -- lower ct_prio = higher weight
    edges_df["ct_priority"] = edges_df["contact_type"].map(ct_priority).astype(np.int16)

    #highest weight row comes first, drop following pairs
    edges_df = (
        edges_df.sort_values(
            by=["source", "target", "ct_priority"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["source", "target"], keep="first")
        .drop(columns=["ct_priority"])
        .reset_index(drop=True)
    )

    #Add weights back
    edges_df["weight"] = edges_df["contact_type"].map(ct_to_weight).astype(np.float32)
    edges_df["source"] = edges_df["source"].astype(np.int32)
    edges_df["target"] = edges_df["target"].astype(np.int32)
    edges_df["contact_type"] = edges_df["contact_type"].astype(str)
    edges_df = edges_df[["source", "target", "weight", "contact_type"]]

    if save:
        if not county:
            raise Exception("Save requested but no county provided")
        save_loc = os.path.join(os.getcwd(), "data", county)
        os.makedirs(save_loc, exist_ok=True)
        out_file = os.path.join(save_loc, (cfg.sim.run_name + "_master_edgeList.parquet"))
        edges_df.to_parquet(out_file, index=False)
        print(f"Master network edge list saved to {out_file}")

    return edges_df



def _factorize_ids(s: pd.Series) -> np.ndarray:
    # Helper to factorize contact types
    x = s.astype("string").replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    codes, _ = pd.factorize(x, sort=False)
    return codes.astype(np.int32)  # missing becomes -1 


def allowed_mask(a, b, hh, wp, sch):
    #Helper to check if a and b share an hh, wp, or sch
    ok = np.ones(a.size, dtype=bool)

    ha = hh[a]; hb = hh[b]
    ok &= ~((ha != -1) & (hb != -1) & (ha == hb))

    wa = wp[a]; wb = wp[b]
    ok &= ~((wa != -1) & (wb != -1) & (wa == wb))

    sa = sch[a]; sb = sch[b]
    ok &= ~((sa != -1) & (sb != -1) & (sa == sb))

    return ok

def sample_casual_edges_bulk(
    pool, #person IDs array
    hh, wp, sch, #int contact-type id codes per person
    k_min, #minimum contacts
    rng,
    seed_factor=1.05,     # >=1.0 ; 1.0 is minimal mean degree=k_min, >1 reduces top-up work
    chunk_candidates=2_000_000,
    topup_chunk=2_000_000,
    max_topup_iters=5
):
    """
    Returns (src, tgt) int arrays of undirected edges

    Ensures every node in pool has degree >= k_min after batch assignment and filtering
    """
    pool = pool.astype(np.int32, copy=False)
    P = pool.size
    if (P < 2 or k_min <= 0):
        return np.empty(0, np.int32), np.empty(0, np.int32)
    
    deg = np.zeros(P, dtype=np.int64)

    src_parts = []
    tgt_parts = []

    #Initial bulk sample
    target_seed_edges = int(np.ceil(seed_factor * (P*k_min / 2.0)))
    accepted = 0

    while accepted < target_seed_edges:
        #chunking for memory considerations
        m = min(chunk_candidates, target_seed_edges - accepted)

        #sample positions
        a_pos = rng.integers(0, P, size=m, dtype=np.int32) #a's index in pool
        delta = rng.integers(1, P, size=m, dtype=np.int32)
        b_pos = (a_pos + delta) % P #b's index in pool

        a = pool[a_pos] #global index of a
        b=pool[b_pos] #global index of b

        ok = allowed_mask(a, b, hh, wp, sch)
        if not ok.any():
            continue

        a_pos_ok = a_pos[ok]
        b_pos_ok = b_pos[ok]
        a_ok = a[ok]
        b_ok = b[ok]

        s = np.minimum(a_ok, b_ok)
        t = np.maximum(a_ok, b_ok)

        src_parts.append(s.astype(np.int32, copy=False))
        tgt_parts.append(t.astype(np.int32, copy=False))

        deg += np.bincount(a_pos_ok, minlength=P)
        deg += np.bincount(b_pos_ok, minlength=P)

        accepted += s.size


    #Top up any with insufficient degree

    for it in range(max_topup_iters):
        deficient = np.flatnonzero(deg < k_min).astype(np.int32, copy=False)
        if deficient.size == 0:
            break

        need = (k_min - deg[deficient]).astype(np.int64, copy=False)
        total_need= int(need.sum())

        sources_pos = np.repeat(deficient, need).astype(np.int32, copy=False)

        start = 0
        while start < sources_pos.size:
            end = min(start + topup_chunk, sources_pos.size)
            a_pos = sources_pos[start:end]

            # targets: uniform over all OTHER pool positions
            delta = rng.integers(1, P, size=(end - start), dtype=np.int32)
            b_pos = (a_pos + delta) % P

            a = pool[a_pos]
            b = pool[b_pos]

            ok = allowed_mask(a, b, hh, wp, sch)
            if ok.any():
                a_pos_ok = a_pos[ok]
                b_pos_ok = b_pos[ok]
                a_ok = a[ok]
                b_ok = b[ok]

                s = np.minimum(a_ok, b_ok)
                t = np.maximum(a_ok, b_ok)

                src_parts.append(s.astype(np.int32, copy=False))
                tgt_parts.append(t.astype(np.int32, copy=False))

                deg += np.bincount(a_pos_ok, minlength=P)
                deg += np.bincount(b_pos_ok, minlength=P)

            start = end

    src = np.concatenate(src_parts) if src_parts else np.empty(0, np.int32)
    tgt = np.concatenate(tgt_parts) if tgt_parts else np.empty(0, np.int32)

    return src.astype(np.int32, copy=False), tgt.astype(np.int32, copy=False)
