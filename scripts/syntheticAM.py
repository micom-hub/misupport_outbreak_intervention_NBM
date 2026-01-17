##Build an adjacency matrix (or layered adjacency matrices) representing synthetic populations
import os
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, save_npz

import igraph as ig

#TODO Right now it reads in all of an individual's contacts as the same weight, which should not be the case

def df_to_adjacency(
    county, state="Michigan", matrixName="adjMat", saveFile=False, sparse=True
):
    """_summary_

    Args:
        county (_type_): _description_
        state (str, optional): _description_. Defaults to "Michigan".
        matrixName (str, optional): _description_. Defaults to "adjMat".
        saveFile (bool, optional): _description_. Defaults to False.

    Returns:
        np.array: adjacency matrix, as CSR if sparse=True
    """

    projectDirectory = os.getcwd()

    hh_weight, wp_weight, sch_weight, gq_weight = 3.2, 2.2, 1.8, 4.1
    weightingTable = {
        "hh_id": hh_weight,
        "wp_id": wp_weight,
        "sch_id": sch_weight,
        "gq_id": gq_weight,
    }

    cd = projectDirectory
    data_dir = cd + "/data"
    contact_data_path = os.path.join(data_dir, county, "contacts.parquet")

    # Might be faster to have dataGather.py pass the df rather than save parquets
    df = pd.read_parquet(contact_data_path)

    pids = df["PID"].tolist()

    pid_ind = {pid: ind for ind, pid in enumerate(pids)}
    N = len(pids)

    adj = lil_matrix((N, N), dtype=float)

    for col, weight in weightingTable.items():
        for locid, group in df.groupby(col):
            if not pd.isnull(locid):
                indices = [pid_ind[pid] for pid in group["PID"]]
                for i in indices:
                    for j in indices:
                        if i != j:
                            adj[
                                i, j
                            ] += weight  # Right now accumulates weight (if same school + work), could change to take max only

    if sparse == True:
        adj_csr = adj.tocsr()
        return adj_csr

    if saveFile == True:
        adj_filepath = os.path.join(data_dir, county, matrixName)
        save_npz(adj_filepath, adj_csr)
        print(f"Sparse CSR adjacency matrix saved to {adj_filename}")

    return adj


if __name__ == "__main__":
    adj = df_to_adjacency(
        "Alcona", state="Michigan", matrixName="adjMat", saveFile=False, sparse=True
    )
