"""
graph_utils.py
Contains utility functions for building contact structure data, called in outbreak_model.py _compute_network_structures()
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import truncnorm
import logging
import hashlib
import os
import json
import shutil
import uuid
from datetime import datetime
from numba import njit, int64, uint64, int32, float32, int16

from scripts.synth_data_processing import build_individual_lookup
from new_scripts.config import ModelConfig
from new_scripts.config import derive_run_seed

logger = logging.getLogger(__name__)



@dataclass
class GraphData:
    """
    Data class for contact structure data that is static for a given model run
    """
    N: int
    edge_list: pd.DataFrame
    adj_matrix: csr_matrix
    individual_lookup: pd.DataFrame
    ages: np.ndarray
    sexes: np.ndarray
    compliances: np.ndarray
    neighbor_map: Dict[int, List[Tuple[int, float, Any]]]
    fast_neighbor_map: Dict[int, Dict[int, Tuple[float, Any]]]
    csr_by_type: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    contact_types: List[str]
    ct_to_id: Dict[str, int]
    id_to_ct: Dict[int, str]
    full_node_list: List[int]

def build_graph_data(
    edge_list: pd.DataFrame,
    contacts_df: pd.DataFrame,
    params: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
    N: Optional[int] = None,
) -> GraphData:
    """
    Builds and returns a GraphData object from synthetic population data

    Args:
        edge_list (pd.DataFrame): DataFrame with cols ['source', 'target', 'weight', 'contact_type']
        contacts_df (pd.DataFrame): contacts DataFrame from synth_data_processing
        params (Dict[str, Any]): model parameter dict
        rng (Optional[np.random.Generator], optional): RNG object, created if not provided
        N (Optional[int], optional): Population Size

    Returns:
        GraphData: GraphData object 
    """
    if rng is None:
        rng = np.random.default_rng()

    if N is None:
        N = int(contacts_df.shape[0])

    #Check edge_list looks correct
    required_cols = {"source", "target", "weight", "contact_type"}
    if not required_cols.issubset(set(edge_list.columns)):
        raise ValueError(f"edge_list must have columns {required_cols}; got {list(edge_list.columns)}")


    #Adjacency Matrix
    adj = _build_adj_matrix_from_edge_list(edge_list, N)


    #Individual Characteristics
    individual_lookup = build_individual_lookup(contacts_df)

    ages = individual_lookup["age"].to_numpy()

    sexes = individual_lookup["sex"].to_numpy()

    avg_compliance = float(np.clip(params.get("mean_compliance", 1.0), 0.0, 1.0))
    sd = 0.15
    a = (0 - avg_compliance) / sd
    b = (1 - avg_compliance) / sd
    compliances = truncnorm.rvs(a, b, loc=avg_compliance, scale=sd, size=N, random_state=rng).astype(np.float32)

    #Neighbor Map
    neighbor_map = _build_neighbor_map(edge_list)
    fast_neighbor_map = {src: {tgt: (w, ct) for (tgt, w, ct) in nbrs} for src, nbrs in neighbor_map.items()}

    #csr by ct
    csr_by_type = _build_type_csrs(neighbor_map, N)
    contact_types = sorted(csr_by_type.keys())
    ct_to_id = {ct: i for i, ct in enumerate(contact_types)}
    id_to_ct = {i: ct for ct, i in ct_to_id.items()}

    #full node list
    full_node_list = sorted(set(edge_list["source"]).union(set(edge_list["target"])))


    return GraphData(
        N=int(N),
        edge_list=edge_list,
        adj_matrix=adj,
        individual_lookup=individual_lookup,
        ages=ages,
        sexes=sexes,
        compliances=compliances,
        neighbor_map=neighbor_map,
        fast_neighbor_map=fast_neighbor_map,
        csr_by_type=csr_by_type,
        contact_types=contact_types,
        ct_to_id=ct_to_id,
        id_to_ct=id_to_ct,
        full_node_list=full_node_list
    )


    
        
def sample_from_master_graphdata(
    graphdata: Dict[str, Any], #not a GraphData object
    config: ModelConfig, 
    base_run_seed: int,
    run_index: int,
    *,
    rng: Optional[np.random.Generator] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Takes a GraphData object for the mastergraph and returns a sampled graph for model run
    """
    N = graphdata["N"]
    contact_types = list(graphdata["contact_types"])
    ct_to_id = graphdata["ct_to_id"]
    run_seed = derive_run_seed(base_run_seed, run_index)
    rng_local = rng if rng is not None else np.random.default_rng(int(run_seed))


    #helper to get the mean contact count for ct
    def _get_mean_k(ct: str) -> float:
        key_map = {
            "wp": "wp_contacts",
            "sch": "sch_contacts",
            "cas": "cas_contacts",
            "gq": "gq_contacts",
            "hh": "hh_contacts"
        }
        key = key_map.get(ct)

        return float(getattr(config.population, key, 0) if hasattr(config.population, key) else 0.0)



    #make per-contact pick_count arrays 
    pick_counts_by_ct: Dict[str, np.ndarray] = {}
    for ct in contact_types:
        mean_k = _get_mean_k(ct)
        if ct == 'hh':
            #take all household contacts
            indptr, indices, weights = graphdata["csr_by_type"].get(ct, (None, None, None))
            if indptr is None:
                pick_counts_by_ct[ct] = np.zeros(N, dtype=np.int32)
            else:
                degs = (indptr[1:] - indptr[:-1]).astype(np.int32)
                pick_counts_by_ct[ct] = degs
            continue
        if mean_k <= 0.0:
            pick_counts_by_ct[ct] = np.zeros(N, dtype=np.int32)
            continue

        #Sample degree from poisson distribution
        k_arr = rng_local.poisson(mean_k, size=N).astype(np.int32)
        k_arr[k_arr < 0] = 0
        indptr, indices, weights = graphdata["csr_by_type"].get(ct, (None, None, None))
        if indptr is None:
            k_arr[:] = 0
        else:
            degs = (indptr[1:] - indptr[:-1]).astype(np.int32)
            if degs.shape[0] != N:
                k_arr[:] = 0
            else:
                k_arr = np.minimum(k_arr, degs)
        pick_counts_by_ct[ct] = k_arr


    total_picks = 0
    for ct in contact_types:
        total_picks += int(pick_counts_by_ct[ct].sum())

    if total_picks == 0:
        rows = []
        if  'hh' in graphdata["csr_by_type"]:
            indptr, indices, weights = graphdata["csr_by_type"]['hh']
            for src in range(N):
                for pos in range(indptr[src], indptr[src+1]):
                    tgt = int(indices[pos])
                    a, b = (src, tgt) if src < tgt else (tgt, src)
                    # master aggregated assures unique pair weight
                    rows.append((int(a), int(b), float(weights[pos]), 'hh'))
        df_empty = pd.DataFrame(rows, columns=['source','target','weight','contact_type'])
        if save_path:
            df_empty.to_parquet(save_path, index=False)
        return df_empty

    src_arr = np.empty(total_picks, dtype=np.int32)
    tgt_arr = np.empty(total_picks, dtype=np.int32)
    weight_arr = np.empty(total_picks, dtype=np.float32)
    ct_id_arr = np.empty(total_picks, dtype=np.int16)

    ptr = 0
    run_seed_u64 = np.uint64(run_seed)
    for ct in contact_types:
        indptr, indices, weights = graphdata["csr_by_type"].get(ct, (None, None, None))
        if indptr is None:
            continue
        pick_counts = pick_counts_by_ct[ct]
        if pick_counts.sum() == 0:
            continue
        ct_id = int(ct_to_id.get(ct, 0))
        ptr = _sample_ct_numba(indptr.astype(np.int64), indices.astype(np.int32), weights.astype(np.float32),
                               pick_counts.astype(np.int32), run_seed_u64, int(ct_id),
                               src_arr, tgt_arr, weight_arr, ct_id_arr, ptr)



    # shorten arrays to actual size
    src_arr = src_arr[:ptr]
    tgt_arr = tgt_arr[:ptr]
    weight_arr = weight_arr[:ptr]
    ct_id_arr = ct_id_arr[:ptr]

    if ptr == 0:
        df_empty = pd.DataFrame(columns=['source','target','weight','contact_type'])
        if save_path:
            df_empty.to_parquet(save_path, index=False)
        return df_empty

    # make pairs unordered and deduplicate
    a = np.minimum(src_arr, tgt_arr).astype(np.int32)
    b = np.maximum(src_arr, tgt_arr).astype(np.int32)
    ct_ids = ct_id_arr.astype(np.int16)
    ws = weight_arr.astype(np.float32)

    neg_w = (-ws).astype(np.float64)  # higher weight first
    order = np.lexsort((ct_ids, neg_w, b, a))
    a_s = a[order]
    b_s = b[order]
    ct_s = ct_ids[order]
    w_s = ws[order]

    # boolean mask to keep unique (a,b) first occurrence
    keep = np.ones(a_s.shape[0], dtype=np.bool_)
    if a_s.shape[0] > 1:
        dup = (a_s[1:] == a_s[:-1]) & (b_s[1:] == b_s[:-1])
        keep[1:] = ~dup

    a_u = a_s[keep]
    b_u = b_s[keep]
    ct_u = ct_s[keep]
    w_u = w_s[keep]

    # map ct_id back to name
    id_to_ct = {int(v): str(k) for k, v in graphdata["ct_to_id"].items()}

    # build df
    contact_type_names = [id_to_ct[int(x)] for x in ct_u]
    df = pd.DataFrame({
        'source': a_u.astype(np.int32),
        'target': b_u.astype(np.int32),
        'weight': w_u.astype(np.float32),
        'contact_type': np.array(contact_type_names, dtype=object)
    })

    df = df.sort_values(by=['contact_type', 'source', 'target']).reset_index(drop=True)

    if save_path:
        df.to_parquet(save_path, index=False)

    return df




#Helpers for graph building
def _build_adj_matrix_from_edge_list(edge_list: pd.DataFrame, N: int) -> csr_matrix:
    """
    Build an adjacency csr matrix from edge_list
    """
    src = edge_list["source"].to_numpy(dtype=np.int32)
    tgt = edge_list["target"].to_numpy(dtype=np.int32)
    weights = edge_list["weight"].to_numpy(dtype=np.float32)

    row = np.concatenate([src, tgt])
    col = np.concatenate([tgt, src])
    dat = np.concatenate([weights, weights])

    adj = csr_matrix((dat, (row, col)), shape=(N, N))
    return adj


def _build_neighbor_map(edge_list: pd.DataFrame) -> Dict[int, List[Tuple[int, float, Any]]]:
    """Return dict: src -> list of (tgt, weight, contact_type).
    Pre-process edge list into a dict of src -> [(tgt, weight, ct), ...]
    """
    neighbor_map: Dict[int, List[Tuple[int, float, Any]]] = {}
    for row in edge_list.itertuples(index=False):
        src, tgt, w, ct = row.source, row.target, row.weight, row.contact_type
        if src not in neighbor_map:
            neighbor_map[src] = []
        neighbor_map[src].append((tgt, w, ct))
        if tgt not in neighbor_map:
            neighbor_map[tgt] = []
        neighbor_map[tgt].append((src, w, ct))
    return neighbor_map

def _build_type_csrs(neighbor_map: Dict[int, List[Tuple[int, float, Any]]], N: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """    
    Builds a dict: ct -> [(indptr, ind, weight)] that indexes from source node to list of neighbors for a given contact type
    """
    total_edges = int(sum(len(neigh) for neigh in neighbor_map.values()))
    if total_edges == 0:
        return {}

    src_array = np.empty(total_edges, dtype=np.int32)
    tgt_array = np.empty(total_edges, dtype=np.int32)
    wt_array = np.empty(total_edges, dtype=np.float32)

    ct_to_id: Dict[Any, int] = {}
    id_to_ct: List[Any] = []
    ct_array = np.empty(total_edges, dtype=np.int16)

    pos = 0
    for src, neighbors in neighbor_map.items():
        for tgt, wt, ct in neighbors:
            src_array[pos] = int(src)
            tgt_array[pos] = int(tgt)
            wt_array[pos] = float(wt)
            if ct not in ct_to_id:
                ct_to_id[ct] = len(id_to_ct)
                id_to_ct.append(ct)
            ct_array[pos] = ct_to_id[ct]
            pos += 1

    # Trim length
    if pos < total_edges:
        src_array = src_array[:pos]
        tgt_array = tgt_array[:pos]
        wt_array = wt_array[:pos]
        ct_array = ct_array[:pos]

    order = np.lexsort((src_array, ct_array))  
    src_s = src_array[order]
    tgt_s = tgt_array[order]
    wt_s = wt_array[order]
    ct_s = ct_array[order]

    csr_by_type: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    unique_cts, ct_starts = np.unique(ct_s, return_index=True)
    ct_starts = list(ct_starts) + [len(ct_s)]
    for k, ct_id in enumerate(unique_cts):
        start = ct_starts[k]
        end = ct_starts[k + 1]
        srcs_ct = src_s[start:end]
        tgts_ct = tgt_s[start:end]
        wts_ct = wt_s[start:end]

        if srcs_ct.size == 0:
            indptr = np.zeros(N + 1, dtype=np.int64)
            indices = np.empty(0, dtype=np.int32)
            weights = np.empty(0, dtype=np.float32)
        else:
            counts = np.bincount(srcs_ct, minlength=N)
            indptr = np.empty(N + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])
            indices = tgts_ct.astype(np.int32, copy=True)
            weights = wts_ct.astype(np.float32, copy=True)

        ct_name = id_to_ct[int(ct_id)]
        csr_by_type[ct_name] = (indptr, indices, weights)

    return csr_by_type


def _build_type_csr_from_edge_list(edge_list: pd.DataFrame, N: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Alternate function for building type_csrs from an edge list instead of neighbor list. Used for master_edge_list to minimize unnecessary computation
    """
    src = edge_list["source"].to_numpy(dtype=np.int32)
    tgt = edge_list["target"].to_numpy(dtype=np.int32)
    wts = edge_list["weight"].to_numpy(dtype=np.float32)
    cts = edge_list["contact_type"].to_numpy(dtype=object)

    src2 = np.concatenate([src, tgt]).astype(np.int32)
    tgt2 = np.concatenate([tgt, src]).astype(np.int32)
    wts2 = np.concatenate([wts, wts]).astype(np.float32)


    unique_cts, ct_indices = np.unique(cts, return_inverse=True)
    ct_ids_dir = np.concatenate([ct_indices, ct_indices]).astype(np.int16)

    order = np.lexsort((src2, ct_ids_dir)) 
    src_s = src2[order]
    tgt_s = tgt2[order]
    wts_s = wts2[order]
    ct_s = ct_ids_dir[order]

    csr_by_type = {}

    uniq_ct_ids, starts = np.unique(ct_s, return_index=True)
    starts = list(starts) + [len(ct_s)]
    for idx_pos, ct_id in enumerate(uniq_ct_ids):
        st = starts[idx_pos]
        en = starts[idx_pos + 1]
        src_ct = src_s[st:en]
        tgt_ct = tgt_s[st:en]
        w_ct = wts_s[st:en]
        if src_ct.size == 0:
            indptr = np.zeros(N + 1, dtype=np.int64)
            indices = np.empty(0, dtype=np.int32)
            weights = np.empty(0, dtype=np.float32)
        else:
            counts = np.bincount(src_ct, minlength=N)
            indptr = np.empty(N + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])
            indices = tgt_ct.astype(np.int32, copy=True)
            weights = w_ct.astype(np.float32, copy=True)
        ct_name = unique_cts[int(ct_id)]
        csr_by_type[str(ct_name)] = (indptr, indices, weights)

    return csr_by_type

def build_minimal_graphdata_from_edge_list(edge_list: pd.DataFrame, N: int):
    """
   Builds a GraphData-like object from an edge_list without computing computationally intensive aspects
    """
    csr_by_type = _build_type_csr_from_edge_list(edge_list, N)
    contact_types = sorted(csr_by_type.keys())
    ct_to_id = {ct: i for i, ct in enumerate(contact_types)}
    id_to_ct = {i: ct for ct, i in ct_to_id.items()}
    full_node_list = sorted(set(edge_list['source']).union(set(edge_list['target'])))
    # pack minimal structure (you can use a small dataclass, but dict is simple)
    return {
        "N": int(N),
        "edge_list": edge_list,
        "csr_by_type": csr_by_type,
        "contact_types": contact_types,
        "ct_to_id": ct_to_id,
        "id_to_ct": id_to_ct,
        "full_node_list": full_node_list
    }


@njit(uint64(uint64))
def _splitmix64_numba(x: uint64) -> uint64:
    # run splitmix64 algorithm
    x = (x + uint64(0x9E3779B97F4A7C15)) & uint64(0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> uint64(30))) * uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> uint64(27))) * uint64(0x94D049BB133111EB)
    x = x ^ (x >> uint64(31))
    return x

#numba sampling routine for contact type
@njit(int64(int64[:], int32[:], float32[:], int32[:], uint64, int32,
            int32[:], int32[:], float32[:], int16[:], int64))
def _sample_ct_numba(indptr, indices, weights, pick_counts, run_seed_u64, ct_id,
                      out_src, out_tgt, out_weight, out_ct_id, start_ptr):
    """
    indptr: int64 array length N+1
    indices: int32 array of neighbor targets
    weights: float32 array of neighbor weights (aligned with indices)
    pick_counts: int32 array length N = number to pick per node (already capped to deg)
    run_seed_u64: uint64 run seed
    ct_id: int32 contact-type id
    out_src/out_tgt/out_weight/out_ct_id: preallocated arrays to write into
    start_ptr: starting write index into out arrays
    returns new write pointer
    """
    ptr = start_ptr
    N = indptr.shape[0] - 1
    for src in range(N):
        start = indptr[src]
        end = indptr[src + 1]
        m = end - start
        k = pick_counts[src]
        if m == 0 or k <= 0:
            continue
        if k >= m:
            for pos in range(start, end):
                out_src[ptr] = src
                out_tgt[ptr] = indices[pos]
                out_weight[ptr] = weights[pos]
                out_ct_id[ptr] = ct_id
                ptr += 1
            continue

        tmp_neigh = np.empty(m, dtype=np.int32)
        tmp_w = np.empty(m, dtype=np.float32)
        for ii in range(m):
            tmp_neigh[ii] = indices[start + ii]
            tmp_w[ii] = weights[start + ii]

        # partial Fisher-Yates with deterministic pseudo-random values via splitmix
        # for t in [0..k-1]: swap tmp[t] with tmp[r] where r in [t, m-1] chosen by splitmix
        for t in range(k):
            # form a unique counter combining run_seed, ct_id, src and t
            # Use additions to avoid overflow behavior (uint64 wraps naturally)
            ctr = uint64(run_seed_u64 + uint64(ct_id) + uint64(src) + uint64(t))
            rnd = _splitmix64_numba(ctr)
            # range length = m - t > 0
            rem = m - t
            r = t + int(rnd % uint64(rem))
            # swap neighbors and weights
            tmp_n = tmp_neigh[t]
            tmp_neigh[t] = tmp_neigh[r]
            tmp_neigh[r] = tmp_n
            tmp_wv = tmp_w[t]
            tmp_w[t] = tmp_w[r]
            tmp_w[r] = tmp_wv
            # record selected (tmp_neigh[t])
            out_src[ptr] = src
            out_tgt[ptr] = tmp_neigh[t]
            out_weight[ptr] = tmp_w[t]
            out_ct_id[ptr] = ct_id
            ptr += 1

    return ptr



    

###################
#Caching functions#
###################




