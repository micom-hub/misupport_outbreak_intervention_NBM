"""
graph_utils.py
Contains utility functions for building contact structure data, called in outbreak_model.py _compute_network_structures()
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import truncnorm
import logging
from line_profiler import profile

from scripts.utils.synth_data_processing import build_individual_lookup
from scripts.config import ModelConfig

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
    degrees_arr: np.ndarray

def build_graph_data(
    edge_list: pd.DataFrame,
    contacts_df: pd.DataFrame,
    config: ModelConfig,
    seed: int,
    N: Optional[int] = None,
) -> GraphData:
    """
    Builds and returns a GraphData object from synthetic population data

    Args:
        edge_list (pd.DataFrame): DataFrame with cols ['source', 'target', 'weight', 'contact_type']
        contacts_df (pd.DataFrame): contacts DataFrame from synth_data_processing
        config: ModelConfig object
        N (Optional[int], optional): Population Size

    Returns:
        GraphData: GraphData object 
    """
    if seed is not None:
        rng = np.random.default_rng(int(seed))
    else:
        raise TypeError("No seed provided to build graph data")

    if N is None:
        N = int(contacts_df.shape[0])

    #Check edge_list looks correct
    required_cols = {"source", "target", "weight", "contact_type"}
    if not required_cols.issubset(set(edge_list.columns)):
        raise ValueError(f"edge_list must have columns {required_cols}; got {list(edge_list.columns)}")


    #Adjacency Matrix
    adj = _build_adj_matrix_from_edge_list(edge_list, N)

    #Degrees (unweighted) tgts per src
    degrees_arr = adj.getnnz(axis=1).astype(np.int32)

    #Individual Characteristics
    individual_lookup = build_individual_lookup(contacts_df)

    ages = individual_lookup["age"].to_numpy()

    sexes = individual_lookup["sex"].to_numpy()

    avg_compliance = float(np.clip(config.lhd.mean_compliance, 0.0, 1.0))
    sd = 0.15
    a = (0 - avg_compliance) / sd
    b = (1 - avg_compliance) / sd
    compliances = truncnorm.rvs(a, b, loc=avg_compliance, scale=sd, size=N, random_state=np.random.RandomState(int(seed))).astype(np.float32)

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
        full_node_list=full_node_list,
        degrees_arr=degrees_arr
    )


    

def sample_from_master_graphdata(
    minimal_master_dict: Dict,
    config: ModelConfig,
    seed: int
) -> pd.DataFrame:
    """
    Samples contacts a minimal representation of a master edge list produced by build_minimal_graphdata_from_edge_list() according to provided ModelConfig object

    Returns a sampled pandas.DataFrame with same columns as master_edgelist
    """
    if seed is not None:
        rng = np.random.default_rng(int(seed))
    else:
        raise TypeError("No seed provided to sample master graphdata")

    N = int(minimal_master_dict["N"])
    csr_by_type = minimal_master_dict["csr_by_type"]
    ct_to_id = minimal_master_dict["ct_to_id"]

    #Mapping contact types to sample (not hh)
    ct_to_popkey = { "wp": "wp_contacts", "sch": "sch_contacts", "cas": "cas_contacts", "gq": "gq_contacts"}

    pop = getattr(config, "population", None)

    src_list = []
    tgt_list = []
    wt_list = []
    ct_list = []

    #Gather all household contacts
    if "hh" in csr_by_type:
        indptr_hh, indices_hh, weights_hh = csr_by_type["hh"]
        indptr_hh = np.asarray(indptr_hh, dtype=np.int64)
        indices_hh = np.asarray(indices_hh, dtype=np.int32)
        weights_hh = np.asarray(weights_hh, dtype=np.float32) if weights_hh is not None else np.ones(indices_hh.shape[0], dtype=np.float32)

        if indptr_hh.ndim != 1 or indptr_hh.shape[0] != N + 1:
            # Fallback: if shapes are unexpected, collect via loop (robust fallback)
            for src in range(N):
                s = int(indptr_hh[src]) if src + 1 < indptr_hh.shape[0] else 0
                e = int(indptr_hh[src + 1]) if src + 1 < indptr_hh.shape[0] else s
                for pos in range(s, e):
                    tgt = int(indices_hh[pos])
                    w = float(weights_hh[pos]) if weights_hh.size > 0 else 1.0
                    a = min(src, tgt)
                    b = max(src, tgt)
                    src_list.append(a)
                    tgt_list.append(b)
                    wt_list.append(w)
                    ct_list.append("hh")
        else:
            counts = (indptr_hh[1:] - indptr_hh[:-1]).astype(np.int32)
            total = int(counts.sum())
            if total > 0:
                # srcs repeated according to counts aligns with indices_hh
                srcs = np.repeat(np.arange(N, dtype=np.int32), counts)
                tgts = indices_hh
                wts = weights_hh
                # a <= b for undirected canonicalization
                i_hh = np.minimum(srcs, tgts).astype(np.int32)
                j_hh = np.maximum(srcs, tgts).astype(np.int32)
                # sort by (a,b) primary and -weight to keep highest-weight first within duplicates
                neg_w = -wts.astype(np.float64)
                order = np.lexsort((neg_w, j_hh, i_hh))  # primary a, then b, then -weight
                i_s = i_hh[order]
                j_s = j_hh[order]
                w_s = wts[order]
                # keep unique (a,b) first occurrence (which is highest weight)
                keep_mask = np.ones(i_s.shape[0], dtype=bool)
                if i_s.shape[0] > 1:
                    dup = (i_s[1:] == i_s[:-1]) & (j_s[1:] == j_s[:-1])
                    keep_mask[1:] = ~dup
                a_u = i_s[keep_mask]
                b_u = j_s[keep_mask]
                w_u = w_s[keep_mask]
                src_list.extend(a_u.tolist())
                tgt_list.extend(b_u.tolist())
                wt_list.extend(w_u.tolist())
                ct_list.extend(["hh"] * len(a_u))

    #Extract per-node information for contacts and degree by ct
    for ct, popkey in ct_to_popkey.items():
        #household gets special treatment and takes all contacts, otherwise sample

        mean_deg = float(getattr(pop, popkey, 0.0)) if pop is not None else 0.0
        if mean_deg <= 0.0:
            continue
        if ct not in csr_by_type:
            continue
        indptr, indices, weights = csr_by_type[ct]
        indptr = np.asarray(indptr, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float32) if weights is not None else np.empty(indices.shape[0], dtype=np.float32)


        if indptr.ndim != 1 or indptr.shape[0] < 2:
            continue
        degs = (indptr[1:] - indptr[:-1]).astype(np.int32)
        if degs.shape[0] != N:
            raise ValueError(f"Contact type '{ct}' degree array length {degs.shape[0]} != minimal_master_dict['N'] {N}")

        #Sample degree per node from poisson, capped by availability
        degrees_arr = rng.poisson(mean_deg, size = N).astype(np.int32)
        degrees_arr = np.minimum(degs, degrees_arr)

        nonzero_nodes = np.nonzero(degrees_arr > 0)[0]
        for src in nonzero_nodes:
            k = int(degrees_arr[src])
            start = int(indptr[src])
            end = int(indptr[src+1])
            m = end - start
            if m <= 0 or k <= 0:
                continue
            if k >= m:
                chosen_positions = np.arange(start, end, dtype = np.int64)
            else:
                choices = rng.choice(m, size = k, replace = False)
                chosen_positions = start + np.asarray(choices, dtype = np.int64)

            for pos in chosen_positions:
                tgt = int(indices[pos])
                w = float(weights[pos])
                src_list.append(src)
                tgt_list.append(tgt)
                wt_list.append(w)
                ct_list.append(ct)

    if len(src_list) == 0:
        return pd.DataFrame(columns=["source","target","weight","contact_type"])

    
    src_arr = np.array(src_list, dtype=np.int32)
    tgt_arr = np.array(tgt_list, dtype=np.int32)
    wt_arr = np.array(wt_list, dtype=np.float32)
    ct_arr = np.array(ct_list, dtype=object)

    #Make pairs unordered, sort by ct, and deduplicate if needed
    i = np.minimum(src_arr, tgt_arr)
    j = np.maximum(src_arr, tgt_arr)
    ct_id_arr = np.array([ct_to_id.get(ct, 0) for ct in ct_arr], dtype=np.int16)

    neg_w = -wt_arr.astype(np.float64)
    order = np.lexsort((ct_id_arr, neg_w, j, i))

    i_sorted = i[order]
    j_sorted = j[order]
    wt_sorted = wt_arr[order]
    ct_sorted = ct_arr[order]

    keep = np.ones(i_sorted.shape[0], dtype = bool)
    if i_sorted.shape[0] > 1 :
        dup = ((i_sorted[1:] == i_sorted[:-1]) & (j_sorted[1:] == j_sorted[:-1]))
        keep[1:] = ~dup
    
    i_final = i_sorted[keep]
    j_final = j_sorted[keep]
    wt_final = wt_sorted[keep]
    ct_final = ct_sorted[keep]

    df = pd.DataFrame({
        "source":i_final.astype(np.int32),
        "target": j_final.astype(np.int32),
        "weight": wt_final.astype(np.float32),
        "contact_type": np.asarray(ct_final, dtype = object)
    })

    df = df.sort_values(by = ["contact_type", "source", "target"]).reset_index(drop = True)

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



