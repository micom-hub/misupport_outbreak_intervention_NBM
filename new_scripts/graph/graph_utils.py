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
import hashlib
import os
import json
import shutil
import uuid
from datetime import datetime

from scripts.synth_data_processing import build_individual_lookup

logger = logging.getLogger(__name__)


#name settings for filenames written in cache
_EDGE_LIST_FN = "edge_list.parquet"
_INDIV_LOOKUP_FN = "individual_lookup.parquet"
_CSR_NPZ_FN = "csr.npz"
_METADATA_FN = "metadata.json"
_LAYOUT_COORDS_FN = "layout_coords.npy"
_LAYOUT_NAMES_FN = "layout_node_names.npy"




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


#Caching functions

def save_graph_cache(graphdata: GraphData, cache_dir: str, overwrite: bool = False) -> None:
    """
    Saves a cache of GraphData to 'cache_dir':
    -edge_list.parquet
    - individual_lookup.parquet
    - csr.npz for per-ct arrays and compliance
    - layout_coords.npy and layout_node_names.npy
    - metadata.json with ct and other relevant data

    cache_dir is a directory path to a temp directory (renamed in place)
    """
    cache_dir = os.path.abspath(cache_dir)
    parent = os.path.dirname(cache_dir)
    os.makedirs(parent, exist_ok=True)

    tmp_dir = os.path.join(parent, f".tmp_graphcache_{uuid.uuid4().hex}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=False)

    try:
        #edge list and individual lookup as .parquets
        edge_path = os.path.join(tmp_dir, _EDGE_LIST_FN)
        graphdata.edge_list.to_parquet(edge_path, index=False)

        indiv_path = os.path.join(tmp_dir, _INDIV_LOOKUP_FN)
        graphdata.individual_lookup.to_parquet(indiv_path, index=False)

        #gather arrays for npz
        arrays = {}
        #ct arrays
        for i, ct in enumerate(graphdata.contact_types):
            indptr, indices, weights = graphdata.csr_by_type[ct]
            arrays[f"ct_{i}__indptr"] = np.asarray(indptr)
            arrays[f"ct_{i}__indices"] = np.asarray(indices)
            arrays[f"ct_{i}__weights"] = np.asarray(weights)

        #compliances
        if getattr(graphdata, "compliances", None) is not None:
            arrays["compliances"] = np.asarray(graphdata.compliances)

        np.savez_compressed(os.path.join(tmp_dir, _CSR_NPZ_FN), **arrays)

        #metadata
        metadata = {
            "contact_types": list(graphdata.contact_types),
            "ct_to_id": {str(k): int(v) for k, v in graphdata.ct_to_id.items()},
            "id_to_ct": {int(k): str(v) for k, v in graphdata.id_to_ct.items()},
            "N": int(graphdata.N),
            "created": datetime.utcnow().isoformat() + "Z"
        }

        with open(os.path.join(tmp_dir, _METADATA_FN), "w") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)

        
        #Move data 
        if os.path.exists(cache_dir):
            if not overwrite:
                raise FileExistsError(f"Cache directory {cache_dir} already exists (use overwrite=True to replace)")
            shutil.rmtree(cache_dir)
        os.rename(tmp_dir, cache_dir)
    except Exception:
        try:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
        except Exception:
            pass
        raise

def load_graph_cache(cache_dir: str) -> GraphData:
    """
    Load cached graph object produced by save_graph_cache and return a GraphData object. Assumes cache_dir contains files written by save_graph_cache
    """

    cache_dir = os.path.abspath(cache_dir)
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Graph cache directory not found: {cache_dir}")

    # read metadata
    md_path = os.path.join(cache_dir, _METADATA_FN)
    if not os.path.exists(md_path):
        raise FileNotFoundError("metadata.json not found in cache")
    with open(md_path, "r") as fh:
        metadata = json.load(fh)

    #read in edge list and individual lookup
    edge_path = os.path.join(cache_dir, _EDGE_LIST_FN)
    if not os.path.exists(edge_path):
        raise FileNotFoundError("edge_list parquet not found in cache")
    edge_list = pd.read_parquet(edge_path)

    indiv_path = os.path.join(cache_dir, _INDIV_LOOKUP_FN)
    if not os.path.exists(indiv_path):
        raise FileNotFoundError("individual_lookup parquet not found in cache")
    individual_lookup = pd.read_parquet(indiv_path)


    #load csr npz
    csr_npz_path = os.path.join(cache_dir, _CSR_NPZ_FN)
    if not os.path.exists(csr_npz_path):
        raise FileNotFoundError("csr npz file not found in cache")
    loaded = np.load(csr_npz_path, allow_pickle=False)

    contact_types = list(metadata.get("contact_types", []))
    csr_by_type = {}
    for i, ct in enumerate(contact_types):
        key_indptr = f"ct_{i}__indptr"
        key_indices = f"ct_{i}__indices"
        key_weights = f"ct_{i}__weights"
        if key_indptr not in loaded or key_indices not in loaded or key_weights not in loaded:
            raise ValueError(f"Missing CSR arrays for contact type index {i} in csr.npz")
        indptr = np.asarray(loaded[key_indptr])
        indices = np.asarray(loaded[key_indices])
        weights = np.asarray(loaded[key_weights])
        csr_by_type[ct] = (indptr, indices, weights)


    compliances = np.asarray(loaded["compliances"]) if "compliances" in loaded.files else None


    #Build unsaved graph utils like is done in build_graph_data
    N = int(metadata.get("N", int(individual_lookup.shape[0])))
    adj = _build_adj_matrix_from_edge_list(edge_list, N)
    neighbor_map = _build_neighbor_map(edge_list)
    fast_neighbor_map = {src: {tgt: (w, ct) for (tgt, w, ct) in nbrs} for src, nbrs in neighbor_map.items()}


    ct_to_id = metadata.get("ct_to_id", {ct: i for i, ct in enumerate(contact_types)})
    id_to_ct = metadata.get("id_to_ct", {int(v): k for k, v in ct_to_id.items()})

    ages = individual_lookup["age"].to_numpy()
    sexes = individual_lookup["sex"].to_numpy()

    #return a populated GraphData object
    loaded_data = GraphData(
        N=int(N),
        edge_list=edge_list,
        adj_matrix=adj,
        individual_lookup=individual_lookup,
        ages=ages,
        sexes=sexes,
        compliances=np.asarray(compliances) if compliances is not None else None,
        neighbor_map=neighbor_map,
        fast_neighbor_map=fast_neighbor_map,
        csr_by_type=csr_by_type,
        contact_types=contact_types,
        ct_to_id={str(k): int(v) for k, v in ct_to_id.items()},
        id_to_ct={int(k): str(v) for k, v in id_to_ct.items()},
    )

    return loaded_data


    def _compute_graph_cache_key(edge_list: pd.DataFrame, params: dict, cache_version: int = 1, sample_size: int = 1024) -> str:
        """
        Create a cache key string for edge_list + params to build graph with a deterministic summary/sample of DF to keep cost low
        """
        cols = [c for c in ["source", "target", "weight", "contact_type"] if c in edge_list.columns]

        nrows = int(len(edge_list))
        agg = {
            "nrows": nrows,
            "sum_src": int(edge_list["source"].sum()) if "source" in edge_list.columns else 0,
            "sum_tgt": int(edge_list["target"].sum()) if "target" in edge_list.columns else 0,
            "sum_w": float(edge_list["weight"].sum()) if "weight" in edge_list.columns else 0.0,
            "ct_counts": {str(k): int(v) for k, v in (edge_list["contact_type"].value_counts().to_dict().items() if "contact_type" in edge_list.columns else {})},
        }







        




