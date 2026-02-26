"""
Collection of modules needed for caching GraphData
"""

import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import json
import hashlib
import uuid
from typing import Any, Dict, Tuple, Optional

from new_scripts.graph.graph_utils import GraphData, _build_adj_matrix_from_edge_list, _build_neighbor_map


#name settings for filenames written in cache
_EDGE_FN = "master_edge_list.parquet"
_EDGE_LIST_FN = "edge_list.parquet"
_INDIV_LOOKUP_FN = "individual_lookup.parquet"
_CSR_NPZ_FN = "csr.npz"
_METADATA_FN = "metadata.json"
_LAYOUT_COORDS_FN = "layout_coords.npy"
_LAYOUT_NAMES_FN = "layout_node_names.npy"




def save_graph_cache(
    graphdata: GraphData,
    cache_dir: str,
    *,
    cache_key: Optional[str] = None,
    edge_list_df: Optional[pd.DataFrame] = None,
    params: Optional[Dict[str, Any]] = None,
    params_keys_for_hash: Optional[List[str]] = None,
    cache_version: int = 1,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Save GraphData to cache_dir.

    Accepts optional metadata fields (cache_key, edge_list_df, params, params_keys_for_hash, cache_version)
    and writes:
      - edge_list.parquet (prefer edge_list_df if provided, else graphdata.edge_list)
      - individual_lookup.parquet (graphdata.individual_lookup is required)
      - csr.npz (per-contact-type indptr/indices/weights + optional compliances)
      - metadata.json

    Returns: metadata dict written.

    Notes: atomic write (tmp dir -> os.replace into final cache_dir).
    """
    cache_dir = os.path.abspath(cache_dir)
    parent = os.path.dirname(cache_dir)
    os.makedirs(parent, exist_ok=True)

    tmp_dir = os.path.join(parent, f".tmp_graphcache_{uuid.uuid4().hex}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=False)

    try:
        # edge list (use provided edge_list_df if present, else graphdata.edge_list)
        edge_df = edge_list_df if edge_list_df is not None else getattr(graphdata, "edge_list", None)
        if edge_df is None:
            raise ValueError("No edge list available to save in cache (edge_list_df or graphdata.edge_list required).")
        edge_path = os.path.join(tmp_dir, _EDGE_LIST_FN)
        # write parquet
        edge_df.to_parquet(edge_path, index=False)

        # individual lookup required
        indiv = getattr(graphdata, "individual_lookup", None)
        if indiv is None:
            raise ValueError("GraphData must include individual_lookup to be cached")
        indiv_path = os.path.join(tmp_dir, _INDIV_LOOKUP_FN)
        indiv.to_parquet(indiv_path, index=False)

        # gather CSR arrays per contact type into an npz
        arrays: Dict[str, np.ndarray] = {}
        for i, ct in enumerate(graphdata.contact_types):
            indptr, indices, weights = graphdata.csr_by_type[ct]
            arrays[f"ct_{i}__indptr"] = np.asarray(indptr)
            arrays[f"ct_{i}__indices"] = np.asarray(indices)
            arrays[f"ct_{i}__weights"] = np.asarray(weights)

        # store compliances if present
        if getattr(graphdata, "compliances", None) is not None:
            arrays["compliances"] = np.asarray(graphdata.compliances)

        np.savez_compressed(os.path.join(tmp_dir, _CSR_NPZ_FN), **arrays)


        # params hash
        params_hash = _compute_params_hash(params or {}, keys=params_keys_for_hash)

        h = hashlib.sha256()
        with open(edge_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        input_hash = h.hexdigest()

        # compute cache key if not provided
        if cache_key is None:
            cache_key = input_hash[:16]

        # prepare metadata
        metadata = {
            "cache_key": str(cache_key),
            "input_hash": str(input_hash),
            "params_hash": str(params_hash),
            "cache_version": int(cache_version),
            "contact_types": list(graphdata.contact_types),
            "ct_to_id": {str(k): int(v) for k, v in graphdata.ct_to_id.items()},
            "id_to_ct": {int(k): str(v) for k, v in graphdata.id_to_ct.items()},
            "N": int(graphdata.N),
            "created_utc": datetime.now().isoformat(),
            "tool": "save_graph_cache",
        }

        with open(os.path.join(tmp_dir, _METADATA_FN), "w") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)

        # atomic move into place
        if os.path.exists(cache_dir):
            if not overwrite:
                raise FileExistsError(f"Cache directory {cache_dir} already exists (use overwrite=True to replace)")
            shutil.rmtree(cache_dir)
        os.replace(tmp_dir, cache_dir)
        return metadata

    except Exception:
        # cleanup tmp on failure
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

def _compute_params_hash(params: Dict[str, Any], keys=None) -> str:
    import json, hashlib
    subset = {k: params.get(k) for k in (keys or sorted(params.keys()))}
    return hashlib.sha256(json.dumps(subset, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_master_edge_list(master_df: pd.DataFrame, cache_dir: str, *, cache_key: str = None, params: Dict[str, Any] = None, params_keys_for_hash=None, overwrite=False) -> Dict[str, Any]:
    """
    Save master edge list + metadata to cache_dir
    """
    cache_dir = os.path.abspath(cache_dir)
    parent = os.path.dirname(cache_dir)
    tmp = os.path.join(parent, f".tmp_master_{uuid.uuid4().hex}")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=False)

    try:
        edge_path = os.path.join(tmp, _EDGE_FN)
        master_df.to_parquet(edge_path, index=False)

        # compute input file hash (fast on the saved parquet)
        h = hashlib.sha256()
        with open(edge_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        input_hash = h.hexdigest()

        params_hash = _compute_params_hash(params or {}, keys=params_keys_for_hash)

        if cache_key is None:
            cache_key = input_hash[:16]

        metadata = {
            "cache_key": str(cache_key),
            "input_hash": input_hash,
            "params_hash": params_hash,
            "contact_types": sorted(master_df['contact_type'].unique().tolist()),
            "N": int(master_df[['source','target']].max().max()) + 1 if not master_df.empty else 0,
            "created": datetime.now().isoformat(),
        }

        with open(os.path.join(tmp, _METADATA_FN), "w") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)

        if os.path.exists(cache_dir):
            if not overwrite:
                raise FileExistsError(f"Master cache exists: {cache_dir}")
            shutil.rmtree(cache_dir)
        os.replace(tmp, cache_dir)
        return metadata
    except Exception:
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        raise

def load_master_edge_list(cache_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load master edge list and metadata from cache_dir.
    Returns (master_df, metadata).
    """

    cache_dir = os.path.abspath(cache_dir)
    edge_path = os.path.join(cache_dir, _EDGE_FN)
    meta_path = os.path.join(cache_dir, _METADATA_FN)
    if not os.path.exists(edge_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Master cache missing files in {cache_dir}")
    df = pd.read_parquet(edge_path)
    with open(meta_path, "r") as fh:
        metadata = json.load(fh)
    return df, metadata



        

