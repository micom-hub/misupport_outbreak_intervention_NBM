import copy
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import pandas as pd
import numpy as np


# A small default config used by DummyModelConfig
DEFAULT_CONFIG = {
    "sim": {"seed": 2026, "county": "COUNTY_X", "state": "STATE_Y"},
    "epi": {
        "base_transmission_prob": 0.5,
        "some_integer": 2,
        "vax_uptake": 0.1
    },
    # add other top-level groups if needed by tests
}


def deep_merge(base: dict, overrides: dict):
    """Recursively merge overrides into base (in-place)."""
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def nested_to_namespace(d):
    """Convert nested dicts to nested SimpleNamespace for attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: nested_to_namespace(v) for k, v in d.items()})
    return d


class DummyModelConfig:
    """
    Minimal stand-in ModelConfig used for tests.

    - to_dict returns a deep copy of an internal nested dict
    - copy_with applies nested overrides (like the real copy_with) and returns a new DummyModelConfig
    - to_json writes config to disk
    - has .sim attribute with .seed, .county, .state
    """

    def __init__(self, data=None):
        self._data = copy.deepcopy(data if data is not None else DEFAULT_CONFIG)
        self._ns = nested_to_namespace(self._data)
        # ensure sim exists as an attribute with expected fields
        self.sim = getattr(self._ns, "sim", SimpleNamespace(**DEFAULT_CONFIG["sim"]))

    def to_dict(self):
        return copy.deepcopy(self._data)

    def copy_with(self, overrides=None):
        newdata = copy.deepcopy(self._data)
        if overrides:
            deep_merge(newdata, overrides)
        return DummyModelConfig(newdata)

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump(self._data, fh, indent=2)


# -------------------------
# Tests for csv_to_LHS etc.
# -------------------------


def test_csv_to_LHS_file_not_found():
    # Should raise FileNotFoundError for nonexistent CSV
    import importlib
    csv_mod = importlib.import_module("scripts.variants.csv_to_lhs")
    with pytest.raises(FileNotFoundError):
        csv_mod.csv_to_LHS("this_file_does_not_exist.csv", n_samples=5)


def test_csv_to_LHS_bad_column_order(tmp_path, monkeypatch):
    # The CSV must have columns in exact order: parameter, minimum, maximum, integer
    import importlib
    csv_mod = importlib.import_module("scripts.variants.csv_to_lhs")
    # Ensure ModelConfig used inside the module is our DummyModelConfig
    monkeypatch.setattr(csv_mod, "ModelConfig", DummyModelConfig)

    # Create CSV with a different column order
    csv_path = tmp_path / "bad_cols.csv"
    df = pd.DataFrame(
        {
            "minimum": [0.1, 1],
            "parameter": ["epi.base_transmission_prob", "epi.some_integer"],
            "maximum": [0.9, 3],
            "integer": [0, 1],
        }
    )
    df.to_csv(csv_path, index=False)

    # Expect ValueError for improperly formatted CSV (wrong column order)
    with pytest.raises(ValueError):
        csv_mod.csv_to_LHS(str(csv_path), n_samples=5)


def test_csv_to_LHS_duplicate_parameter(tmp_path, monkeypatch):
    import importlib
    csv_mod = importlib.import_module("scripts.variants.csv_to_lhs")
    monkeypatch.setattr(csv_mod, "ModelConfig", DummyModelConfig)

    csv_path = tmp_path / "dupes.csv"
    df = pd.DataFrame(
        {
            "parameter": ["epi.some_integer", "epi.some_integer"],
            "minimum": [1, 1],
            "maximum": [3, 3],
            "integer": [1, 1],
        }
    )
    df.to_csv(csv_path, index=False)

    # Expect ValueError for duplicate parameters
    with pytest.raises(ValueError):
        csv_mod.csv_to_LHS(str(csv_path), n_samples=5)


def test_csv_to_LHS_min_greater_than_max(tmp_path, monkeypatch):
    import importlib
    csv_mod = importlib.import_module("scripts.variants.csv_to_lhs")
    monkeypatch.setattr(csv_mod, "ModelConfig", DummyModelConfig)

    csv_path = tmp_path / "min_gt_max.csv"
    df = pd.DataFrame(
        {
            "parameter": ["epi.base_transmission_prob"],
            "minimum": [0.9],
            "maximum": [0.1],
            "integer": [0],
        }
    )
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        csv_mod.csv_to_LHS(str(csv_path), n_samples=5)


def test_csv_to_LHS_missing_parameter_in_modelconfig(tmp_path, monkeypatch):
    import importlib
    csv_mod = importlib.import_module("scripts.variants.csv_to_lhs")
    monkeypatch.setattr(csv_mod, "ModelConfig", DummyModelConfig)

    csv_path = tmp_path / "missing_param.csv"
    df = pd.DataFrame(
        {
            "parameter": ["nonexistent.section.param"],
            "minimum": [0.0],
            "maximum": [1.0],
            "integer": [0],
        }
    )
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        csv_mod.csv_to_LHS(str(csv_path), n_samples=3)


def test_csv_to_LHS_success_int_and_float_and_output(tmp_path, monkeypatch):
    import importlib
    csv_mod = importlib.import_module("scripts.variants.csv_to_lhs")
    monkeypatch.setattr(csv_mod, "ModelConfig", DummyModelConfig)

    csv_path = tmp_path / "good.csv"
    df = pd.DataFrame(
        {
            "parameter": ["epi.base_transmission_prob", "epi.some_integer", "epi.vax_uptake"],
            "minimum": [0.1, 1, 0.2],
            "maximum": [0.9, 3, 0.2],  # vax_uptake min==max to test fixed float
            "integer": [0, 1, 0],
        }
    )
    df.to_csv(csv_path, index=False)

    outdir = tmp_path / "outdir"
    lhs_df = csv_mod.csv_to_LHS(str(csv_path), n_samples=7, output_dir=str(outdir), seed=1234)

    # Ensure shape and columns
    assert lhs_df.shape == (7, 3)
    assert list(lhs_df.columns) == ["epi.base_transmission_prob", "epi.some_integer", "epi.vax_uptake"]

    # dtype checks: integer column should be int64, float columns float64
    assert lhs_df["epi.some_integer"].dtype.kind in ("i", "u")  # integer
    assert lhs_df["epi.base_transmission_prob"].dtype == np.float64
    assert lhs_df["epi.vax_uptake"].dtype == np.float64

    # bounds checks
    assert lhs_df["epi.base_transmission_prob"].between(0.1, 0.9).all()
    assert lhs_df["epi.some_integer"].between(1, 3).all()
    # vax_uptake was min==max so all equal to that value
    assert (lhs_df["epi.vax_uptake"] == 0.2).all()

    # LHS.csv file written
    saved = outdir / "LHS.csv"
    assert saved.exists()
    saved_df = pd.read_csv(saved)
    assert saved_df.shape == lhs_df.shape


def test_csv_to_LHS_reproducible_for_given_seed(tmp_path, monkeypatch):
    import importlib
    csv_mod = importlib.import_module("scripts.variants.csv_to_lhs")
    monkeypatch.setattr(csv_mod, "ModelConfig", DummyModelConfig)

    csv_path = tmp_path / "repro.csv"
    df = pd.DataFrame(
        {
            "parameter": ["epi.base_transmission_prob", "epi.some_integer"],
            "minimum": [0.1, 1],
            "maximum": [0.9, 3],
            "integer": [0, 1],
        }
    )
    df.to_csv(csv_path, index=False)

    lhs_a = csv_mod.csv_to_LHS(str(csv_path), n_samples=10, seed=42)
    lhs_b = csv_mod.csv_to_LHS(str(csv_path), n_samples=10, seed=42)

    # DataFrames should be equal when the seed is the same
    pd.testing.assert_frame_equal(lhs_a.reset_index(drop=True), lhs_b.reset_index(drop=True))


# -------------------------
# Tests for LHS_to_cfg
# -------------------------


def test_LHS_to_cfg_success_and_overrides(monkeypatch):
    import importlib
    csv_mod = importlib.import_module("scripts.variants.csv_to_lhs")
    monkeypatch.setattr(csv_mod, "ModelConfig", DummyModelConfig)

    # Build lhs_df with two rows
    lhs_df = pd.DataFrame(
        {
            "epi.base_transmission_prob": [0.12, 0.34],
            "epi.some_integer": [2, 3],
        }
    )

    base_cfg = DummyModelConfig()
    configs = csv_mod.LHS_to_cfg(lhs_df=lhs_df, base_cfg=base_cfg)

    assert isinstance(configs, list)
    assert len(configs) == 2
    for idx, cfg in enumerate(configs):
        d = cfg.to_dict()
        assert pytest.approx(lhs_df.iloc[idx]["epi.base_transmission_prob"]) == d["epi"]["base_transmission_prob"]
        assert int(lhs_df.iloc[idx]["epi.some_integer"]) == int(d["epi"]["some_integer"])


def test_LHS_to_cfg_type_and_key_errors(monkeypatch):
    import importlib
    csv_mod = importlib.import_module("scripts.variants.csv_to_lhs")
    monkeypatch.setattr(csv_mod, "ModelConfig", DummyModelConfig)

    base_cfg = DummyModelConfig()

    # Passing non-DataFrame should raise TypeError
    with pytest.raises(TypeError):
        csv_mod.LHS_to_cfg(lhs_df=["not", "a", "df"], base_cfg=base_cfg)

    # Empty column name should raise ValueError
    lhs_df_empty_col = pd.DataFrame([[1]], columns=[""])
    with pytest.raises(ValueError):
        csv_mod.LHS_to_cfg(lhs_df=lhs_df_empty_col, base_cfg=base_cfg)

    # Unknown dotted key should raise KeyError
    lhs_df_unknown = pd.DataFrame({ "nonexistent.section.value": [1] })
    with pytest.raises(KeyError):
        csv_mod.LHS_to_cfg(lhs_df=lhs_df_unknown, base_cfg=base_cfg)


# -------------------------
# csv_to_cfg pipeline test
# -------------------------


def test_csv_to_cfg_pipeline(tmp_path, monkeypatch):
    import importlib
    csv_mod = importlib.import_module("scripts.variants.csv_to_lhs")
    monkeypatch.setattr(csv_mod, "ModelConfig", DummyModelConfig)

    csv_path = tmp_path / "pipeline.csv"
    df = pd.DataFrame(
        {
            "parameter": ["epi.base_transmission_prob", "epi.some_integer"],
            "minimum": [0.1, 1],
            "maximum": [0.9, 3],
            "integer": [0, 1],
        }
    )
    df.to_csv(csv_path, index=False)

    outdir = tmp_path / "pipeline_out"
    configs = csv_mod.csv_to_cfg(
        csv_path=str(csv_path),
        N=4,
        output_dir=str(outdir),
        default_config=DummyModelConfig(),
        seed=2025,
    )

    assert isinstance(configs, list)
    assert len(configs) == 4
    # LHS.csv should have been written
    assert (Path(outdir) / "LHS.csv").exists()


# -------------------------
# Integration test for prepare_run
# -------------------------


def test_prepare_run_integration(monkeypatch, tmp_path):
    """
    Integration-style test for prepare_run. We monkeypatch external dependencies
    used by prepare_run so the function performs orchestration correctly.
    """
    import importlib
    rvf = importlib.import_module("scripts.variants.run_variants_funcs")

    # Replace ModelConfig used by the module
    monkeypatch.setattr(rvf, "ModelConfig", DummyModelConfig)

    # Fake prepare_contacts: assert we receive county/state from base_cfg.sim.* and return a small DataFrame
    def fake_prepare_contacts(county, state, data_dir, overwrite_files, save_files):
        assert county == DEFAULT_CONFIG["sim"]["county"]
        assert state == DEFAULT_CONFIG["sim"]["state"]
        return pd.DataFrame({"person_id": [1, 2, 3]})

    monkeypatch.setattr(rvf, "prepare_contacts", fake_prepare_contacts)

    # Fake read_or_build_master -> return a tiny master edge DataFrame
    def fake_read_or_build_master(contacts_df, cfg, run_dir, seed, variant):
        # echo back a dataframe with a single "edge"
        return pd.DataFrame({"u": [0], "v": [1]})

    monkeypatch.setattr(rvf, "read_or_build_master", fake_read_or_build_master)

    # Fake build_minimal_graphdata_from_edge_list -> return a simple structure
    def fake_build_minimal_graphdata_from_edge_list(master_df, N):
        return {"master_edges": master_df.copy(), "N": N}

    monkeypatch.setattr(
        rvf,
        "build_minimal_graphdata_from_edge_list",
        fake_build_minimal_graphdata_from_edge_list,
    )

    # Fake csv_to_cfg -> return two DummyModelConfig objects
    def fake_csv_to_cfg(csv_path, N, output_dir, default_config, seed):
        assert isinstance(default_config, DummyModelConfig)
        return [DummyModelConfig(), DummyModelConfig()]

    monkeypatch.setattr(rvf, "csv_to_cfg", fake_csv_to_cfg)

    # Call prepare_run (output_dir will be created by the function)
    outdir = tmp_path / "prep_run_out"
    contacts_df, configs_list, master_gd = rvf.prepare_run(
        csv_path="ignored-by-monkeypatch.csv",
        n_samples=2,
        output_dir=str(outdir),
        base_cfg=None,  # will use the module's ModelConfig (monkeypatched)
        data_dir="data_dir_test",
        overwrite_files=True,
        save_files=False,
        seed=999,
    )

    # Validate that the function returned the monkeypatched outputs
    assert isinstance(contacts_df, pd.DataFrame)
    assert contacts_df.shape[0] == 3
    assert isinstance(configs_list, list)
    assert len(configs_list) == 2
    assert isinstance(master_gd, dict)
    assert "master_edges" in master_gd
    assert Path(outdir).exists()