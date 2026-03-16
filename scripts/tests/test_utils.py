import os
import numpy as np
import pandas as pd
import pytest

# import the synth module (try both likely locations)
try:
    from scripts.utils.synth_data_processing import synthetic_data_process, build_individual_lookup, build_edge_list
    from scripts.utils.fred_fetch import downloadPopData

except Exception:
    from scripts.synth_data_process import synthetic_data_process, build_individual_lookup, build_edge_list  

from scripts.config import ModelConfig


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """
    Create a temporary working directory with a data/<county>/ containing small
    people.txt and gq_people.txt files in the format expected by synthetic_data_process.
    The fixture changes cwd to tmp_path for the duration of the test and returns the county name.
    """
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    county = "TestCounty"
    county_dir = data_dir / county
    county_dir.mkdir()

    # Create a simple people.txt with tab-separated columns
    people_txt = county_dir / "people.txt"
    people_rows = [
        # header
        "sp_id\tsp_hh_id\tschool_id\twork_id\tage\tsex\trelate\trace",
        # two household members (0,1), one with school/work, others without
        "0\tH1\tS1\tW1\t34\tM\thead\twhite",
        "1\tH1\tS1\tW1\t6\tF\tchild\twhite",
        # one elder in different household
        "2\tH2\tX\tW2\t70\tM\thead\tblack",
        # one single worker
        "3\tH3\tX\tW3\t25\tF\tworker\tasian",
    ]
    people_txt.write_text("\n".join(people_rows))

    # Create a simple gq_people.txt with a single group-quarter resident
    gq_txt = county_dir / "gq_people.txt"
    gq_rows = [
        "sp_id\tsp_gq_id\tage\tsex",
        "4\tGQ1\t45\tM"
    ]
    gq_txt.write_text("\n".join(gq_rows))

    yield str(county)

    # cleanup happens automatically when tmp_path fixture is removed


def test_synthetic_data_process_reads_and_merges(tmp_data_dir):
    county = tmp_data_dir
    # call code under test - do not save files to disk from the function
    contacts = synthetic_data_process(county, save_files=False)

    # assert output is a DataFrame with expected columns
    expected_cols = ["PID", "hh_id", "wp_id", "sch_id", "gq_id", "age", "sex", "race", "relate", "gq"]
    assert isinstance(contacts, pd.DataFrame)
    for c in expected_cols:
        assert c in contacts.columns

    # check number of rows: people (4) + gq (1) = 5
    assert contacts.shape[0] == 5

    # find the gq row (gq True)
    gq_rows = contacts[contacts["gq"]]
    assert len(gq_rows) == 1
    assert gq_rows.iloc[0]["PID"] == "4"


def test_build_individual_lookup(tmp_data_dir):
    county = tmp_data_dir
    contacts = synthetic_data_process(county, save_files=False)
    lookup = build_individual_lookup(contacts)
    # keys should be age, race, sex
    assert list(lookup.columns) == ["age", "race", "sex"]
    assert len(lookup) == len(contacts)


def test_factorize_and_build_edge_list_small(tmp_data_dir):
    # Build a small contacts_df manually (similar to synthetic_data_process output)
    contacts = pd.DataFrame([
        {"PID": "0", "hh_id": "H1", "wp_id": "W1", "sch_id": "S1", "gq_id": np.nan, "age": 34, "sex": "M", "race": "white", "relate": "head", "gq": False},
        {"PID": "1", "hh_id": "H1", "wp_id": "W1", "sch_id": "S1", "gq_id": np.nan, "age": 6, "sex": "F", "race": "white", "relate": "child", "gq": False},
        {"PID": "2", "hh_id": "H2", "wp_id": "W2", "sch_id": "X",  "gq_id": np.nan, "age": 70, "sex": "M", "race": "black", "relate": "head", "gq": False},
        {"PID": "3", "hh_id": "H3", "wp_id": "X",  "sch_id": "X",  "gq_id": np.nan, "age": 25, "sex": "F", "race": "asian", "relate": "worker", "gq": False},
    ])
    cfg = ModelConfig()
    # produce edges
    edges_df = build_edge_list(contacts.copy(), cfg, seed=123, save=False, county=None, master_casual_contacts=2)

    # schema checks
    assert set(edges_df.columns) == {"source", "target", "weight", "contact_type"}
    assert edges_df["source"].dtype == np.int32
    assert edges_df["target"].dtype == np.int32

    # hh pair (0,1) should exist as hh contact
    hh_row = edges_df[(edges_df["source"] == 0) & (edges_df["target"] == 1)]
    assert not hh_row.empty
    assert hh_row.iloc[0]["contact_type"] == "hh"

    # no self edges
    assert not ((edges_df["source"] == edges_df["target"]).any())

    # All pairs unique by (source,target)
    assert edges_df.shape[0] == len(edges_df[["source", "target"]].drop_duplicates())


# @pytest.mark.skipif(not os.environ.get("RUN_SELENIUM_TESTS"), reason="Selenium tests disabled by default")
def test_downloadPopData_integration():
    # Integration test: requires ChromeDriver & network access
    state = "Michigan"
    county = "Keweenaw"
    outzip = downloadPopData(state, county, os.getcwd())
    assert os.path.isfile(outzip)
    assert outzip.endswith(".zip")
    os.remove(outzip)