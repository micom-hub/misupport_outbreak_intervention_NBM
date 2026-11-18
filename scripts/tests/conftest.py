from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from scripts.config import ModelConfig
from scripts.graph.graph_utils import GraphData
from scripts.simulation.outbreak_model import NetworkModel


@dataclass
class ToyModelBundle:
    """Reusable six-person population, network, and model factory."""

    contacts: pd.DataFrame
    edge_list: pd.DataFrame
    graphdata: GraphData
    tmp_path: Path

    def config(
        self,
        *,
        policy_name: str = "observe_only",
        transmission_probability: float = 0.0,
        epi_overrides: Optional[Dict[str, Any]] = None,
        lhd_overrides: Optional[Dict[str, Any]] = None,
        sim_overrides: Optional[Dict[str, Any]] = None,
        population_overrides: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ModelConfig:
        """Create a small, fast configuration."""
        epi = {
            "base_transmission_prob":
                float(transmission_probability),
            "latent_period": 1.0,
            "infectious_period": 8.0,
            "gamma_alpha": 1_000_000.0,
            "latent_period_vax": 1.0,
            "infectious_period_vax": 8.0,
            "vax_uptake": 0.0,
            "vax_efficacy": 0.0,
            "relative_infectiousness_vax": 1.0,
            "symptomatic_fraction": 1.0,
            "symptom_onset_delay_mean": 0.0,
            "symptom_onset_shape": 4.0,
        }
        epi.update(epi_overrides or {})

        lhd = {
            "policy_name": policy_name,
            "lhd_daily_capacity": 100.0,
            "p_detect_symptomatic_daily": 1.0,
            "p_detect_asymptomatic_daily": 0.0,
            "p_detect_imported_daily": 1.0,
            "imported_detection_start_day": 1,
            "passive_report_delay_days": 0,
            "trace_recall_prob": 1.0,
            "trace_delay_days": 1,
            "trace_max_per_case": 25,
            "test_turnaround_days": 1,
            "test_report_delay_days": 0,
            "test_sensitivity_exposed": 1.0,
            "test_sensitivity_infectious": 1.0,
            "test_specificity": 1.0,
            "lhd_default_int_reduction": 1.0,
            "lhd_default_int_duration": 3,
            "escalation_trigger_metric": "new_reports",
            "escalation_trigger_threshold": 1.0,
            "escalation_consecutive_days": 1,
            "escalation_implementation_delay_days": 1,
            "emergency_household_reduction": 1.0,
            "emergency_school_reduction": 1.0,
            "emergency_workplace_reduction": 1.0,
            "emergency_casual_reduction": 1.0,
            "emergency_group_quarter_reduction": 1.0,
            "phase2_trace_mode": "stop",
        }
        lhd.update(lhd_overrides or {})

        simulation = {
            "n_replicates": 1,
            "simulation_duration": 8,
            "I0": 1,
            "seed": 12345,
            "run_name": "toy_run",
            "county": "Toy",
            "state": "Michigan",
            "master_casual_candidates": 10,
            "overwrite_master": True,
        }
        simulation.update(sim_overrides or {})

        population = {
            "wp_contacts": 1,
            "sch_contacts": 1,
            "gq_contacts": 1,
            "cas_contacts": 1,
            "hh_weight": 1.0,
            "wp_weight": 1.0,
            "sch_weight": 1.0,
            "gq_weight": 1.0,
            "cas_weight": 1.0,
        }
        population.update(
            population_overrides or {}
        )

        return ModelConfig().copy_with(
            {
                "epi": epi,
                "lhd": lhd,
                "sim": simulation,
                "population": population,
            }
        )

    def model(
        self,
        *,
        policy_name: str = "observe_only",
        transmission_probability: float = 0.0,
        epi_overrides: Optional[Dict[str, Any]] = None,
        lhd_overrides: Optional[Dict[str, Any]] = None,
        sim_overrides: Optional[Dict[str, Any]] = None,
        seed: int = 12345,
        simulate: bool = False,
    ) -> NetworkModel:
        """Construct an in-memory NetworkModel."""
        cfg = self.config(
            policy_name=policy_name,
            transmission_probability=(
                transmission_probability
            ),
            epi_overrides=epi_overrides,
            lhd_overrides=lhd_overrides,
            sim_overrides=sim_overrides,
        )

        model = NetworkModel(
            config=cfg,
            graphdata=self.graphdata,
            run_dir=str(self.tmp_path),
            seed=seed,
        )

        if simulate:
            model.simulate()

        return model


@pytest.fixture
def toy_model_factory(
    tmp_path: Path,
) -> ToyModelBundle:
    """
    Six-node undirected network with five contact types.

        0 --hh-- 1 --wp-- 2 --sch-- 3 --cas-- 4 --gq-- 5
        |________________________________________________|
                              cas
    """
    contacts = pd.DataFrame(
        {
            "PID": [f"P{i}" for i in range(6)],
            "hh_id": ["H0", "H0", "H1", "H2", "H3", "H4"],
            "wp_id": [pd.NA, "W0", "W0", pd.NA, pd.NA, pd.NA],
            "sch_id": [pd.NA, pd.NA, "S0", "S0", pd.NA, pd.NA],
            "gq_id": [pd.NA, pd.NA, pd.NA, pd.NA, "G0", "G0"],
            "age": [20, 30, 40, 10, 70, 50],
            "sex": ["F", "M", "F", "M", "F", "M"],
            "race": ["A"] * 6,
            "relate": [pd.NA] * 6,
            "gq": [False, False, False, False, True, True],
        }
    )

    edge_list = pd.DataFrame(
        [
            (0, 1, 1.0, "hh"),
            (1, 2, 1.0, "wp"),
            (2, 3, 1.0, "sch"),
            (3, 4, 1.0, "cas"),
            (4, 5, 1.0, "gq"),
            (0, 5, 1.0, "cas"),
        ],
        columns=[
            "source",
            "target",
            "weight",
            "contact_type",
        ],
    )

    row = np.concatenate(
        [
            edge_list["source"].to_numpy(np.int32),
            edge_list["target"].to_numpy(np.int32),
        ]
    )
    column = np.concatenate(
        [
            edge_list["target"].to_numpy(np.int32),
            edge_list["source"].to_numpy(np.int32),
        ]
    )
    weights = np.ones(
        len(row),
        dtype=np.float32,
    )

    adjacency = csr_matrix(
        (weights, (row, column)),
        shape=(6, 6),
    )

    neighbor_map = {node: [] for node in range(6)}

    for edge in edge_list.itertuples(index=False):
        neighbor_map[int(edge.source)].append(
            (
                int(edge.target),
                float(edge.weight),
                str(edge.contact_type),
            )
        )
        neighbor_map[int(edge.target)].append(
            (
                int(edge.source),
                float(edge.weight),
                str(edge.contact_type),
            )
        )

    contact_types = sorted(
        edge_list["contact_type"].unique()
    )
    contact_type_to_id = {
        contact_type: index
        for index, contact_type
        in enumerate(contact_types)
    }
    id_to_contact_type = {
        index: contact_type
        for contact_type, index
        in contact_type_to_id.items()
    }

    csr_by_type = {}

    for contact_type in contact_types:
        subset = edge_list[
            edge_list["contact_type"]
            == contact_type
        ]
        type_rows = np.concatenate(
            [
                subset["source"].to_numpy(np.int32),
                subset["target"].to_numpy(np.int32),
            ]
        )
        type_columns = np.concatenate(
            [
                subset["target"].to_numpy(np.int32),
                subset["source"].to_numpy(np.int32),
            ]
        )
        type_weights = np.concatenate(
            [
                subset["weight"].to_numpy(np.float32),
                subset["weight"].to_numpy(np.float32),
            ]
        )

        type_csr = csr_matrix(
            (
                type_weights,
                (type_rows, type_columns),
            ),
            shape=(6, 6),
        )

        csr_by_type[contact_type] = (
            type_csr.indptr.astype(np.int64),
            type_csr.indices.astype(np.int32),
            type_csr.data.astype(np.float32),
        )

    graphdata = GraphData(
        N=6,
        edge_list=edge_list,
        adj_matrix=adjacency,
        individual_lookup=contacts[
            ["age", "race", "sex"]
        ].copy(),
        ages=contacts["age"].to_numpy(np.int32),
        sexes=contacts["sex"].to_numpy(),
        compliances=np.ones(6, dtype=np.float32),
        neighbor_map=neighbor_map,
        fast_neighbor_map={
            source: {
                target: (weight, contact_type)
                for target, weight, contact_type
                in neighbors
            }
            for source, neighbors
            in neighbor_map.items()
        },
        csr_by_type=csr_by_type,
        contact_types=contact_types,
        ct_to_id=contact_type_to_id,
        id_to_ct=id_to_contact_type,
        full_node_list=list(range(6)),
        degrees_arr=adjacency.getnnz(
            axis=1
        ).astype(np.int32),
    )

    return ToyModelBundle(
        contacts=contacts,
        edge_list=edge_list,
        graphdata=graphdata,
        tmp_path=tmp_path,
    )