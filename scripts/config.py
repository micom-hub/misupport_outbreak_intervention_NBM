from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EpiParams:
    # Transmission
    base_transmission_prob: float = 0.25

    # E -> I duration. This is a latent period, not symptom incubation.
    latent_period: float = 10.5
    infectious_period: float = 5.0
    gamma_alpha: float = 20.0

    # Vaccinated natural-history parameters
    latent_period_vax: float = 10.5
    infectious_period_vax: float = 5.0
    relative_infectiousness_vax: float = 0.05

    # Vaccination
    vax_efficacy: float = 0.7
    vax_uptake: float = 0.85

    # Age-specific susceptibility
    susceptibility_multiplier_under_five: float = 1.0
    susceptibility_multiplier_elderly: float = 1.0

    # Symptom natural history
    symptomatic_fraction: float = 0.70

    # Symptoms begin this many days after infectious onset. Individual delays
    # are gamma distributed using symptom_onset_shape.
    symptom_onset_delay_mean: float = 1.0
    symptom_onset_shape: float = 4.0


@dataclass(frozen=True)
class PopulationParams:
    wp_contacts: int = 10
    sch_contacts: int = 10
    gq_contacts: int = 10
    cas_contacts: int = 10

    hh_weight: float = 1.0
    wp_weight: float = 0.5
    sch_weight: float = 0.6
    gq_weight: float = 0.3
    cas_weight: float = 0.1


@dataclass(frozen=True)
class LHDParams:
    policy_name: str = "trace_and_test"
    # Standardized operational work units available per day.
    lhd_daily_capacity: float = 1000.0

    # ------------------------------------------------------------------
    # Standardized work-unit costs
    # ------------------------------------------------------------------
    case_investigation_cost: float = 4.0
    isolation_support_cost: float = 1.0
    contact_interview_cost: float = 2.0
    contact_notification_cost: float = 1.0
    testing_referral_cost: float = 1.0
    automated_notification_cost: float = 0.05
    community_mitigation_cost: float = 20.0

    # ------------------------------------------------------------------
    # Base task priorities
    # ------------------------------------------------------------------
    case_investigation_priority: float = 5.0
    isolation_support_priority: float = 4.5
    contact_interview_priority: float = 3.0
    contact_notification_priority: float = 2.5
    testing_referral_priority: float = 2.0

    # Added to effective priority for every day a task waits.
    task_aging_priority_per_day: float = 0.25

    # ------------------------------------------------------------------
    # Task usefulness windows
    # ------------------------------------------------------------------
    case_investigation_deadline_days: int = 7
    isolation_support_deadline_days: int = 3
    contact_interview_deadline_days: int = 7
    contact_notification_deadline_days: int = 7
    testing_referral_deadline_days: int = 5

    # Retained for later operational development.
    mean_compliance: float = 1.0

    lhd_default_int_reduction: float = 0.9
    lhd_default_int_duration: int = 10

    # Daily passive ascertainment hazards while infectious and undetected.
    p_detect_symptomatic_daily: float = 0.35
    p_detect_asymptomatic_daily: float = 0.03

    # Additional daily ascertainment channel for imported infectious cases.
    # Imported cases can also be found through symptom-dependent passive
    # ascertainment. This represents external notification, anomaly detection,
    # clinician concern, travel-associated investigation, etc.
    p_detect_imported_daily: float = 0.10
    imported_detection_start_day: int = 1

    # Delay between passive detection and receipt of the report by the LHD.
    passive_report_delay_days: int = 1

    # Contact tracing
    trace_recall_prob: float = 0.4
    trace_delay_days: int = 1
    trace_max_per_case: int = 25

    # Active testing
    test_turnaround_days: int = 1
    test_report_delay_days: int = 0
    test_sensitivity_exposed: float = 0.50
    test_sensitivity_infectious: float = 0.99
    test_specificity: float = 1.0


@dataclass(frozen=True)
class SimulationParams:
    n_replicates: int = 10

    run_name: str = field(
        default_factory=lambda: (
            "RUN_" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        )
    )

    overwrite_master: bool = True
    simulation_duration: int = 100

    # Number of imported infectious cases. Imported cases are sampled separately
    # for each stochastic replicate and are not counted as local incidence.
    I0: int = 2

    seed: int = 2026
    county: str = "Keweenaw"
    state: str = "Michigan"

    master_casual_candidates: int = 100
    save_master: bool = True
    save_plots: bool = True
    save_data_files: bool = True
    make_movie: bool = False
    display_plots: bool = False
    save_run_timeseries: bool = True

    # Detailed outputs
    save_event_log: bool = True
    save_daily_epi: bool = True


@dataclass(frozen=True)
class ModelConfig:
    epi: EpiParams = field(default_factory=EpiParams)
    population: PopulationParams = field(default_factory=PopulationParams)
    lhd: LHDParams = field(default_factory=LHDParams)
    sim: SimulationParams = field(default_factory=SimulationParams)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epi": asdict(self.epi),
            "population": asdict(self.population),
            "lhd": asdict(self.lhd),
            "sim": asdict(self.sim),
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls.from_nested_dict(data)

    @classmethod
    def from_nested_dict(cls, nested: Dict[str, Any]) -> "ModelConfig":
        epi = EpiParams(**nested.get("epi", {}))
        population = PopulationParams(**nested.get("population", {}))
        lhd = LHDParams(**nested.get("lhd", {}))
        sim = SimulationParams(**nested.get("sim", {}))

        return cls(
            epi=epi,
            population=population,
            lhd=lhd,
            sim=sim,
        )

    def copy_with(
        self,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "ModelConfig":
        if not overrides:
            return self

        new_epi = self.epi
        new_population = self.population
        new_lhd = self.lhd
        new_sim = self.sim

        if "epi" in overrides:
            new_epi = replace(self.epi, **overrides["epi"])

        if "population" in overrides:
            new_population = replace(
                self.population,
                **overrides["population"],
            )

        if "lhd" in overrides:
            new_lhd = replace(self.lhd, **overrides["lhd"])

        if "sim" in overrides:
            new_sim = replace(self.sim, **overrides["sim"])

        return ModelConfig(
            epi=new_epi,
            population=new_population,
            lhd=new_lhd,
            sim=new_sim,
        )

    def validate(self) -> None:
        if not (0.0 <= self.epi.base_transmission_prob <= 1.0):
            raise ValueError(
                "epi.base_transmission_prob must be in [0, 1]"
            )

        if not (0.0 <= self.epi.vax_uptake <= 1.0):
            raise ValueError("epi.vax_uptake must be in [0, 1]")

        if not (0.0 <= self.epi.symptomatic_fraction <= 1.0):
            raise ValueError(
                "epi.symptomatic_fraction must be in [0, 1]"
            )

        if self.epi.latent_period <= 0:
            raise ValueError("epi.latent_period must be > 0")

        if self.epi.infectious_period <= 0:
            raise ValueError("epi.infectious_period must be > 0")

        if self.epi.gamma_alpha <= 0:
            raise ValueError("epi.gamma_alpha must be > 0")

        if self.epi.symptom_onset_delay_mean < 0:
            raise ValueError(
                "epi.symptom_onset_delay_mean must be >= 0"
            )

        if self.epi.symptom_onset_shape <= 0:
            raise ValueError("epi.symptom_onset_shape must be > 0")

        if self.sim.n_replicates < 1:
            raise ValueError("sim.n_replicates must be >= 1")

        if not isinstance(self.sim.I0, int) or isinstance(self.sim.I0, bool):
            raise ValueError(
                "sim.I0 must be an integer count of imported cases"
            )

        if self.sim.I0 < 0:
            raise ValueError("sim.I0 must be >= 0")

        if self.lhd.lhd_daily_capacity < 0:
            raise ValueError(
                "lhd.lhd_daily_capacity must be >= 0"
            )

        if (
            self.sim.master_casual_candidates
            < self.population.cas_contacts
        ):
            raise ValueError(
                "sim.master_casual_candidates must be at least "
                "population.cas_contacts"
            )

        probability_fields = {
            "lhd.p_detect_symptomatic_daily":
                self.lhd.p_detect_symptomatic_daily,
            "lhd.p_detect_asymptomatic_daily":
                self.lhd.p_detect_asymptomatic_daily,
            "lhd.p_detect_imported_daily":
                self.lhd.p_detect_imported_daily,
            "lhd.trace_recall_prob":
                self.lhd.trace_recall_prob,
            "lhd.test_sensitivity_exposed":
                self.lhd.test_sensitivity_exposed,
            "lhd.test_sensitivity_infectious":
                self.lhd.test_sensitivity_infectious,
            "lhd.test_specificity":
                self.lhd.test_specificity,
        }

        for name, value in probability_fields.items():
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")

        from scripts.lhd.policy_catalog import POLICIES

        if self.lhd.policy_name not in POLICIES:
            raise ValueError(
                f"lhd.policy_name '{self.lhd.policy_name}' was not "
                "found in policy_catalog.POLICIES"
            )