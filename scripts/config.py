# scripts/config.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]

VALID_TRIGGER_METRICS = {
    "new_reports",
    "rolling_mean_reports",
    "rolling_report_growth",
    "backlog_to_capacity_ratio",
    "capacity_utilization",
    "maximum_queue_age",
}

VALID_INTERVENTION_PROFILES = {
    "combined_emergency",
    "stay_home",
    "school_closure",
    "workplace_mitigation",
    "gathering_restrictions",
}

VALID_PHASE2_TRACE_MODES = {
    "stop",
    "targeted",
    "continue",
}


@dataclass(frozen=True)
class EpiParams:
    """Epidemiological and natural-history parameters."""

    # Daily per-edge transmission probability
    base_transmission_prob: float = 0.25

    # E-to-I and I-to-R durations
    latent_period: float = 10.5
    infectious_period: float = 5.0

    # Gamma-distribution shape
    gamma_alpha: float = 20.0

    # Vaccinated disease durations
    latent_period_vax: float = 10.5
    infectious_period_vax: float = 5.0

    # Vaccinated source infectiousness
    relative_infectiousness_vax: float = 0.05

    # Vaccination susceptibility effect
    vax_efficacy: float = 0.70
    vax_uptake: float = 0.85

    # Age-specific susceptibility
    susceptibility_multiplier_under_five: float = 1.0
    susceptibility_multiplier_elderly: float = 1.0

    # Symptomatic infection probability
    symptomatic_fraction: float = 0.70

    # Symptoms after infectious onset
    symptom_onset_delay_mean: float = 1.0
    symptom_onset_shape: float = 4.0


@dataclass(frozen=True)
class PopulationParams:
    """Contact-network sampling parameters."""

    # Source-side contact sampling means
    wp_contacts: int = 10
    sch_contacts: int = 10
    gq_contacts: int = 10
    cas_contacts: int = 10

    # Relative transmission weights
    hh_weight: float = 1.0
    wp_weight: float = 0.5
    sch_weight: float = 0.6
    gq_weight: float = 0.3
    cas_weight: float = 0.1


@dataclass(frozen=True)
class LHDParams:
    """Surveillance, workload, policy, and intervention parameters."""

    # Active response policy
    policy_name: str = "adaptive_escalation"

    # Daily standardized work units
    lhd_daily_capacity: float = 1000.0

    # ------------------------------------------------------------------
    # Operational work costs
    # ------------------------------------------------------------------

    # Work units per task
    case_investigation_cost: float = 4.0
    isolation_support_cost: float = 1.0
    contact_interview_cost: float = 2.0
    contact_notification_cost: float = 1.0
    testing_referral_cost: float = 1.0
    automated_notification_cost: float = 0.05

    # ------------------------------------------------------------------
    # Task priorities
    # ------------------------------------------------------------------

    # Base scheduling priorities
    case_investigation_priority: float = 5.0
    isolation_support_priority: float = 4.5
    contact_interview_priority: float = 3.0
    contact_notification_priority: float = 2.5
    automated_notification_priority: float = 2.5
    testing_referral_priority: float = 2.0

    # Daily waiting-time bonus
    task_aging_priority_per_day: float = 0.25

    # ------------------------------------------------------------------
    # Task usefulness windows
    # ------------------------------------------------------------------

    # Maximum pending days
    case_investigation_deadline_days: int = 7
    isolation_support_deadline_days: int = 3
    contact_interview_deadline_days: int = 7
    contact_notification_deadline_days: int = 7
    automated_notification_deadline_days: int = 3
    testing_referral_deadline_days: int = 5

    # ------------------------------------------------------------------
    # Passive surveillance
    # ------------------------------------------------------------------

    # Daily ascertainment hazards
    p_detect_symptomatic_daily: float = 0.35
    p_detect_asymptomatic_daily: float = 0.03
    p_detect_imported_daily: float = 0.10

    # Imported-case eligibility day
    imported_detection_start_day: int = 1

    # Detection-to-report delay
    passive_report_delay_days: int = 1

    # ------------------------------------------------------------------
    # Contact tracing
    # ------------------------------------------------------------------

    # Contact recall probability
    trace_recall_prob: float = 0.40

    # Interview completion delay
    trace_delay_days: int = 1

    # Contacts returned per case
    trace_max_per_case: int = 25

    # ------------------------------------------------------------------
    # Active testing
    # ------------------------------------------------------------------

    # Test processing and reporting
    test_turnaround_days: int = 1
    test_report_delay_days: int = 0

    # Stage-specific test performance
    test_sensitivity_exposed: float = 0.50
    test_sensitivity_infectious: float = 0.99
    test_specificity: float = 1.0

    # ------------------------------------------------------------------
    # Case isolation
    # ------------------------------------------------------------------

    # Fraction of eligible contact removed
    lhd_default_int_reduction: float = 0.90

    # Isolation duration
    lhd_default_int_duration: int = 10

    # Mean node compliance; not yet applied to active policies
    mean_compliance: float = 1.0

    # ------------------------------------------------------------------
    # Escalation trigger
    # ------------------------------------------------------------------

    # Observable trigger measure
    escalation_trigger_metric: str = "new_reports"

    # Trigger cutoff
    escalation_trigger_threshold: float = 10.0

    # Rolling-measure window
    escalation_window_days: int = 3

    # Consecutive threshold days
    escalation_consecutive_days: int = 1

    # Growth-rate stability floor
    escalation_minimum_reports: int = 3

    # Order-to-activation delay
    escalation_implementation_delay_days: int = 1

    # Zero means indefinite
    escalation_duration_days: int = 0

    # Emergency intervention bundle
    escalation_intervention_profile: str = "combined_emergency"

    # ------------------------------------------------------------------
    # Emergency contact reductions
    # ------------------------------------------------------------------

    # Fractions of contacts removed
    emergency_school_reduction: float = 0.80
    emergency_workplace_reduction: float = 0.60
    emergency_casual_reduction: float = 0.70
    emergency_household_reduction: float = 0.0
    emergency_group_quarter_reduction: float = 0.0

    # ------------------------------------------------------------------
    # Phase-2 tracing
    # ------------------------------------------------------------------

    # stop, targeted, or continue
    phase2_trace_mode: str = "stop"

    # Cancel existing tracing work
    cancel_tracing_backlog_on_escalation: bool = True

    # ------------------------------------------------------------------
    # Automated notification
    # ------------------------------------------------------------------

    # Contact-reduction effect
    automated_notification_reduction: float = 0.30

    # Reduction duration
    automated_notification_duration_days: int = 5

    # ------------------------------------------------------------------
    # Perfect-information benchmark
    # ------------------------------------------------------------------

    # Full-contact reduction
    perfect_information_reduction: float = 1.0

    # Truth-based isolation duration
    perfect_information_isolation_duration_days: int = 30


@dataclass(frozen=True)
class SimulationParams:
    """Simulation, population, and reproducibility settings."""

    # Stochastic replicates
    n_replicates: int = 10

    # Output-run label
    run_name: str = field(
        default_factory=lambda: (
            "RUN_"
            + datetime.now().strftime(
                "%m-%d-%Y_%H-%M-%S"
            )
        )
    )

    # Maximum simulated days
    simulation_duration: int = 100

    # Imported infectious cases
    I0: int = 2

    # Base random seed
    seed: int = 2026

    # Synthetic population location to use
    county: str = "Keweenaw"
    state: str = "Michigan"

    # Casual-contact candidate pool
    master_casual_candidates: int = 100

    # Rebuild cached master graph
    # (False saves compute)
    overwrite_master: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Top-level immutable model configuration."""

    epi: EpiParams = field(
        default_factory=EpiParams
    )
    population: PopulationParams = field(
        default_factory=PopulationParams
    )
    lhd: LHDParams = field(
        default_factory=LHDParams
    )
    sim: SimulationParams = field(
        default_factory=SimulationParams
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionaries."""
        return {
            "epi": asdict(self.epi),
            "population": asdict(
                self.population
            ),
            "lhd": asdict(self.lhd),
            "sim": asdict(self.sim),
        }

    def to_json(
        self,
        path: PathLike,
    ) -> None:
        """Write configuration JSON."""
        path = Path(path).expanduser()
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(path, "w") as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
                default=str,
            )

    @classmethod
    def from_json(
        cls,
        path: PathLike,
    ) -> "ModelConfig":
        """Load configuration JSON."""
        with open(
            Path(path).expanduser(),
            "r",
        ) as file:
            data = json.load(file)

        return cls.from_nested_dict(data)

    @classmethod
    def from_nested_dict(
        cls,
        nested: Dict[str, Any],
    ) -> "ModelConfig":
        """Build from nested dictionaries."""
        return cls(
            epi=EpiParams(
                **nested.get("epi", {})
            ),
            population=PopulationParams(
                **nested.get(
                    "population",
                    {},
                )
            ),
            lhd=LHDParams(
                **nested.get("lhd", {})
            ),
            sim=SimulationParams(
                **nested.get("sim", {})
            ),
        )

    def copy_with(
        self,
        overrides: Optional[
            Dict[str, Dict[str, Any]]
        ] = None,
    ) -> "ModelConfig":
        """Return a copy with section overrides."""
        if not overrides:
            return self

        return ModelConfig(
            epi=replace(
                self.epi,
                **overrides.get("epi", {}),
            ),
            population=replace(
                self.population,
                **overrides.get(
                    "population",
                    {},
                ),
            ),
            lhd=replace(
                self.lhd,
                **overrides.get("lhd", {}),
            ),
            sim=replace(
                self.sim,
                **overrides.get("sim", {}),
            ),
        )

    def validate(self) -> None:
        """Validate model configuration."""
        self._validate_epi()
        self._validate_population()
        self._validate_lhd()
        self._validate_simulation()

    def _validate_epi(self) -> None:
        """Validate epidemiological parameters."""
        self._require_probability(
            "epi.base_transmission_prob",
            self.epi.base_transmission_prob,
        )
        self._require_probability(
            "epi.vax_efficacy",
            self.epi.vax_efficacy,
        )
        self._require_probability(
            "epi.vax_uptake",
            self.epi.vax_uptake,
        )
        self._require_probability(
            "epi.symptomatic_fraction",
            self.epi.symptomatic_fraction,
        )

        self._require_positive(
            "epi.latent_period",
            self.epi.latent_period,
        )
        self._require_positive(
            "epi.infectious_period",
            self.epi.infectious_period,
        )
        self._require_positive(
            "epi.latent_period_vax",
            self.epi.latent_period_vax,
        )
        self._require_positive(
            "epi.infectious_period_vax",
            self.epi.infectious_period_vax,
        )
        self._require_positive(
            "epi.gamma_alpha",
            self.epi.gamma_alpha,
        )
        self._require_positive(
            "epi.symptom_onset_shape",
            self.epi.symptom_onset_shape,
        )

        self._require_nonnegative(
            "epi.symptom_onset_delay_mean",
            self.epi.symptom_onset_delay_mean,
        )
        self._require_nonnegative(
            "epi.relative_infectiousness_vax",
            self.epi.relative_infectiousness_vax,
        )
        self._require_nonnegative(
            "epi.susceptibility_multiplier_under_five",
            self.epi.susceptibility_multiplier_under_five,
        )
        self._require_nonnegative(
            "epi.susceptibility_multiplier_elderly",
            self.epi.susceptibility_multiplier_elderly,
        )

    def _validate_population(self) -> None:
        """Validate network parameters."""
        contact_counts = {
            "population.wp_contacts":
                self.population.wp_contacts,
            "population.sch_contacts":
                self.population.sch_contacts,
            "population.gq_contacts":
                self.population.gq_contacts,
            "population.cas_contacts":
                self.population.cas_contacts,
        }

        for name, value in contact_counts.items():
            self._require_nonnegative(
                name,
                value,
            )

        contact_weights = {
            "population.hh_weight":
                self.population.hh_weight,
            "population.wp_weight":
                self.population.wp_weight,
            "population.sch_weight":
                self.population.sch_weight,
            "population.gq_weight":
                self.population.gq_weight,
            "population.cas_weight":
                self.population.cas_weight,
        }

        for name, value in contact_weights.items():
            self._require_nonnegative(
                name,
                value,
            )

    def _validate_lhd(self) -> None:
        """Validate surveillance and response parameters."""
        self._require_nonnegative(
            "lhd.lhd_daily_capacity",
            self.lhd.lhd_daily_capacity,
        )

        probabilities = {
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
            "lhd.lhd_default_int_reduction":
                self.lhd.lhd_default_int_reduction,
            "lhd.automated_notification_reduction":
                self.lhd.automated_notification_reduction,
            "lhd.perfect_information_reduction":
                self.lhd.perfect_information_reduction,
            "lhd.emergency_school_reduction":
                self.lhd.emergency_school_reduction,
            "lhd.emergency_workplace_reduction":
                self.lhd.emergency_workplace_reduction,
            "lhd.emergency_casual_reduction":
                self.lhd.emergency_casual_reduction,
            "lhd.emergency_household_reduction":
                self.lhd.emergency_household_reduction,
            "lhd.emergency_group_quarter_reduction":
                self.lhd.emergency_group_quarter_reduction,
            "lhd.mean_compliance":
                self.lhd.mean_compliance,
        }

        for name, value in probabilities.items():
            self._require_probability(
                name,
                value,
            )

        nonnegative_fields = {
            "lhd.case_investigation_cost":
                self.lhd.case_investigation_cost,
            "lhd.isolation_support_cost":
                self.lhd.isolation_support_cost,
            "lhd.contact_interview_cost":
                self.lhd.contact_interview_cost,
            "lhd.contact_notification_cost":
                self.lhd.contact_notification_cost,
            "lhd.testing_referral_cost":
                self.lhd.testing_referral_cost,
            "lhd.automated_notification_cost":
                self.lhd.automated_notification_cost,
            "lhd.task_aging_priority_per_day":
                self.lhd.task_aging_priority_per_day,
            "lhd.imported_detection_start_day":
                self.lhd.imported_detection_start_day,
            "lhd.passive_report_delay_days":
                self.lhd.passive_report_delay_days,
            "lhd.trace_delay_days":
                self.lhd.trace_delay_days,
            "lhd.trace_max_per_case":
                self.lhd.trace_max_per_case,
            "lhd.test_turnaround_days":
                self.lhd.test_turnaround_days,
            "lhd.test_report_delay_days":
                self.lhd.test_report_delay_days,
            "lhd.lhd_default_int_duration":
                self.lhd.lhd_default_int_duration,
            "lhd.escalation_trigger_threshold":
                self.lhd.escalation_trigger_threshold,
            "lhd.escalation_window_days":
                self.lhd.escalation_window_days,
            "lhd.escalation_consecutive_days":
                self.lhd.escalation_consecutive_days,
            "lhd.escalation_minimum_reports":
                self.lhd.escalation_minimum_reports,
            "lhd.escalation_implementation_delay_days":
                self.lhd.escalation_implementation_delay_days,
            "lhd.escalation_duration_days":
                self.lhd.escalation_duration_days,
            "lhd.automated_notification_duration_days":
                self.lhd.automated_notification_duration_days,
            "lhd.perfect_information_isolation_duration_days":
                self.lhd.perfect_information_isolation_duration_days,
        }

        for name, value in nonnegative_fields.items():
            self._require_nonnegative(
                name,
                value,
            )

        from scripts.lhd.policy_catalog import POLICIES

        if self.lhd.policy_name not in POLICIES:
            raise ValueError(
                f"Unknown LHD policy: "
                f"{self.lhd.policy_name}"
            )

        if (
            self.lhd.escalation_trigger_metric
            not in VALID_TRIGGER_METRICS
        ):
            raise ValueError(
                "Unsupported escalation trigger: "
                f"{self.lhd.escalation_trigger_metric}"
            )

        if (
            self.lhd.escalation_intervention_profile
            not in VALID_INTERVENTION_PROFILES
        ):
            raise ValueError(
                "Unsupported intervention profile: "
                f"{self.lhd.escalation_intervention_profile}"
            )

        if (
            self.lhd.phase2_trace_mode
            not in VALID_PHASE2_TRACE_MODES
        ):
            raise ValueError(
                "Unsupported Phase-2 trace mode: "
                f"{self.lhd.phase2_trace_mode}"
            )

    def _validate_simulation(self) -> None:
        """Validate simulation settings."""
        if (
            not isinstance(self.sim.I0, int)
            or isinstance(self.sim.I0, bool)
        ):
            raise ValueError(
                "sim.I0 must be an integer"
            )

        self._require_nonnegative(
            "sim.I0",
            self.sim.I0,
        )
        self._require_positive(
            "sim.n_replicates",
            self.sim.n_replicates,
        )
        self._require_positive(
            "sim.simulation_duration",
            self.sim.simulation_duration,
        )

        if (
            self.sim.master_casual_candidates
            < self.population.cas_contacts
        ):
            raise ValueError(
                "sim.master_casual_candidates "
                "must be at least "
                "population.cas_contacts"
            )

    @staticmethod
    def _require_probability(
        name: str,
        value: float,
    ) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{name} must be in [0, 1]"
            )

    @staticmethod
    def _require_positive(
        name: str,
        value: float,
    ) -> None:
        if float(value) <= 0.0:
            raise ValueError(
                f"{name} must be > 0"
            )

    @staticmethod
    def _require_nonnegative(
        name: str,
        value: float,
    ) -> None:
        if float(value) < 0.0:
            raise ValueError(
                f"{name} must be >= 0"
            )