# scripts/config.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]


# Metrics available to the adaptive-escalation policy.
VALID_TRIGGER_METRICS = {
    "new_reports",
    "rolling_mean_reports",
    "rolling_report_growth",
    "backlog_to_capacity_ratio",
    "capacity_utilization",
    "maximum_queue_age",
}

# Population-level intervention bundles.
VALID_INTERVENTION_PROFILES = {
    "combined_emergency",
    "stay_home",
    "school_closure",
    "workplace_mitigation",
    "gathering_restrictions",
}

# Contact-tracing behavior after escalation.
VALID_PHASE2_TRACE_MODES = {
    "stop",
    "targeted",
    "continue",
}


@dataclass(frozen=True)
class EpiParams:
    """
    Epidemic transmission and natural-history parameters.

    Time values are measured in simulation days. Probabilities and fractions
    use the closed interval [0, 1]. Multipliers are dimensionless ratios.
    """

    # ------------------------------------------------------------------
    # Transmission
    # ------------------------------------------------------------------

    # Baseline daily transmission probability for an infectious-susceptible
    # edge with weight 1.0 and no vaccination, age, or intervention effects.
    # Increase for more transmissible pathogens; decrease for weaker spread.
    # Unit: probability per eligible edge per day.
    base_transmission_prob: float = 0.25

    # ------------------------------------------------------------------
    # Disease durations
    # ------------------------------------------------------------------

    # Mean time from local exposure to infectiousness for an unvaccinated
    # person. The E state is noninfectious in the current model.
    # Unit: days.
    latent_period: float = 10.5

    # Mean time spent infectious before recovery for an unvaccinated person.
    # Unit: days.
    infectious_period: float = 5.0

    # Shared gamma-distribution shape for latent and infectious durations.
    # Larger values produce less individual variation around the mean;
    # smaller values produce more variation. Dimensionless and > 0.
    gamma_alpha: float = 20.0

    # Mean E-to-I duration for vaccinated infections.
    # Set equal to latent_period if vaccination should not affect timing.
    # Unit: days.
    latent_period_vax: float = 10.5

    # Mean infectious duration for vaccinated infections.
    # Set equal to infectious_period if vaccination should not affect timing.
    # Unit: days.
    infectious_period_vax: float = 5.0

    # Infectiousness of a vaccinated infectious source relative to an
    # unvaccinated infectious source. For example, 0.05 means 5% as
    # infectious; 1.0 means no reduction. Dimensionless multiplier.
    relative_infectiousness_vax: float = 0.05

    # ------------------------------------------------------------------
    # Vaccination
    # ------------------------------------------------------------------

    # Fraction of susceptibility removed by vaccination. For example, 0.70
    # multiplies vaccinated susceptibility by 0.30. This is not a direct
    # probability of avoiding infection over the full outbreak.
    # Unit: fraction in [0, 1].
    vax_efficacy: float = 0.70

    # Fraction of the population assigned vaccinated status independently at
    # the start of each replicate.
    # Unit: population fraction in [0, 1].
    vax_uptake: float = 0.85

    # ------------------------------------------------------------------
    # Age-specific susceptibility
    # ------------------------------------------------------------------

    # Susceptibility multiplier for targets age 5 or younger.
    # 1.0 means no age effect; 0.5 means half the baseline susceptibility;
    # values above 1.0 increase susceptibility.
    # Unit: dimensionless multiplier.
    susceptibility_multiplier_under_five: float = 1.0

    # Susceptibility multiplier for targets age 65 or older.
    # Unit: dimensionless multiplier.
    susceptibility_multiplier_elderly: float = 1.0

    # ------------------------------------------------------------------
    # Symptoms
    # ------------------------------------------------------------------

    # Fraction of infections assigned a symptomatic disposition.
    # Remaining infections never develop symptoms.
    # Unit: fraction of infected people in [0, 1].
    symptomatic_fraction: float = 0.70

    # Mean delay from infectious onset to symptom onset among people assigned
    # a symptomatic disposition. Symptoms cannot precede infectiousness in
    # the current disease structure.
    # Unit: days.
    symptom_onset_delay_mean: float = 1.0

    # Gamma-distribution shape for symptom-onset delays. Larger values produce
    # less variation around symptom_onset_delay_mean.
    # Unit: dimensionless shape parameter, > 0.
    symptom_onset_shape: float = 4.0


@dataclass(frozen=True)
class PopulationParams:
    """
    Contact-network sampling and transmission-weight parameters.

    Contact counts control source-side sampling from a master candidate graph.
    They are not guaranteed to equal realized mean degree because selections
    from both endpoints are combined and duplicate edges are removed.
    """

    # ------------------------------------------------------------------
    # Contact sampling
    # ------------------------------------------------------------------

    # Mean source-side workplace contacts requested per node when sampling a
    # run graph. Realized workplace degree may differ.
    # Unit: contacts per person.
    wp_contacts: int = 10

    # Mean source-side school contacts requested per node.
    # Unit: contacts per person.
    sch_contacts: int = 10

    # Mean source-side group-quarter contacts requested per node.
    # Unit: contacts per person.
    gq_contacts: int = 10

    # Mean source-side casual contacts requested per node.
    # Must not exceed sim.master_casual_candidates.
    # Unit: contacts per person.
    cas_contacts: int = 10

    # ------------------------------------------------------------------
    # Contact-type transmission weights
    # ------------------------------------------------------------------

    # Household edge weight. A value of 1.0 leaves the baseline per-edge
    # transmission probability unchanged.
    # Unit: dimensionless transmission multiplier.
    hh_weight: float = 1.0

    # Workplace edge transmission multiplier.
    # Unit: dimensionless.
    wp_weight: float = 0.5

    # School edge transmission multiplier.
    # Unit: dimensionless.
    sch_weight: float = 0.6

    # Group-quarter edge transmission multiplier.
    # Unit: dimensionless.
    gq_weight: float = 0.3

    # Casual edge transmission multiplier.
    # Unit: dimensionless.
    cas_weight: float = 0.1


@dataclass(frozen=True)
class LHDParams:
    """
    Surveillance, workload, scheduling, and intervention parameters.

    Capacity and task costs use standardized work units rather than staff-hours.
    Their relative values determine throughput and prioritization. They have not
    yet been translated into specific staffing categories or clock time.
    """

    # ------------------------------------------------------------------
    # Policy and capacity
    # ------------------------------------------------------------------

    # Name of the active response policy in lhd/policy_catalog.py.
    # Common options: observe_only, case_isolation_only,
    # trace_centered_response, immediate_scalable_mitigation,
    # adaptive_escalation, and perfect_information.
    policy_name: str = "adaptive_escalation"

    # Total work the LHD can complete per simulation day. Increase to reduce
    # backlog and waiting times; decrease to induce operational constraint.
    # Unit: standardized work units per day.
    lhd_daily_capacity: float = 1000.0

    # ------------------------------------------------------------------
    # Operational work costs
    # ------------------------------------------------------------------

    # Cost to investigate one reported case. Completion can generate
    # isolation-support and contact-interview tasks, depending on policy.
    # Unit: work units per case investigation.
    case_investigation_cost: float = 4.0

    # Cost to provide isolation recommendation or support to one case.
    # Unit: work units per person.
    isolation_support_cost: float = 1.0

    # Cost to interview one case and elicit recalled contacts.
    # Unit: work units per case interview.
    contact_interview_cost: float = 2.0

    # Cost to directly notify one elicited contact.
    # Unit: work units per notified contact.
    contact_notification_cost: float = 1.0

    # Cost to refer one contact for testing.
    # Unit: work units per testing referral.
    testing_referral_cost: float = 1.0

    # Cost to send one automated notification. This is intentionally much
    # lower than direct notification to represent a scalable channel.
    # Unit: work units per automated notification.
    automated_notification_cost: float = 0.05

    # ------------------------------------------------------------------
    # Scheduler priorities
    # ------------------------------------------------------------------

    # Base task scores used by the scheduler. Higher values increase the
    # chance of selection. The scheduler also considers task cost and age.
    # Unit: dimensionless priority score.
    case_investigation_priority: float = 5.0
    isolation_support_priority: float = 4.5
    contact_interview_priority: float = 3.0
    contact_notification_priority: float = 2.5
    automated_notification_priority: float = 2.5
    testing_referral_priority: float = 2.0

    # Priority added for each day a task waits. Increase to favor old tasks
    # more strongly and reduce indefinite starvation.
    # Unit: priority-score increase per waiting day.
    task_aging_priority_per_day: float = 0.25

    # ------------------------------------------------------------------
    # Task usefulness windows
    # ------------------------------------------------------------------

    # Maximum age of each pending task before it expires. A task created on
    # day t has deadline t + configured value and expires after that deadline.
    # Unit: days after task creation.
    case_investigation_deadline_days: int = 7
    isolation_support_deadline_days: int = 3
    contact_interview_deadline_days: int = 7
    contact_notification_deadline_days: int = 7
    automated_notification_deadline_days: int = 3
    testing_referral_deadline_days: int = 5

    # ------------------------------------------------------------------
    # Passive surveillance
    # ------------------------------------------------------------------

    # Daily probability that an infectious, currently symptomatic, undetected
    # person is passively detected. Repeated daily opportunities mean eventual
    # detection probability is greater than this single-day value.
    # Unit: probability per eligible person per day.
    p_detect_symptomatic_daily: float = 0.35

    # Daily probability that an infectious, not-currently-symptomatic,
    # undetected person is passively detected.
    # Unit: probability per eligible person per day.
    p_detect_asymptomatic_daily: float = 0.03

    # Additional daily detection probability for imported infectious cases.
    # This is applied in addition to symptom-dependent passive surveillance
    # and can represent external notification or anomaly detection.
    # Unit: probability per eligible imported case per day.
    p_detect_imported_daily: float = 0.10

    # First simulation day on which imported cases become eligible for the
    # additional imported-case ascertainment channel.
    # Unit: simulation day index; day 0 is initialization.
    imported_detection_start_day: int = 1

    # Delay from passive detection to receipt of the case report by the LHD.
    # The LHD ordinarily cannot act before report receipt.
    # Unit: days.
    passive_report_delay_days: int = 1

    # ------------------------------------------------------------------
    # Contact tracing
    # ------------------------------------------------------------------

    # Probability that each eligible static-network neighbor is recalled
    # during a completed contact interview.
    # Unit: probability per eligible contact in [0, 1].
    trace_recall_prob: float = 0.40

    # Delay from ordering a contact interview to trace completion and contact
    # elicitation. Runtime logic enforces at least one day.
    # Unit: days.
    trace_delay_days: int = 1

    # Maximum recalled contacts returned for one interviewed case. Set to 0
    # only if intentionally requesting no cap under the current implementation.
    # Unit: contacts per completed case interview.
    trace_max_per_case: int = 25

    # ------------------------------------------------------------------
    # Active testing
    # ------------------------------------------------------------------

    # Delay from testing referral/order to test completion. Test performance
    # is evaluated using disease state at completion.
    # Unit: days.
    test_turnaround_days: int = 1

    # Additional delay from a positive test result to LHD report receipt.
    # Unit: days.
    test_report_delay_days: int = 0

    # Probability of a positive test when the tested person is exposed but
    # not yet infectious at test completion.
    # Unit: probability in [0, 1].
    test_sensitivity_exposed: float = 0.50

    # Probability of a positive test when the person is infectious at test
    # completion.
    # Unit: probability in [0, 1].
    test_sensitivity_infectious: float = 0.99

    # Probability of a negative result for a person who is neither exposed nor
    # infectious at test completion. One minus this value is false positivity.
    # Unit: probability in [0, 1].
    test_specificity: float = 1.0

    # ------------------------------------------------------------------
    # Case isolation
    # ------------------------------------------------------------------

    # Fraction of eligible school, workplace, and casual contact removed by
    # case isolation. Household and group-quarter contacts are unchanged by
    # the default isolation task.
    #
    # The model multiplies both incoming and outgoing contact multipliers.
    # Effects can therefore compound when both endpoints are intervened upon.
    # Unit: fraction removed in [0, 1].
    lhd_default_int_reduction: float = 0.90

    # Duration of case-isolation contact reduction.
    # Unit: days.
    lhd_default_int_duration: int = 10

    # Mean used when generating person-level compliance values in GraphData.
    # Heterogeneous compliance is not currently applied by active LHD policy
    # interventions, so this should not be varied in the first experiment.
    # Unit: mean fraction in [0, 1].
    mean_compliance: float = 1.0

    # ------------------------------------------------------------------
    # Escalation trigger
    # ------------------------------------------------------------------

    # LHD-observable metric used by adaptive escalation:
    # new_reports, rolling_mean_reports, rolling_report_growth,
    # backlog_to_capacity_ratio, capacity_utilization, or maximum_queue_age.
    escalation_trigger_metric: str = "new_reports"

    # Trigger threshold. Units depend on escalation_trigger_metric:
    # - new_reports: cases reported per day
    # - rolling_mean_reports: mean cases reported per day
    # - rolling_report_growth: proportional change, e.g. 0.25 = 25% growth
    # - backlog_to_capacity_ratio: days of capacity, e.g. 2.0 = two days' work
    # - capacity_utilization: fraction, e.g. 0.90 = 90%
    # - maximum_queue_age: days
    escalation_trigger_threshold: float = 10.0

    # Window used for rolling report metrics.
    # Unit: days.
    escalation_window_days: int = 3

    # Number of consecutive evaluated days at or above the threshold required
    # before escalation is ordered.
    # Unit: days.
    escalation_consecutive_days: int = 1

    # Minimum current rolling report count required before proportional growth
    # can trigger escalation. Used only for rolling_report_growth.
    # Unit: mean reports per day over the current window.
    escalation_minimum_reports: int = 3

    # Delay from escalation order to intervention activation. Because orders
    # are made after daily transmission, activation cannot occur before the
    # following day even when configured as zero.
    # Unit: days.
    escalation_implementation_delay_days: int = 1

    # Duration of population-level emergency mitigation. Zero means the
    # intervention remains active through the rest of the simulation.
    # Unit: days; 0 = indefinite.
    escalation_duration_days: int = 0

    # Population intervention bundle activated at escalation.
    # See VALID_INTERVENTION_PROFILES.
    escalation_intervention_profile: str = "combined_emergency"

    # ------------------------------------------------------------------
    # Emergency contact reductions
    # ------------------------------------------------------------------

    # Fractions of each contact type removed by population-wide mitigation.
    #
    # Current implementation applies the remaining-contact factor to both
    # incoming and outgoing multipliers for all people. Thus effects compound
    # at the edge level: a 70% reduction gives factor 0.30 at each endpoint,
    # producing an effective edge factor of 0.09. These values require careful
    # calibration before substantive experiments.
    # Unit: fraction removed in [0, 1].
    emergency_school_reduction: float = 0.80
    emergency_workplace_reduction: float = 0.60
    emergency_casual_reduction: float = 0.70
    emergency_household_reduction: float = 0.0
    emergency_group_quarter_reduction: float = 0.0

    # ------------------------------------------------------------------
    # Phase-2 tracing behavior
    # ------------------------------------------------------------------

    # Behavior after escalation:
    # - stop: no new contact interviews
    # - targeted: continue household-only tracing
    # - continue: retain ordinary tracing
    phase2_trace_mode: str = "stop"

    # If true, pending contact interview, direct notification, and testing
    # referral tasks are cancelled when escalation becomes active.
    cancel_tracing_backlog_on_escalation: bool = True

    # ------------------------------------------------------------------
    # Automated notification
    # ------------------------------------------------------------------

    # Fraction of eligible school, workplace, and casual contact removed after
    # automated notification. Incoming and outgoing effects may compound.
    # Unit: fraction removed in [0, 1].
    automated_notification_reduction: float = 0.30

    # Duration of automated-notification contact reduction.
    # Unit: days.
    automated_notification_duration_days: int = 5

    # ------------------------------------------------------------------
    # Perfect-information benchmark
    # ------------------------------------------------------------------

    # Fraction of all contact types removed for every true exposed or
    # infectious person identified by the perfect-information benchmark.
    # The runtime uses a small positive multiplier floor, so 1.0 is near-total
    # rather than mathematically exact zero transmission.
    # Unit: fraction removed in [0, 1].
    perfect_information_reduction: float = 1.0

    # Duration of truth-based isolation in the perfect-information benchmark.
    # Unit: days.
    perfect_information_isolation_duration_days: int = 30


@dataclass(frozen=True)
class SimulationParams:
    """
    Simulation horizon, population selection, and reproducibility settings.
    """

    # Number of independent stochastic epidemic replicates run for each model
    # and policy. Increase to estimate outcome probabilities more precisely.
    # Unit: replicate count.
    n_replicates: int = 10

    # Human-readable output-directory label. Generated from local date and time
    # when not explicitly supplied.
    # Unit: string.
    run_name: str = field(
        default_factory=lambda: (
            "RUN_"
            + datetime.now().strftime(
                "%m-%d-%Y_%H-%M-%S"
            )
        )
    )

    # Maximum epidemic simulation horizon. Runs may end earlier when no exposed
    # or infectious people remain. Pending LHD work is not currently drained
    # after epidemiological extinction.
    # Unit: days.
    simulation_duration: int = 100

    # Number of imported infectious people sampled at replicate initialization.
    # Imports are generation 0 and excluded from local incidence and local
    # outbreak size.
    # Unit: people.
    I0: int = 2

    # Base seed used to derive graph, model, replicate, LHD, surveillance, and
    # transmission seeds. Keep fixed for exact reproducibility.
    # Unit: integer seed.
    seed: int = 2026

    # FRED synthetic-population location. Names must match available or
    # downloadable population files.
    county: str = "Keweenaw"
    state: str = "Michigan"

    # Number of casual-contact candidates available per eligible person in the
    # master graph. Must be at least population.cas_contacts.
    # Unit: candidate contacts per person.
    master_casual_candidates: int = 100

    # If true, rebuild the master candidate graph rather than loading a cached
    # MasterEdgelist.parquet when one is available.
    # Unit: boolean.
    overwrite_master: bool = True

# =====================================================================
# Combine all the above fields, and give them some methods to validate 
# =====================================================================

@dataclass(frozen=True)
class ModelConfig:
    """
    Top-level immutable model configuration.

    Use copy_with() to create modified configurations without mutating the
    original object.
    """

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
        """Convert the configuration to nested dictionaries."""
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
        """Write the complete configuration to JSON."""
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
        """Load a complete configuration from JSON."""
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
        """Build a configuration from nested section dictionaries."""
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
        """
        Return a configuration with section-level field overrides.

        Example:
            cfg.copy_with({
                "epi": {"base_transmission_prob": 0.10},
                "lhd": {"lhd_daily_capacity": 100.0},
                "sim": {"n_replicates": 50},
            })
        """
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
        """Validate ranges, required values, and named options."""
        self._validate_epi()
        self._validate_population()
        self._validate_lhd()
        self._validate_simulation()

    def _validate_epi(self) -> None:
        """Validate epidemic parameters."""
        probability_fields = {
            "epi.base_transmission_prob":
                self.epi.base_transmission_prob,
            "epi.vax_efficacy":
                self.epi.vax_efficacy,
            "epi.vax_uptake":
                self.epi.vax_uptake,
            "epi.symptomatic_fraction":
                self.epi.symptomatic_fraction,
        }

        for name, value in probability_fields.items():
            self._require_probability(
                name,
                value,
            )

        positive_fields = {
            "epi.latent_period":
                self.epi.latent_period,
            "epi.infectious_period":
                self.epi.infectious_period,
            "epi.latent_period_vax":
                self.epi.latent_period_vax,
            "epi.infectious_period_vax":
                self.epi.infectious_period_vax,
            "epi.gamma_alpha":
                self.epi.gamma_alpha,
            "epi.symptom_onset_shape":
                self.epi.symptom_onset_shape,
        }

        for name, value in positive_fields.items():
            self._require_positive(
                name,
                value,
            )

        nonnegative_fields = {
            "epi.symptom_onset_delay_mean":
                self.epi.symptom_onset_delay_mean,
            "epi.relative_infectiousness_vax":
                self.epi.relative_infectiousness_vax,
            "epi.susceptibility_multiplier_under_five":
                self.epi.susceptibility_multiplier_under_five,
            "epi.susceptibility_multiplier_elderly":
                self.epi.susceptibility_multiplier_elderly,
        }

        for name, value in nonnegative_fields.items():
            self._require_nonnegative(
                name,
                value,
            )

        # Individual isolation must always have a positive duration. Unlike
        # escalation_duration_days, zero does not mean indefinite here.
        self._require_positive(
            "lhd.lhd_default_int_duration",
            self.lhd.lhd_default_int_duration,
        )
        self._require_positive(
            "lhd.perfect_information_isolation_duration_days",
            self.lhd.perfect_information_isolation_duration_days,
        )

    def _validate_population(self) -> None:
        """Validate contact-network parameters."""
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
        """Validate LHD surveillance and response parameters."""
        self._require_nonnegative(
            "lhd.lhd_daily_capacity",
            self.lhd.lhd_daily_capacity,
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
            "lhd.lhd_default_int_reduction":
                self.lhd.lhd_default_int_reduction,
            "lhd.mean_compliance":
                self.lhd.mean_compliance,
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
        }

        for name, value in probability_fields.items():
            self._require_probability(
                name,
                value,
            )

        nonnegative_fields = {
            # Task costs
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

            # Task priorities
            "lhd.case_investigation_priority":
                self.lhd.case_investigation_priority,
            "lhd.isolation_support_priority":
                self.lhd.isolation_support_priority,
            "lhd.contact_interview_priority":
                self.lhd.contact_interview_priority,
            "lhd.contact_notification_priority":
                self.lhd.contact_notification_priority,
            "lhd.testing_referral_priority":
                self.lhd.testing_referral_priority,
            "lhd.automated_notification_priority":
                self.lhd.automated_notification_priority,
            "lhd.task_aging_priority_per_day":
                self.lhd.task_aging_priority_per_day,

            # Task deadlines
            "lhd.case_investigation_deadline_days":
                self.lhd.case_investigation_deadline_days,
            "lhd.isolation_support_deadline_days":
                self.lhd.isolation_support_deadline_days,
            "lhd.contact_interview_deadline_days":
                self.lhd.contact_interview_deadline_days,
            "lhd.contact_notification_deadline_days":
                self.lhd.contact_notification_deadline_days,
            "lhd.testing_referral_deadline_days":
                self.lhd.testing_referral_deadline_days,
            "lhd.automated_notification_deadline_days":
                self.lhd.automated_notification_deadline_days,

            # Surveillance delays and limits
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


            # Escalation values
            "lhd.escalation_trigger_threshold":
                self.lhd.escalation_trigger_threshold,
            "lhd.escalation_minimum_reports":
                self.lhd.escalation_minimum_reports,
            "lhd.escalation_implementation_delay_days":
                self.lhd.escalation_implementation_delay_days,
            "lhd.escalation_duration_days":
                self.lhd.escalation_duration_days,

            # Intervention durations
            "lhd.automated_notification_duration_days":
                self.lhd.automated_notification_duration_days,
        }

        for name, value in nonnegative_fields.items():
            self._require_nonnegative(
                name,
                value,
            )

        self._require_positive(
            "lhd.escalation_window_days",
            self.lhd.escalation_window_days,
        )
        self._require_positive(
            "lhd.escalation_consecutive_days",
            self.lhd.escalation_consecutive_days,
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
        """Validate simulation and reproducibility settings."""
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
        self._require_nonnegative(
            "sim.master_casual_candidates",
            self.sim.master_casual_candidates,
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
        """Require a value in the closed interval [0, 1]."""
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{name} must be in [0, 1]"
            )

    @staticmethod
    def _require_positive(
        name: str,
        value: float,
    ) -> None:
        """Require a value greater than zero."""
        if float(value) <= 0.0:
            raise ValueError(
                f"{name} must be > 0"
            )

    @staticmethod
    def _require_nonnegative(
        name: str,
        value: float,
    ) -> None:
        """Require a value greater than or equal to zero."""
        if float(value) < 0.0:
            raise ValueError(
                f"{name} must be >= 0"
            )