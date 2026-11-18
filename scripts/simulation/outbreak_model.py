# scripts/simulation/outbreak_model.py
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List as NumbaList

from scripts.config import ModelConfig
from scripts.graph.graph_utils import GraphData
from scripts.lhd.lhd import LocalHealthDepartment
from scripts.recorder.event_log import EventLedger
from scripts.utils.rng_utility import derive_seed_from_base


@njit
def _determine_transmissions_numba(
    state,
    infectious_indices,
    indptr_list,
    indices_list,
    weights_list,
    out_mat,
    in_mat,
    is_vax,
    ages,
    base_prob,
    vax_efficacy,
    susceptibility_under_five,
    susceptibility_elderly,
    relative_infectiousness_vax,
    infected_stamp,
    stamp_value,
    contact_type_ids,
    transmission_source_buffer,
    transmission_target_buffer,
    transmission_type_buffer,
    legacy_seed,
):
    """Evaluate daily transmission over contact-type-specific sparse graphs."""
    np.random.seed(int(legacy_seed))

    output_count = 0
    buffer_capacity = transmission_source_buffer.shape[0]

    for infectious_index in range(infectious_indices.shape[0]):
        source = infectious_indices[infectious_index]

        source_infectiousness = (
            relative_infectiousness_vax
            if is_vax[source]
            else 1.0
        )

        for type_index in range(len(indptr_list)):
            indptr = indptr_list[type_index]
            indices = indices_list[type_index]
            weights = weights_list[type_index]

            start = indptr[source]
            end = indptr[source + 1]

            if end <= start:
                continue

            source_multiplier = out_mat[type_index, source]
            if source_multiplier <= 0.0:
                continue

            contact_type_id = contact_type_ids[type_index]

            for edge_index in range(start, end):
                target = indices[edge_index]

                if state[target] != 0:
                    continue

                # A susceptible node may be infected only once per day.
                if infected_stamp[target] == stamp_value:
                    continue

                effective_weight = (
                    weights[edge_index]
                    * source_multiplier
                    * in_mat[type_index, target]
                )

                if effective_weight <= 0.0:
                    continue

                probability = (
                    base_prob
                    * effective_weight
                    * source_infectiousness
                )

                if is_vax[target]:
                    probability *= 1.0 - vax_efficacy

                age = ages[target]
                if age <= 5:
                    probability *= susceptibility_under_five
                elif age >= 65:
                    probability *= susceptibility_elderly

                if probability <= 0.0:
                    continue

                probability = min(probability, 1.0)

                if np.random.random() < probability:
                    infected_stamp[target] = stamp_value

                    transmission_source_buffer[
                        output_count
                    ] = source
                    transmission_target_buffer[
                        output_count
                    ] = target
                    transmission_type_buffer[
                        output_count
                    ] = contact_type_id

                    output_count += 1

                    if output_count >= buffer_capacity:
                        return output_count

    return output_count


class NetworkModel:
    """Network-based epidemic model coupled to an operational LHD response."""

    def __init__(
        self,
        config: ModelConfig,
        graphdata: GraphData,
        run_dir: str,
        *,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.config.validate()

        self._assign_graphdata_to_model(graphdata)

        self.Tmax = int(self.config.sim.simulation_duration)
        self.n_replicates = int(self.config.sim.n_replicates)
        self.county = self.config.sim.county
        self.results_folder = str(run_dir)

        self._base_seed = int(
            seed if seed is not None else self.config.sim.seed
        )

        self.rng = None
        self.replicate_ind = None
        self.replicate_seed = None
        self._numba_legacy_seed_base = None
        self._numba_ready = False
        self._seed_metadata: Dict[str, int] = {}

        self.event_ledger = EventLedger()

        # Per-replicate epidemic trajectories retained for audits and optional
        # visualizations. Compact analysis outputs are generated separately.
        self.all_states_over_time = [None] * self.n_replicates
        self.all_new_exposures = [None] * self.n_replicates
        self.all_surveillance_batches = [None] * self.n_replicates
        self.all_daily_epi = [None] * self.n_replicates
        self.all_lhd_daily_logs = [None] * self.n_replicates
        self.all_lhd_tasks = [None] * self.n_replicates

        self.all_stochastic_dieout = np.zeros(
            self.n_replicates,
            dtype=bool,
        )
        self.all_end_days = np.full(
            self.n_replicates,
            self.Tmax,
            dtype=np.int32,
        )

        # Replicate-level natural-history and reproducibility metadata.
        self.all_seed_metadata = [None] * self.n_replicates
        self.all_imported_ids = [None] * self.n_replicates
        self.all_vax_status = [None] * self.n_replicates
        self.all_infection_generation = [None] * self.n_replicates
        self.all_infector = [None] * self.n_replicates
        self.all_infection_day = [None] * self.n_replicates
        self.all_infectious_onset_day = [None] * self.n_replicates
        self.all_symptom_onset_day = [None] * self.n_replicates
        self.all_recovery_day = [None] * self.n_replicates
        self.all_will_be_symptomatic = [None] * self.n_replicates

    def _assign_graphdata_to_model(
        self,
        graphdata: GraphData,
    ) -> None:
        """Attach the immutable run network and demographic arrays."""
        self.N = int(graphdata.N)
        self.edge_list = graphdata.edge_list
        self.adj_matrix = graphdata.adj_matrix
        self.individual_lookup = graphdata.individual_lookup
        self.ages = graphdata.ages.astype(np.int32)
        self.sexes = graphdata.sexes
        self.compliances = graphdata.compliances
        self.neighbor_map = graphdata.neighbor_map
        self.fast_neighbor_map = graphdata.fast_neighbor_map
        self.csr_by_type = graphdata.csr_by_type
        self.contact_types = list(graphdata.contact_types)
        self.ct_to_id = dict(graphdata.ct_to_id)
        self.id_to_ct = dict(graphdata.id_to_ct)
        self.full_node_list = graphdata.full_node_list
        self.degrees_arr = graphdata.degrees_arr

        self.in_multiplier = {
            contact_type: np.ones(
                self.N,
                dtype=np.float32,
            )
            for contact_type in self.contact_types
        }
        self.out_multiplier = {
            contact_type: np.ones(
                self.N,
                dtype=np.float32,
            )
            for contact_type in self.contact_types
        }

    # ------------------------------------------------------------------
    # Event and output interfaces
    # ------------------------------------------------------------------

    def record_event(
        self,
        *,
        day: int,
        event_type: str,
        node_id: Optional[int] = None,
        source_node_id: Optional[int] = None,
        infection_origin: Optional[str] = None,
        infection_generation: Optional[int] = None,
        disease_stage: Optional[str] = None,
        symptomatic: Optional[bool] = None,
        detection_source: Optional[str] = None,
        report_source: Optional[str] = None,
        contact_type: Optional[str] = None,
        result: Optional[str] = None,
        value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        action_id: Optional[str] = None,
        task_id: Optional[str] = None,
        policy_phase: Optional[int] = None,
    ) -> int:
        """Append an event, filling node and policy metadata when available."""
        node = None if node_id is None else int(node_id)

        if (
            node is not None
            and hasattr(self, "infection_generation")
            and 0 <= node < self.N
        ):
            node_generation = int(
                self.infection_generation[node]
            )

            if (
                infection_generation is None
                and node_generation >= 0
            ):
                infection_generation = node_generation

            if infection_origin is None:
                if node_generation == 0:
                    infection_origin = "imported"
                elif node_generation > 0:
                    infection_origin = "local"

            if (
                symptomatic is None
                and hasattr(self, "is_symptomatic")
            ):
                symptomatic = bool(
                    self.is_symptomatic[node]
                )

        policy_name = self.config.lhd.policy_name

        if hasattr(self, "lhd") and self.lhd is not None:
            policy_name = self.lhd.policy_name

            if policy_phase is None:
                policy_phase = getattr(
                    self.lhd,
                    "policy_phase",
                    None,
                )

        return self.event_ledger.append(
            run_number=int(self.replicate_ind),
            day=int(day),
            event_type=event_type,
            node_id=node,
            source_node_id=source_node_id,
            action_id=action_id,
            task_id=task_id,
            policy_name=policy_name,
            policy_phase=policy_phase,
            infection_origin=infection_origin,
            infection_generation=infection_generation,
            disease_stage=disease_stage,
            symptomatic=symptomatic,
            detection_source=detection_source,
            report_source=report_source,
            contact_type=contact_type,
            result=result,
            value=value,
            metadata=metadata,
        )

    def event_log_to_df(self) -> pd.DataFrame:
        return self.event_ledger.to_dataframe()

    def daily_epi_to_df(self) -> pd.DataFrame:
        return self._concat_frames(self.all_daily_epi)

    def lhd_daily_to_df(self) -> pd.DataFrame:
        return self._concat_frames(self.all_lhd_daily_logs)

    def lhd_tasks_to_df(self) -> pd.DataFrame:
        frames = [
            frame
            for frame in self.all_lhd_tasks
            if isinstance(frame, pd.DataFrame)
            and not frame.empty
        ]

        if frames:
            return pd.concat(
                frames,
                ignore_index=True,
                sort=False,
            )

        # A stable empty schema prevents failures for policies with no tasks.
        return pd.DataFrame(
            columns=[
                "run_number",
                "task_id",
                "task_type",
                "target_kind",
                "target",
                "source_case",
                "created_day",
                "earliest_start_day",
                "deadline_day",
                "base_priority",
                "work_cost",
                "status",
                "selected_day",
                "started_day",
                "completed_day",
                "expired_day",
                "cancelled_day",
                "expiration_reason",
                "cancellation_reason",
                "source_algorithm",
                "reason",
                "params",
            ]
        )

    @staticmethod
    def _concat_frames(
        frames: Sequence[Optional[pd.DataFrame]],
    ) -> pd.DataFrame:
        usable = [
            frame
            for frame in frames
            if isinstance(frame, pd.DataFrame)
            and not frame.empty
        ]

        if not usable:
            return pd.DataFrame()

        return pd.concat(
            usable,
            ignore_index=True,
            sort=False,
        )

    # ------------------------------------------------------------------
    # Replicate initialization and natural history
    # ------------------------------------------------------------------

    def _initialize_replicate(
        self,
        run: int,
    ) -> None:
        """Initialize one epidemic replicate and its operational LHD."""
        self.current_time = 0
        self.replicate_ind = int(run)
        # clear LHD from last run 
        self.lhd = None


        self.replicate_seed = int(
            derive_seed_from_base(
                self._base_seed,
                self.replicate_ind,
            )
        )

        self.rng = np.random.default_rng(
            self.replicate_seed
        )

        self.lhd_seed = int(
            derive_seed_from_base(
                self.replicate_seed,
                "lhd",
            )
        )
        self.surv_seed = int(
            derive_seed_from_base(
                self.replicate_seed,
                "surveillance",
            )
        )
        self._numba_legacy_seed_base = int(
            derive_seed_from_base(
                self.replicate_seed,
                "numba",
            )
        )

        self._seed_metadata = {
            "base_seed": int(self._base_seed),
            "replicate_ind": int(self.replicate_ind),
            "replicate_seed": int(self.replicate_seed),
            "lhd_seed": int(self.lhd_seed),
            "surveillance_seed": int(self.surv_seed),
        }
        self.all_seed_metadata[run] = dict(
            self._seed_metadata
        )

        self.is_vaccinated = (
            self.rng.random(self.N)
            < float(self.config.epi.vax_uptake)
        )

        # SEIR states: S=0, E=1, I=2, R=3.
        self.state = np.zeros(
            self.N,
            dtype=np.int8,
        )
        self.time_in_state = np.zeros(
            self.N,
            dtype=np.float32,
        )
        self.latent_periods = np.full(
            self.N,
            np.nan,
            dtype=np.float32,
        )
        self.infectious_periods = np.full(
            self.N,
            np.nan,
            dtype=np.float32,
        )

        self.infection_day = np.full(
            self.N,
            -1,
            dtype=np.int32,
        )
        self.infectious_onset_day = np.full(
            self.N,
            -1,
            dtype=np.int32,
        )
        self.recovery_day = np.full(
            self.N,
            -1,
            dtype=np.int32,
        )
        self.infection_generation = np.full(
            self.N,
            -1,
            dtype=np.int32,
        )
        self.infector = np.full(
            self.N,
            -1,
            dtype=np.int32,
        )
        self.infection_contact_type = np.full(
            self.N,
            -1,
            dtype=np.int16,
        )

        self._initialize_symptom_disposition()

        self.states_over_time: List[List[List[int]]] = []
        self.new_exposures: List[np.ndarray] = []
        self.new_infections: List[np.ndarray] = []
        self.daily_epi_rows: List[Dict[str, Any]] = []

        self._initialize_imported_cases()

        self.simulation_end_day = self.Tmax
        self.stochastic_dieout = False

        # Intervention multipliers must be reset between replicates.
        for contact_type in self.contact_types:
            self.in_multiplier[contact_type].fill(1.0)
            self.out_multiplier[contact_type].fill(1.0)

        self._prepare_transmission_buffers()

        self.lhd = LocalHealthDepartment(
            model=self,
            seed=self.lhd_seed,
            surv_seed=self.surv_seed,
            capacity=self.config.lhd.lhd_daily_capacity,
            policy_name=self.config.lhd.policy_name,
        )
        self.lhd.reset_for_run()

        self._append_daily_epi(
            day=0,
            batch=None,
            new_local_exposures=0,
            new_infectious_onsets=0,
            new_symptom_onsets=int(
                np.count_nonzero(
                    self.is_symptomatic
                )
            ),
        )

    def _initialize_symptom_disposition(self) -> None:
        """Assign whether and when each person would develop symptoms."""
        self.will_be_symptomatic = (
            self.rng.random(self.N)
            < float(
                self.config.epi.symptomatic_fraction
            )
        )

        symptom_mean = float(
            self.config.epi.symptom_onset_delay_mean
        )
        symptom_shape = float(
            self.config.epi.symptom_onset_shape
        )

        if symptom_mean <= 0.0:
            self.symptom_onset_delays = np.zeros(
                self.N,
                dtype=np.int32,
            )
        else:
            symptom_draws = self.rng.gamma(
                shape=symptom_shape,
                scale=symptom_mean / symptom_shape,
                size=self.N,
            )
            self.symptom_onset_delays = np.ceil(
                symptom_draws
            ).astype(np.int32)

        self.symptom_onset_day = np.full(
            self.N,
            -1,
            dtype=np.int32,
        )
        self.is_symptomatic = np.zeros(
            self.N,
            dtype=np.bool_,
        )

    def _initialize_imported_cases(self) -> None:
        """Sample imported infectious cases and record day-zero state."""
        imported_count = int(self.config.sim.I0)

        if imported_count > self.N:
            raise ValueError(
                f"sim.I0 ({imported_count}) cannot exceed "
                f"population size ({self.N})"
            )

        if imported_count:
            imported_ids = np.sort(
                self.rng.choice(
                    self.N,
                    size=imported_count,
                    replace=False,
                ).astype(np.int32)
            )
        else:
            imported_ids = np.empty(
                0,
                dtype=np.int32,
            )

        self.imported_ids = imported_ids
        self.I0 = imported_ids.tolist()
        self.all_imported_ids[
            self.replicate_ind
        ] = imported_ids.copy()

        if imported_ids.size:
            self.state[imported_ids] = 2
            self.infection_day[imported_ids] = 0
            self.infectious_onset_day[
                imported_ids
            ] = 0
            self.infection_generation[
                imported_ids
            ] = 0
            self.infectious_periods[
                imported_ids
            ] = self.assign_infectious_period(
                imported_ids
            )

            symptomatic_imports = imported_ids[
                self.will_be_symptomatic[
                    imported_ids
                ]
            ]
            self.symptom_onset_day[
                symptomatic_imports
            ] = self.symptom_onset_delays[
                symptomatic_imports
            ]

            day_zero_symptoms = symptomatic_imports[
                self.symptom_onset_day[
                    symptomatic_imports
                ] <= 0
            ]
            self.is_symptomatic[
                day_zero_symptoms
            ] = True

            for node in imported_ids:
                node = int(node)

                self.record_event(
                    day=0,
                    event_type="importation",
                    node_id=node,
                    infection_origin="imported",
                    infection_generation=0,
                    disease_stage="infectious",
                )
                self.record_event(
                    day=0,
                    event_type="infectious_onset",
                    node_id=node,
                    infection_origin="imported",
                    infection_generation=0,
                    disease_stage="infectious",
                )

            for node in day_zero_symptoms:
                self.record_event(
                    day=0,
                    event_type="symptom_onset",
                    node_id=int(node),
                    disease_stage="infectious",
                    symptomatic=True,
                )

        susceptible = np.where(
            self.state == 0
        )[0].tolist()

        self.states_over_time.append(
            [
                susceptible,
                [],
                imported_ids.tolist(),
                [],
            ]
        )
        self.new_exposures.append(
            np.empty(0, dtype=np.int32)
        )
        self.new_infections.append(
            np.empty(0, dtype=np.int32)
        )

    def assign_latent_period(
        self,
        indices,
    ) -> np.ndarray:
        """Draw E-to-I durations for newly exposed people."""
        indices = np.atleast_1d(indices).astype(
            np.int32,
            copy=False,
        )

        means = np.where(
            self.is_vaccinated[indices],
            float(
                self.config.epi.latent_period_vax
            ),
            float(self.config.epi.latent_period),
        )

        shape = float(self.config.epi.gamma_alpha)

        return self.rng.gamma(
            shape=shape,
            scale=means / shape,
        )

    def assign_infectious_period(
        self,
        indices,
    ) -> np.ndarray:
        """Draw infectious durations for newly infectious people."""
        indices = np.atleast_1d(indices).astype(
            np.int32,
            copy=False,
        )

        means = np.where(
            self.is_vaccinated[indices],
            float(
                self.config.epi.infectious_period_vax
            ),
            float(
                self.config.epi.infectious_period
            ),
        )

        shape = float(self.config.epi.gamma_alpha)

        return self.rng.gamma(
            shape=shape,
            scale=means / shape,
        )

    # ------------------------------------------------------------------
    # Transmission
    # ------------------------------------------------------------------

    def _prepare_transmission_buffers(self) -> None:
        if (
            not hasattr(self, "_trans_src_buf")
            or self._trans_src_buf.shape[0]
            != self.N
        ):
            self._trans_src_buf = np.empty(
                self.N,
                dtype=np.int32,
            )
            self._trans_tgt_buf = np.empty(
                self.N,
                dtype=np.int32,
            )
            self._trans_ct_buf = np.empty(
                self.N,
                dtype=np.int16,
            )

        if (
            not hasattr(self, "_infected_stamp")
            or self._infected_stamp.shape[0]
            != self.N
        ):
            self._infected_stamp = np.full(
                self.N,
                -1,
                dtype=np.int32,
            )

        self._infected_stamp.fill(-1)

    def _prepare_numba_structures(self) -> None:
        """Build Numba-compatible contact arrays in stable contact-type order."""
        contact_types = sorted(self.contact_types)
        self._numba_contact_types = contact_types
        self._numba_ct_ids = np.asarray(
            [
                self.ct_to_id[contact_type]
                for contact_type in contact_types
            ],
            dtype=np.int16,
        )

        indptr_list = NumbaList()
        indices_list = NumbaList()
        weights_list = NumbaList()

        for contact_type in contact_types:
            indptr, indices, weights = (
                self.csr_by_type[contact_type]
            )

            indptr_list.append(
                np.ascontiguousarray(
                    np.asarray(
                        indptr,
                        dtype=np.int64,
                    )
                )
            )
            indices_list.append(
                np.ascontiguousarray(
                    np.asarray(
                        indices,
                        dtype=np.int64,
                    )
                )
            )
            weights_list.append(
                np.ascontiguousarray(
                    np.asarray(
                        weights,
                        dtype=np.float32,
                    )
                )
            )

        shape = (
            len(contact_types),
            self.N,
        )
        self._numba_out_mat = np.ones(
            shape,
            dtype=np.float32,
        )
        self._numba_in_mat = np.ones(
            shape,
            dtype=np.float32,
        )

        self._numba_indptr_list = indptr_list
        self._numba_indices_list = indices_list
        self._numba_weights_list = weights_list
        self._numba_ready = True

    def _update_multiplier_matrices(self) -> None:
        for index, contact_type in enumerate(
            self._numba_contact_types
        ):
            self._numba_out_mat[index] = (
                self.out_multiplier[
                    contact_type
                ]
            )
            self._numba_in_mat[index] = (
                self.in_multiplier[
                    contact_type
                ]
            )

    def determine_transmissions(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return source, target, and contact-type arrays for today's infections."""
        infectious_indices = np.where(
            self.state == 2
        )[0]

        if infectious_indices.size == 0:
            return self._empty_transmissions()

        if not self._numba_ready:
            self._prepare_numba_structures()

        if not self._numba_contact_types:
            return self._empty_transmissions()

        self._update_multiplier_matrices()

        legacy_seed = int(
            derive_seed_from_base(
                self._numba_legacy_seed_base,
                int(self.current_time),
            )
        )

        total = _determine_transmissions_numba(
            self.state,
            infectious_indices,
            self._numba_indptr_list,
            self._numba_indices_list,
            self._numba_weights_list,
            self._numba_out_mat,
            self._numba_in_mat,
            self.is_vaccinated,
            self.ages,
            float(
                self.config.epi.base_transmission_prob
            ),
            float(self.config.epi.vax_efficacy),
            float(
                self.config.epi
                .susceptibility_multiplier_under_five
            ),
            float(
                self.config.epi
                .susceptibility_multiplier_elderly
            ),
            float(
                self.config.epi
                .relative_infectiousness_vax
            ),
            self._infected_stamp,
            np.int32(self.current_time),
            self._numba_ct_ids,
            self._trans_src_buf,
            self._trans_tgt_buf,
            self._trans_ct_buf,
            legacy_seed,
        )

        return (
            self._trans_src_buf[:total].copy(),
            self._trans_tgt_buf[:total].copy(),
            self._trans_ct_buf[:total].copy(),
        )

    @staticmethod
    def _empty_transmissions(
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int16),
        )

    # ------------------------------------------------------------------
    # Daily epidemic update
    # ------------------------------------------------------------------

    def step(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Dict[str, np.ndarray],
        int,
    ]:
        """Advance transmission, disease progression, and symptoms by one day."""
        day = int(self.current_time)

        transmission_sources, newly_exposed, transmission_types = (
            self.determine_transmissions()
        )

        self._apply_new_exposures(
            day=day,
            sources=transmission_sources,
            targets=newly_exposed,
            contact_types=transmission_types,
        )

        newly_infectious = self._advance_exposed(
            day
        )
        newly_recovered = self._advance_infectious(
            day
        )
        new_symptom_onsets = (
            self._advance_symptoms(day)
        )

        susceptible = np.where(
            self.state == 0
        )[0]
        exposed = np.where(self.state == 1)[0]
        infectious = np.where(
            self.state == 2
        )[0]
        recovered = np.where(
            self.state == 3
        )[0]

        self.states_over_time.append(
            [
                susceptible.tolist(),
                exposed.tolist(),
                infectious.tolist(),
                recovered.tolist(),
            ]
        )
        self.new_exposures.append(
            newly_exposed.copy()
        )
        self.new_infections.append(
            newly_infectious.copy()
        )

        epi_state = {
            "new_pre_ids": newly_exposed.copy(),
            "new_inf_ids": newly_infectious.copy(),
            "pre_ids": exposed.astype(
                np.int32,
                copy=False,
            ),
            "inf_ids": infectious.astype(
                np.int32,
                copy=False,
            ),
            "symptomatic_ids": np.where(
                self.is_symptomatic
                & (self.state == 2)
            )[0].astype(
                np.int32,
                copy=False,
            ),
            "imported_ids": self.imported_ids.copy(),
            "infection_generation":
                self.infection_generation.copy(),
        }

        return (
            transmission_sources,
            newly_exposed,
            transmission_types,
            epi_state,
            int(new_symptom_onsets.size),
        )

    def _apply_new_exposures(
        self,
        *,
        day: int,
        sources: np.ndarray,
        targets: np.ndarray,
        contact_types: np.ndarray,
    ) -> None:
        if targets.size == 0:
            return

        self.state[targets] = 1
        self.time_in_state[targets] = -1
        self.latent_periods[
            targets
        ] = self.assign_latent_period(targets)
        self.infection_day[targets] = day
        self.infection_generation[
            targets
        ] = (
            self.infection_generation[sources] + 1
        )
        self.infector[targets] = sources
        self.infection_contact_type[
            targets
        ] = contact_types

        for source, target, type_id in zip(
            sources,
            targets,
            contact_types,
        ):
            type_id = int(type_id)

            self.record_event(
                day=day,
                event_type="exposure",
                node_id=int(target),
                source_node_id=int(source),
                infection_origin="local",
                infection_generation=int(
                    self.infection_generation[
                        int(target)
                    ]
                ),
                disease_stage="exposed",
                contact_type=self.id_to_ct.get(
                    type_id,
                    str(type_id),
                ),
            )

    def _advance_exposed(
        self,
        day: int,
    ) -> np.ndarray:
        exposed = np.where(self.state == 1)[0]
        self.time_in_state[exposed] += 1

        newly_infectious = exposed[
            self.time_in_state[exposed]
            >= self.latent_periods[exposed]
        ].astype(
            np.int32,
            copy=False,
        )

        if newly_infectious.size == 0:
            return newly_infectious

        self.state[newly_infectious] = 2
        self.time_in_state[newly_infectious] = -1
        self.infectious_onset_day[
            newly_infectious
        ] = day
        self.infectious_periods[
            newly_infectious
        ] = self.assign_infectious_period(
            newly_infectious
        )

        symptomatic = newly_infectious[
            self.will_be_symptomatic[
                newly_infectious
            ]
        ]
        self.symptom_onset_day[symptomatic] = (
            day
            + self.symptom_onset_delays[
                symptomatic
            ]
        )

        for node in newly_infectious:
            self.record_event(
                day=day,
                event_type="infectious_onset",
                node_id=int(node),
                disease_stage="infectious",
            )

        return newly_infectious

    def _advance_infectious(
        self,
        day: int,
    ) -> np.ndarray:
        infectious = np.where(
            self.state == 2
        )[0]
        self.time_in_state[infectious] += 1

        newly_recovered = infectious[
            self.time_in_state[infectious]
            >= self.infectious_periods[infectious]
        ].astype(
            np.int32,
            copy=False,
        )

        if newly_recovered.size:
            self.state[newly_recovered] = 3
            self.time_in_state[
                newly_recovered
            ] = 0
            self.recovery_day[
                newly_recovered
            ] = day

            for node in newly_recovered:
                self.record_event(
                    day=day,
                    event_type="recovery",
                    node_id=int(node),
                    disease_stage="recovered",
                )

        recovered = np.where(
            self.state == 3
        )[0]
        self.time_in_state[recovered] += 1

        return newly_recovered

    def _advance_symptoms(
        self,
        day: int,
    ) -> np.ndarray:
        symptom_onsets = np.where(
            (self.state == 2)
            & self.will_be_symptomatic
            & (~self.is_symptomatic)
            & (self.symptom_onset_day >= 0)
            & (self.symptom_onset_day <= day)
        )[0].astype(
            np.int32,
            copy=False,
        )

        if symptom_onsets.size:
            self.is_symptomatic[
                symptom_onsets
            ] = True

            for node in symptom_onsets:
                self.record_event(
                    day=day,
                    event_type="symptom_onset",
                    node_id=int(node),
                    disease_stage="infectious",
                    symptomatic=True,
                )

        return symptom_onsets

    # ------------------------------------------------------------------
    # Simulation orchestration
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        for run in range(self.n_replicates):
            surveillance_history = []

            self._initialize_replicate(run)
            self.all_vax_status[
                run
            ] = self.is_vaccinated.copy()

            day = 0

            while day < self.Tmax:
                day += 1
                self.current_time = day

                # Scheduled intervention changes occur before transmission.
                self.lhd.pre_epi_step(day)

                (
                    _transmission_sources,
                    transmission_targets,
                    _transmission_types,
                    epi_state,
                    new_symptom_onsets,
                ) = self.step()

                batch = self.lhd.step(
                    t=day,
                    epi_state=epi_state,
                )
                surveillance_history.append(batch)

                self._append_daily_epi(
                    day=day,
                    batch=batch,
                    new_local_exposures=int(
                        transmission_targets.size
                    ),
                    new_infectious_onsets=int(
                        epi_state[
                            "new_inf_ids"
                        ].size
                    ),
                    new_symptom_onsets=(
                        new_symptom_onsets
                    ),
                )

                if not np.any(
                    (self.state == 1)
                    | (self.state == 2)
                ):
                    self.simulation_end_day = day
                    self.stochastic_dieout = True
                    break

            self._store_replicate_outputs(
                run=run,
                surveillance_history=(
                    surveillance_history
                ),
            )

    def _store_replicate_outputs(
        self,
        *,
        run: int,
        surveillance_history: List[
            Dict[str, np.ndarray]
        ],
    ) -> None:
        self.all_states_over_time[run] = [
            [
                list(state_group)
                for state_group in day_state
            ]
            for day_state in self.states_over_time
        ]
        self.all_new_exposures[run] = [
            exposures.copy()
            for exposures in self.new_exposures
        ]
        self.all_surveillance_batches[
            run
        ] = surveillance_history
        self.all_stochastic_dieout[
            run
        ] = self.stochastic_dieout
        self.all_end_days[
            run
        ] = self.simulation_end_day

        self.all_daily_epi[run] = pd.DataFrame(
            self.daily_epi_rows
        )
        self.all_lhd_daily_logs[
            run
        ] = self.lhd.lhd_daily_log_to_df()
        self.all_lhd_tasks[
            run
        ] = self.lhd.tasks_to_df()

        self.all_infection_generation[
            run
        ] = self.infection_generation.copy()
        self.all_infector[
            run
        ] = self.infector.copy()
        self.all_infection_day[
            run
        ] = self.infection_day.copy()
        self.all_infectious_onset_day[
            run
        ] = self.infectious_onset_day.copy()
        self.all_symptom_onset_day[
            run
        ] = self.symptom_onset_day.copy()
        self.all_recovery_day[
            run
        ] = self.recovery_day.copy()
        self.all_will_be_symptomatic[
            run
        ] = self.will_be_symptomatic.copy()

    def _append_daily_epi(
        self,
        *,
        day: int,
        batch: Optional[Dict[str, np.ndarray]],
        new_local_exposures: int,
        new_infectious_onsets: int,
        new_symptom_onsets: int,
    ) -> None:
        """Append one compact daily epidemiological summary."""
        local_mask = self.infection_generation > 0
        imported_mask = (
            self.infection_generation == 0
        )

        if batch is None:
            detections = 0
            reports = 0
            tests_completed = 0
            positive_tests = 0
        else:
            detections = self._array_size(
                batch,
                "detected_cases",
            )
            reports = self._array_size(
                batch,
                "reported_cases",
            )
            tests_completed = self._array_size(
                batch,
                "tests_completed",
            )
            positive_tests = int(
                np.count_nonzero(
                    np.asarray(
                        batch.get(
                            "test_results_positive",
                            np.empty(
                                0,
                                dtype=bool,
                            ),
                        ),
                        dtype=bool,
                    )
                )
            )

        self.daily_epi_rows.append(
            {
                "run_number": int(
                    self.replicate_ind
                ),
                "day": int(day),
                "susceptible": int(
                    np.count_nonzero(
                        self.state == 0
                    )
                ),
                "local_exposed": int(
                    np.count_nonzero(
                        (self.state == 1)
                        & local_mask
                    )
                ),
                "local_infectious": int(
                    np.count_nonzero(
                        (self.state == 2)
                        & local_mask
                    )
                ),
                "local_recovered": int(
                    np.count_nonzero(
                        (self.state == 3)
                        & local_mask
                    )
                ),
                "imported_infectious": int(
                    np.count_nonzero(
                        (self.state == 2)
                        & imported_mask
                    )
                ),
                "new_local_exposures": int(
                    new_local_exposures
                ),
                "new_local_infectious_onsets": int(
                    new_infectious_onsets
                ),
                "cumulative_local_infections": int(
                    np.count_nonzero(local_mask)
                ),
                "new_symptom_onsets": int(
                    new_symptom_onsets
                ),
                "currently_symptomatic": int(
                    np.count_nonzero(
                        self.is_symptomatic
                        & (self.state == 2)
                    )
                ),
                "new_detections": detections,
                "new_reports_received": reports,
                "new_tests_completed":
                    tests_completed,
                "new_positive_tests":
                    positive_tests,
            }
        )

    @staticmethod
    def _array_size(
        mapping: Dict[str, Any],
        key: str,
    ) -> int:
        return int(
            np.asarray(
                mapping.get(
                    key,
                    np.empty(0),
                )
            ).size
        )

    # ------------------------------------------------------------------
    # Compact summaries
    # ------------------------------------------------------------------

    def results_to_df(
        self,
        metrics: Optional[
            Sequence[str]
        ] = None,
    ) -> pd.DataFrame:
        """Return legacy-compatible peak and outbreak-size metrics."""
        if metrics is None:
            metrics = (
                "peakPrev",
                "peakTime",
                "outbreakSize",
            )

        metrics = list(metrics)
        supported = {
            "peakPrev",
            "peakTime",
            "outbreakSize",
        }
        unknown = [
            metric
            for metric in metrics
            if metric not in supported
        ]

        if unknown:
            warnings.warn(
                f"Unsupported metrics requested: {unknown}. "
                f"Supported metrics are {sorted(supported)}."
            )

        rows = []

        for run in range(self.n_replicates):
            daily = self.all_daily_epi[run]

            if (
                isinstance(daily, pd.DataFrame)
                and not daily.empty
            ):
                local_infectious = daily[
                    "local_infectious"
                ].to_numpy(dtype=float)

                peak_index = int(
                    np.argmax(local_infectious)
                )
                peak_prevalence = float(
                    local_infectious[peak_index]
                    / self.N
                )
                peak_time = int(
                    daily.iloc[peak_index]["day"]
                )
                outbreak_size = int(
                    daily[
                        "cumulative_local_infections"
                    ].iloc[-1]
                )
            else:
                peak_prevalence = 0.0
                peak_time = pd.NA
                outbreak_size = 0

            values = {
                "run_number": int(run),
                "peakPrev": peak_prevalence,
                "peakTime": peak_time,
                "outbreakSize": outbreak_size,
            }
            rows.append(
                {
                    "run_number": values[
                        "run_number"
                    ],
                    **{
                        metric: values.get(
                            metric,
                            pd.NA,
                        )
                        for metric in metrics
                    },
                }
            )

        frame = pd.DataFrame(rows)

        if "peakTime" in frame.columns:
            frame["peakTime"] = frame[
                "peakTime"
            ].astype("Int64")

        if "outbreakSize" in frame.columns:
            frame["outbreakSize"] = frame[
                "outbreakSize"
            ].astype("Int64")

        return frame[
            ["run_number", *metrics]
        ]

    def detection_summary_to_df(
        self,
    ) -> pd.DataFrame:
        """Summarize detection by origin and symptom disposition."""
        events = self.event_log_to_df()
        detections = events[
            events["event_type"] == "detection"
        ]

        rows = []

        for run in range(self.n_replicates):
            generations = (
                self.all_infection_generation[run]
            )
            symptom_disposition = (
                self.all_will_be_symptomatic[
                    run
                ]
            )

            if generations is None:
                continue

            infected = np.where(
                generations >= 0
            )[0]
            detected_nodes = set(
                detections.loc[
                    detections["run_number"]
                    == run,
                    "node_id",
                ]
                .dropna()
                .astype(int)
                .tolist()
            )

            groups = {
                "all_infected": infected,
                "imported": np.where(
                    generations == 0
                )[0],
                "local": np.where(
                    generations > 0
                )[0],
                "symptomatic_disposition":
                    infected[
                        symptom_disposition[
                            infected
                        ]
                    ],
                "asymptomatic_disposition":
                    infected[
                        ~symptom_disposition[
                            infected
                        ]
                    ],
            }

            for group_name, nodes in groups.items():
                node_set = set(
                    np.asarray(
                        nodes,
                        dtype=np.int32,
                    ).tolist()
                )
                detected_count = len(
                    node_set & detected_nodes
                )
                total = len(node_set)

                rows.append(
                    {
                        "run_number": int(run),
                        "group": group_name,
                        "n_infected": int(total),
                        "n_detected": int(
                            detected_count
                        ),
                        "detection_fraction": (
                            detected_count / total
                            if total
                            else np.nan
                        ),
                    }
                )

        return pd.DataFrame(rows)

    def factual_run_summary_to_df(
        self,
    ) -> pd.DataFrame:
        """Return factual epidemic and operational summaries by replicate."""
        events = self.event_log_to_df()
        daily_epi = self.daily_epi_to_df()
        daily_lhd = self.lhd_daily_to_df()
        tasks = self.lhd_tasks_to_df()

        rows = []

        for run in range(self.n_replicates):
            run_events = events[
                events["run_number"] == run
            ].copy()
            run_epi = daily_epi[
                daily_epi["run_number"] == run
            ].sort_values("day")
            run_lhd = daily_lhd[
                daily_lhd["run_number"] == run
            ].sort_values("t")
            run_tasks = tasks[
                tasks["run_number"] == run
            ].copy()

            first_detection = self._first_event_day(
                run_events,
                "detection",
            )
            first_report = self._first_event_day(
                run_events,
                "report_received",
            )
            first_imported_report = (
                self._first_event_day(
                    run_events,
                    "report_received",
                    origin="imported",
                )
            )
            first_local_report = (
                self._first_event_day(
                    run_events,
                    "report_received",
                    origin="local",
                )
            )
            first_investigation = (
                self._first_event_day(
                    run_events,
                    "case_investigation_completed",
                )
            )
            first_isolation = (
                self._first_event_day(
                    run_events,
                    "isolation_started",
                )
            )
            first_notification = min(
                [
                    day
                    for day in (
                        self._first_event_day(
                            run_events,
                            "contact_notified",
                        ),
                        self._first_event_day(
                            run_events,
                            "automated_contact_notified",
                        ),
                    )
                    if not pd.isna(day)
                ],
                default=pd.NA,
            )

            exposures = run_events[
                run_events["event_type"]
                == "exposure"
            ]
            escalation_day = (
                self._first_event_day(
                    run_events,
                    "escalation_started",
                )
            )

            report_delays = self._task_delay_series(
                run_tasks=run_tasks,
                run_events=run_events,
                task_type="case_investigation",
            )
            isolation_delays = self._task_delay_series(
                run_tasks=run_tasks,
                run_events=run_events,
                task_type="isolation_support",
            )
            notification_delays = (
                self._notification_delay_series(
                    run_tasks
                )
            )

            generations = np.asarray(
                self.all_infection_generation[run]
            )

            rows.append(
                {
                    "run_number": int(run),
                    "policy_name":
                        self.config.lhd.policy_name,
                    "total_local_infections": int(
                        np.count_nonzero(
                            generations > 0
                        )
                    ),
                    "peak_local_prevalence_count":
                        self._frame_max(
                            run_epi,
                            "local_infectious",
                        ),
                    "peak_local_incidence_count":
                        self._frame_max(
                            run_epi,
                            "new_local_exposures",
                        ),
                    "extinction_day": int(
                        self.all_end_days[run]
                    ),
                    "stochastic_dieout": bool(
                        self.all_stochastic_dieout[
                            run
                        ]
                    ),
                    "first_detection_day":
                        first_detection,
                    "first_report_day":
                        first_report,
                    "first_imported_report_day":
                        first_imported_report,
                    "first_local_report_day":
                        first_local_report,
                    "first_investigation_day":
                        first_investigation,
                    "first_isolation_day":
                        first_isolation,
                    "first_contact_notification_day":
                        first_notification,
                    "transmissions_before_first_report":
                        self._count_exposures_before(
                            exposures,
                            first_report,
                        ),
                    "transmissions_before_first_intervention":
                        self._count_exposures_before(
                            exposures,
                            first_isolation,
                        ),
                    "transmissions_before_escalation":
                        self._count_exposures_before(
                            exposures,
                            escalation_day,
                        ),
                    "transmissions_after_escalation":
                        self._count_exposures_after(
                            exposures,
                            escalation_day,
                        ),
                    "total_work_units":
                        self._frame_sum(
                            run_lhd,
                            "capacity_used",
                        ),
                    "peak_daily_work_units":
                        self._frame_max_float(
                            run_lhd,
                            "capacity_used",
                        ),
                    "peak_pending_work_units":
                        self._frame_max_float(
                            run_lhd,
                            "pending_work_units_end",
                        ),
                    "days_at_full_capacity":
                        self._days_at_capacity(
                            run_lhd
                        ),
                    "mean_report_to_investigation_delay":
                        self._series_mean(
                            report_delays
                        ),
                    "mean_report_to_isolation_delay":
                        self._series_mean(
                            isolation_delays
                        ),
                    "mean_contact_notification_delay":
                        self._series_mean(
                            notification_delays
                        ),
                    "escalation_order_day":
                        self._first_nonmissing(
                            run_lhd,
                            "escalation_order_day",
                        ),
                    "escalation_effective_day":
                        escalation_day,
                    "maximum_generation": (
                        int(generations.max())
                        if np.any(
                            generations >= 0
                        )
                        else pd.NA
                    ),
                }
            )

        return pd.DataFrame(rows)

    def transmissions_to_df(
        self,
    ) -> pd.DataFrame:
        """Return one compact row per recorded local transmission."""
        events = self.event_log_to_df()
        exposures = events[
            events["event_type"] == "exposure"
        ].copy()

        columns = [
            "run_number",
            "day",
            "source_node_id",
            "node_id",
            "contact_type",
            "infection_generation",
            "policy_name",
            "policy_phase",
        ]

        return exposures[columns].rename(
            columns={
                "source_node_id": "source",
                "node_id": "target",
                "infection_generation":
                    "target_generation",
            }
        )

    def timeseries_to_df(
        self,
        type: str = "prevalence",
    ) -> pd.DataFrame:
        """Return fixed-horizon local incidence or prevalence trajectories."""
        series_type = str(type).lower().strip()

        if series_type not in {
            "incidence",
            "prevalence",
        }:
            raise ValueError(
                "type must be 'incidence' or "
                "'prevalence'"
            )

        horizon = self.Tmax + 1
        data = np.zeros(
            (
                self.n_replicates,
                horizon,
            ),
            dtype=np.float32,
        )

        value_column = (
            "local_infectious"
            if series_type == "prevalence"
            else "new_local_exposures"
        )

        for run in range(self.n_replicates):
            daily = self.all_daily_epi[run]

            if not isinstance(
                daily,
                pd.DataFrame,
            ):
                continue

            for row in daily.itertuples(
                index=False
            ):
                day = int(row.day)

                if 0 <= day < horizon:
                    data[run, day] = (
                        float(
                            getattr(
                                row,
                                value_column,
                            )
                        )
                        / float(self.N)
                    )

        frame = pd.DataFrame(
            data,
            columns=[
                f"t_{day}"
                for day in range(horizon)
            ],
        )
        frame.insert(
            0,
            "run_number",
            np.arange(
                self.n_replicates,
                dtype=np.int64,
            ),
        )

        return frame

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_event_day(
        events: pd.DataFrame,
        event_type: str,
        *,
        origin: Optional[str] = None,
    ):
        subset = events[
            events["event_type"] == event_type
        ]

        if origin is not None:
            subset = subset[
                subset["infection_origin"]
                == origin
            ]

        if subset.empty:
            return pd.NA

        return int(subset["day"].min())

    @staticmethod
    def _count_exposures_before(
        exposures: pd.DataFrame,
        day,
    ) -> int:
        if exposures.empty:
            return 0

        if pd.isna(day):
            return int(len(exposures))

        return int(
            (
                exposures["day"] < int(day)
            ).sum()
        )

    @staticmethod
    def _count_exposures_after(
        exposures: pd.DataFrame,
        day,
    ) -> int:
        if exposures.empty or pd.isna(day):
            return 0

        return int(
            (
                exposures["day"] >= int(day)
            ).sum()
        )

    @staticmethod
    def _task_delay_series(
        *,
        run_tasks: pd.DataFrame,
        run_events: pd.DataFrame,
        task_type: str,
    ) -> pd.Series:
        tasks = run_tasks[
            run_tasks["task_type"] == task_type
        ].copy()

        if tasks.empty:
            return pd.Series(dtype=float)

        report_days = (
            run_events[
                run_events["event_type"]
                == "report_received"
            ][["node_id", "day"]]
            .dropna()
            .drop_duplicates(
                subset=["node_id"],
                keep="first",
            )
            .set_index("node_id")["day"]
            .to_dict()
        )

        tasks["report_day"] = tasks[
            "target"
        ].map(report_days)

        return (
            pd.to_numeric(
                tasks["completed_day"],
                errors="coerce",
            )
            - pd.to_numeric(
                tasks["report_day"],
                errors="coerce",
            )
        ).dropna()

    @staticmethod
    def _notification_delay_series(
        run_tasks: pd.DataFrame,
    ) -> pd.Series:
        tasks = run_tasks[
            run_tasks["task_type"].isin(
                {
                    "contact_notification",
                    "automated_notification",
                }
            )
        ]

        if tasks.empty:
            return pd.Series(dtype=float)

        return (
            pd.to_numeric(
                tasks["completed_day"],
                errors="coerce",
            )
            - pd.to_numeric(
                tasks["created_day"],
                errors="coerce",
            )
        ).dropna()

    @staticmethod
    def _frame_sum(
        frame: pd.DataFrame,
        column: str,
    ) -> float:
        if frame.empty or column not in frame:
            return 0.0

        return float(
            pd.to_numeric(
                frame[column],
                errors="coerce",
            ).fillna(0).sum()
        )

    @staticmethod
    def _frame_max(
        frame: pd.DataFrame,
        column: str,
    ) -> int:
        if frame.empty or column not in frame:
            return 0

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).dropna()

        return (
            int(values.max())
            if not values.empty
            else 0
        )

    @staticmethod
    def _frame_max_float(
        frame: pd.DataFrame,
        column: str,
    ) -> float:
        if frame.empty or column not in frame:
            return 0.0

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).dropna()

        return (
            float(values.max())
            if not values.empty
            else 0.0
        )

    @staticmethod
    def _series_mean(
        series: pd.Series,
    ) -> float:
        return (
            float(series.mean())
            if not series.empty
            else np.nan
        )

    @staticmethod
    def _days_at_capacity(
        daily_lhd: pd.DataFrame,
    ) -> int:
        required = {
            "capacity_used",
            "capacity_available",
        }

        if (
            daily_lhd.empty
            or not required.issubset(
                daily_lhd.columns
            )
        ):
            return 0

        return int(
            (
                pd.to_numeric(
                    daily_lhd["capacity_used"],
                    errors="coerce",
                )
                >= pd.to_numeric(
                    daily_lhd[
                        "capacity_available"
                    ],
                    errors="coerce",
                ) - 1e-9
            ).sum()
        )

    @staticmethod
    def _first_nonmissing(
        frame: pd.DataFrame,
        column: str,
    ):
        if frame.empty or column not in frame:
            return pd.NA

        values = frame[column].dropna()

        if values.empty:
            return pd.NA

        return int(values.iloc[0])