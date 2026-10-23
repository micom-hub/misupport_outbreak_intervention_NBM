# scripts/simulation/outbreak_model.py
from __future__ import annotations

import os
import warnings
from typing import List, Optional

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from line_profiler import profile
from numba import njit
from numba.typed import List as NumbaList

from scripts.config import ModelConfig
from scripts.graph.graph_utils import GraphData
from scripts.lhd.lhd import LocalHealthDepartment
from scripts.recorder.event_log import EventLedger
from scripts.utils.rng_utility import derive_seed_from_base


# Define numba loop outside of model, called by determine_new_exposures
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
    susc_under5,
    susc_elderly,
    rel_inf_vax,
    infected_stamp,
    stamp_value,
    ct_ids,              
    trans_src_buf, 
    trans_tgt_buf, 
    trans_ct_buf,  
    legacy_seed
):
    # Seed legacy RNG
    np.random.seed(int(legacy_seed))

    n_ct = len(indptr_list)
    out_n = 0
    buf_cap = trans_src_buf.shape[0]

    for ii in range(infectious_indices.shape[0]):
        src = infectious_indices[ii]

        # compute per-src factors
        src_inf_mult = rel_inf_vax if is_vax[src] else 1.0

        for ct_ind in range(n_ct):
            indptr = indptr_list[ct_ind]
            indices = indices_list[ct_ind]
            weights = weights_list[ct_ind]

            start = indptr[src]
            end = indptr[src + 1]
            if end <= start:
                continue

            out_mult_src = out_mat[ct_ind, src]
            if out_mult_src <= 0.0:
                continue

            ct_id = ct_ids[ct_ind]

            for k in range(start, end):
                neighbor = indices[k]

                # ensure neighbor is susceptible
                if state[neighbor] != 0:
                    continue

                # ensure neighbor wasn't already infected this step
                if infected_stamp[neighbor] == stamp_value:
                    continue

                w = weights[k]
                in_mult_neighbor = in_mat[ct_ind, neighbor]
                effective_w = w * out_mult_src * in_mult_neighbor
                if effective_w <= 0.0:
                    continue

                prob = base_prob * effective_w * src_inf_mult

                if vax_efficacy != 0.0 and is_vax[neighbor]:
                    prob = prob * (1.0 - vax_efficacy)

                a = ages[neighbor]
                if a <= 5:
                    prob = prob * susc_under5
                elif a >= 65:
                    prob = prob * susc_elderly

                if prob <= 0.0:
                    continue
                if prob >=1.0:
                    prob = 1.0

                if np.random.random() < prob:
                    infected_stamp[neighbor] = stamp_value

                    trans_src_buf[out_n] = src
                    trans_tgt_buf[out_n] = neighbor
                    trans_ct_buf[out_n] = ct_id
                    out_n += 1

                    if out_n >= buf_cap:
                        return out_n
    return out_n


# -----Outbreak Model-------
class NetworkModel:
    def __init__(
        self,  
        config: ModelConfig,
        graphdata: GraphData,
        run_dir: str,
        *,
        seed: Optional[int] = None,
        ):
        """
        Unpack config and graphdata
        """

        self.config = config
        self.config.validate()

        self._assign_graphdata_to_model(graphdata)
        self.Tmax = self.config.sim.simulation_duration

        # Handle RNGs, seeds derived and assigned in _initialize_replicate
        if seed is not None:
            self._base_seed = int(seed)
        else:
            print("No seed passed to model, using configuration default")
            seed_from_params = self.config.sim.seed
            self._base_seed = int(seed_from_params)
        self.rng = None
        self.replicate_ind = None
        self.replicate_seed = None
        self._numba_legacy_seed_base = None
        self._seed_metadata = {}

        # run metadata:
        self.county = self.config.sim.county
        self.n_replicates = self.config.sim.n_replicates

        # Set up storage for full run
        self.all_states_over_time = [None]*self.n_replicates
        self.all_new_exposures = [None]*self.n_replicates
        self.all_transmissions = [None]*self.n_replicates
        self.all_surveillance_batches = [None] * self.n_replicates
        self.all_lhd_daily_logs = [None] * self.n_replicates
        self.all_stochastic_dieout = np.zeros(self.n_replicates, dtype = bool)
        self.all_end_days = np.ones(self.n_replicates, dtype = int)*self.Tmax
        self.all_lhd_results = [None] * self.n_replicates
        self.all_lhd_tasks = [None] * self.n_replicates
        self.all_vax_status = [None] * self.n_replicates
        # Unified event and daily epidemiological records.
        self.event_ledger = EventLedger()
        self.all_daily_epi = [None] * self.n_replicates
        self.all_imported_ids = [None] * self.n_replicates
        self.all_infection_generation = [None] * self.n_replicates
        self.all_infector = [None] * self.n_replicates
        self.all_infection_day = [None] * self.n_replicates
        self.all_infectious_onset_day = [None] * self.n_replicates
        self.all_symptom_onset_day = [None] * self.n_replicates
        self.all_recovery_day = [None] * self.n_replicates
        self.all_will_be_symptomatic = [None] * self.n_replicates
        # Imported infectious nodes are sampled independently for each replicate.
        self.all_imported_ids = [None] * self.n_replicates        

        # Results that are saved go to run_dir
        self.results_folder = run_dir 
        # Create if not created by driver
        if self.config.sim.save_data_files:
            os.makedirs(self.results_folder, exist_ok=True)

    def _assign_graphdata_to_model(self, graphdata: GraphData):
        # graphdata assumed to be a full GraphData object returned by build_graph_data
        self.N = graphdata.N
        self.edge_list = graphdata.edge_list
        self.adj_matrix = graphdata.adj_matrix
        self.individual_lookup = graphdata.individual_lookup
        self.ages = graphdata.ages.astype(np.int32)
        self.sexes = graphdata.sexes
        self.compliances = getattr(graphdata, "compliances", getattr(self, "compliances", None))
        self.neighbor_map = graphdata.neighbor_map
        self.fast_neighbor_map = graphdata.fast_neighbor_map
        self.csr_by_type = graphdata.csr_by_type
        self.contact_types = graphdata.contact_types
        self.ct_to_id = graphdata.ct_to_id
        self.id_to_ct = graphdata.id_to_ct
        self.full_node_list = graphdata.full_node_list
        self.degrees_arr = graphdata.degrees_arr
        # reinitialize in/out multipliers
        self.in_multiplier = {ct: np.ones(self.N, dtype=np.float32) for ct in self.contact_types}
        self.out_multiplier = {ct: np.ones(self.N, dtype=np.float32) for ct in self.contact_types}

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
        metadata: Optional[dict] = None,
        action_id: Optional[str] = None,
        task_id: Optional[str] = None,
        policy_phase: Optional[int] = None,
    ) -> int:
        """
        Append an event while automatically filling infection metadata from
        the current replicate when a node is supplied.
        """
        node = None if node_id is None else int(node_id)

        if node is not None:
            if (
                infection_generation is None
                and hasattr(self, "infection_generation")
                and 0 <= node < self.N
            ):
                generation = int(self.infection_generation[node])
                if generation >= 0:
                    infection_generation = generation

            if (
                infection_origin is None
                and hasattr(self, "infection_generation")
                and 0 <= node < self.N
            ):
                generation = int(self.infection_generation[node])
                if generation == 0:
                    infection_origin = "imported"
                elif generation > 0:
                    infection_origin = "local"

            if (
                symptomatic is None
                and hasattr(self, "is_symptomatic")
                and 0 <= node < self.N
            ):
                symptomatic = bool(self.is_symptomatic[node])

        policy_name = None
        if hasattr(self, "lhd") and self.lhd is not None:
            policy_name = getattr(self.lhd, "policy_name", None)

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
        frames = [
            df
            for df in self.all_daily_epi
            if isinstance(df, pd.DataFrame) and not df.empty
        ]

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True, sort=False)

    def _initialize_replicate(self, run: int):
        """
        Initialize one stochastic replicate.

        Imported cases begin infectious on day zero. All other people begin
        susceptible. Symptomatic disposition and symptom-onset delays are
        preassigned for all nodes so they remain stable within the replicate.
        """
        self.current_time = 0
        self.replicate_ind = int(run)

        self.replicate_seed = int(
            derive_seed_from_base(
                self._base_seed,
                self.replicate_ind,
            )
        )
        self.rng = np.random.default_rng(self.replicate_seed)

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

        self._seed_metadata = {
            "base_seed": int(self._base_seed),
            "replicate_ind": int(self.replicate_ind),
            "replicate_seed": int(self.replicate_seed),
            "lhd_seed": int(self.lhd_seed),
            "surveillance_seed": int(self.surv_seed),
        }

        self._numba_legacy_seed_base = int(
            derive_seed_from_base(
                self.replicate_seed,
                "numba",
            )
        )

        # Vaccination
        self.is_vaccinated = (
            self.rng.random(self.N)
            < float(self.config.epi.vax_uptake)
        )

        # SEIR state: S=0, E=1, I=2, R=3
        self.state = np.zeros(self.N, dtype=np.int8)
        self.time_in_state = np.zeros(self.N, dtype=np.float32)
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

        # Infection history
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

        # Symptom disposition is assigned independently of whether infection
        # eventually occurs.
        self.will_be_symptomatic = (
            self.rng.random(self.N)
            < float(self.config.epi.symptomatic_fraction)
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

        # Per-replicate trajectories
        self.states_over_time = []
        self.new_exposures = []
        self.new_infections = []
        self.daily_epi_rows = []

        # Imported cases
        n_imported = int(self.config.sim.I0)
        if n_imported > self.N:
            raise ValueError(
                f"sim.I0 ({n_imported}) cannot exceed N ({self.N})"
            )

        if n_imported > 0:
            imported_ids = np.sort(
                self.rng.choice(
                    self.N,
                    size=n_imported,
                    replace=False,
                ).astype(np.int32)
            )
        else:
            imported_ids = np.empty(0, dtype=np.int32)

        self.imported_ids = imported_ids
        self.I0 = imported_ids.tolist()
        self.all_imported_ids[self.replicate_ind] = (
            imported_ids.copy()
        )

        if imported_ids.size:
            self.state[imported_ids] = 2
            self.infection_day[imported_ids] = 0
            self.infectious_onset_day[imported_ids] = 0
            self.infection_generation[imported_ids] = 0

            self.infectious_periods[imported_ids] = (
                self.assign_infectious_period(imported_ids)
            )

            symptomatic_imports = imported_ids[
                self.will_be_symptomatic[imported_ids]
            ]

            if symptomatic_imports.size:
                self.symptom_onset_day[symptomatic_imports] = (
                    self.symptom_onset_delays[
                        symptomatic_imports
                    ]
                )

                onset_day_zero = symptomatic_imports[
                    self.symptom_onset_day[
                        symptomatic_imports
                    ] <= 0
                ]
                self.is_symptomatic[onset_day_zero] = True

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

            for node in np.where(self.is_symptomatic)[0]:
                self.record_event(
                    day=0,
                    event_type="symptom_onset",
                    node_id=int(node),
                    disease_stage="infectious",
                    symptomatic=True,
                )

        imported_set = set(imported_ids.tolist())

        susceptible = [
            node
            for node in range(self.N)
            if node not in imported_set
        ]
        exposed = []
        infectious = imported_ids.tolist()
        recovered = []

        self.states_over_time.append(
            [susceptible, exposed, infectious, recovered]
        )
        self.new_exposures.append(
            np.empty(0, dtype=np.int32)
        )
        self.new_infections.append(
            np.empty(0, dtype=np.int32)
        )

        self.simulation_end_day = self.Tmax
        self.stochastic_dieout = False

        self.in_multiplier = {
            ct: np.ones(self.N, dtype=np.float32)
            for ct in self.contact_types
        }
        self.out_multiplier = {
            ct: np.ones(self.N, dtype=np.float32)
            for ct in self.contact_types
        }

        if (
            not hasattr(self, "_trans_src_buf")
            or self._trans_src_buf.shape[0] != self.N
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
            or self._infected_stamp.shape[0] != self.N
        ):
            self._infected_stamp = np.full(
                self.N,
                -1,
                dtype=np.int32,
            )

        self._infected_stamp.fill(-1)

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
                np.count_nonzero(self.is_symptomatic)
            ),
        )
   
    def assign_latent_period(self, inds):
        """
        Assign E -> I durations to newly exposed nodes.
        """
        inds = np.atleast_1d(inds).astype(
            np.int32,
            copy=False,
        )

        is_vax_inds = self.is_vaccinated[inds]

        mean_latent = np.where(
            is_vax_inds,
            float(self.config.epi.latent_period_vax),
            float(self.config.epi.latent_period),
        )

        shape = float(self.config.epi.gamma_alpha)

        return self.rng.gamma(
            shape=shape,
            scale=mean_latent / shape,
        )


    def assign_infectious_period(self, inds):
        """
        Assign infectious durations to newly infectious nodes.
        """
        inds = np.atleast_1d(inds).astype(
            np.int32,
            copy=False,
        )

        is_vax_inds = self.is_vaccinated[inds]

        mean_infectious = np.where(
            is_vax_inds,
            float(self.config.epi.infectious_period_vax),
            float(self.config.epi.infectious_period),
        )

        shape = float(self.config.epi.gamma_alpha)

        return self.rng.gamma(
            shape=shape,
            scale=mean_infectious / shape,
        )

    def _prepare_numba_structures(self):
        """
        Convert self.csr_by_type and multiplier dicts into Numba-friendly structures
        (typed lists and contiguous arrays). Called once (or at first use).
        """
        contact_types = sorted(self.contact_types)
        self._numba_contact_types = contact_types
        self._numba_ct_ids = np.array([self.ct_to_id[ct] for ct in contact_types], dtype=np.int16)
        n_ct = len(contact_types)
        N = self.N

        # Build typed lists for indptr, indices, weights (ct order must match contact_types)
        indptr_list = NumbaList()
        indices_list = NumbaList()
        weights_list = NumbaList()

        for ct in contact_types:
            indptr, indices, weights = self.csr_by_type[ct]
            indptr_list.append(np.ascontiguousarray(indptr.astype(np.int64)))
            indices_list.append(np.ascontiguousarray(indices.astype(np.int64)))
            weights_list.append(np.ascontiguousarray(weights.astype(np.float32)))

        # Multiplier matrices (n_ct x N). We'll update these in place each step.
        out_mat = np.empty((n_ct, N), dtype=np.float32)
        in_mat = np.empty((n_ct, N), dtype=np.float32)
        for j, ct in enumerate(contact_types):
            out_mat[j, :] = np.ascontiguousarray(self.out_multiplier[ct].astype(np.float32))
            in_mat[j, :] = np.ascontiguousarray(self.in_multiplier[ct].astype(np.float32))

        # store
        self._numba_indptr_list = indptr_list
        self._numba_indices_list = indices_list
        self._numba_weights_list = weights_list
        self._numba_out_mat = out_mat
        self._numba_in_mat = in_mat
        self._numba_ready = True

    def _update_multiplier_matrices(self):
        """
        Update the in/out multiplier matrices in-place from current dicts.
        Call before each call into the njit function if multipliers may have changed.
        """
        for j, ct in enumerate(self._numba_contact_types):
            # in-place copy to avoid reallocations
            self._numba_out_mat[j, :] = self.out_multiplier[ct].astype(np.float32)
            self._numba_in_mat[j, :] = self.in_multiplier[ct].astype(np.float32)

    @profile     
    def determine_transmissions(self):
        """
        Wrapper for _determine_transmissions_numba that calls the numba compiled function, and returns newly-exposed individuals

        """
        infectious_indices = np.where(self.state == 2)[0]
        if infectious_indices.size == 0:
            return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int16))

        base_prob = float(self.config.epi.base_transmission_prob)
        vax_efficacy = float(self.config.epi.vax_efficacy)
        susc_under5 = float(self.config.epi.susceptibility_multiplier_under_five)
        susc_elderly = float(self.config.epi.susceptibility_multiplier_elderly)
        rel_inf_vax = float(self.config.epi.relative_infectiousness_vax)

        if not getattr(self, "_numba_ready", False):
            self._prepare_numba_structures()

        if not getattr(self, "_numba_contact_types", None) or len(self._numba_contact_types) == 0:
            return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int16))

        self._update_multiplier_matrices()

        stamp_value = np.int32(self.current_time) #check for same-day inf

        legacy_seed = int(derive_seed_from_base(self._numba_legacy_seed_base, int(self.current_time)))

        tot = _determine_transmissions_numba(
            self.state,                      
            infectious_indices,
            self._numba_indptr_list,
            self._numba_indices_list,
            self._numba_weights_list,
            self._numba_out_mat,
            self._numba_in_mat,
            self.is_vaccinated,
            self.ages,    
            base_prob,
            vax_efficacy,
            susc_under5,
            susc_elderly,
            rel_inf_vax,
            self._infected_stamp,
            stamp_value,
            self._numba_ct_ids,
            self._trans_src_buf,
            self._trans_tgt_buf,
            self._trans_ct_buf,
            legacy_seed
        )

        # Gather new transmission info
        src = self._trans_src_buf[:tot]
        tgt = self._trans_tgt_buf[:tot]
        ct  = self._trans_ct_buf[:tot]
        return src, tgt, ct

    @profile
    def step(self):
        """
        Advance the epidemic by one day.

        Daily order:
          1. Infectious nodes transmit.
          2. New local exposures enter E.
          3. E nodes may enter I.
          4. I nodes may recover.
          5. Symptom onsets are updated.
          6. End-of-step epidemiological truth is assembled for surveillance.

        LHD surveillance and intervention occur after this method, so an action
        selected on day t first affects transmission on day t+1.
        """
        day = int(self.current_time)

        # 1. Transmission
        trans_src, newly_exposed, trans_ct = (
            self.determine_transmissions()
        )

        newly_exposed = np.asarray(
            newly_exposed,
            dtype=np.int32,
        )
        trans_src = np.asarray(
            trans_src,
            dtype=np.int32,
        )
        trans_ct = np.asarray(
            trans_ct,
            dtype=np.int16,
        )

        # 2. S -> E
        if newly_exposed.size > 0:
            self.state[newly_exposed] = 1
            self.time_in_state[newly_exposed] = -1
            self.latent_periods[newly_exposed] = (
                self.assign_latent_period(newly_exposed)
            )
            self.infection_day[newly_exposed] = day

            source_generations = self.infection_generation[
                trans_src
            ]
            self.infection_generation[newly_exposed] = (
                source_generations + 1
            )
            self.infector[newly_exposed] = trans_src
            self.infection_contact_type[newly_exposed] = trans_ct

            for source, target, contact_type_id in zip(
                trans_src,
                newly_exposed,
                trans_ct,
            ):
                source = int(source)
                target = int(target)
                contact_type_id = int(contact_type_id)

                contact_type = self.id_to_ct.get(
                    contact_type_id,
                    str(contact_type_id),
                )

                self.record_event(
                    day=day,
                    event_type="exposure",
                    node_id=target,
                    source_node_id=source,
                    infection_origin="local",
                    infection_generation=int(
                        self.infection_generation[target]
                    ),
                    disease_stage="exposed",
                    contact_type=contact_type,
                )

        # 3. E -> I
        exposed = np.where(self.state == 1)[0]
        self.time_in_state[exposed] += 1

        to_infectious = exposed[
            self.time_in_state[exposed]
            >= self.latent_periods[exposed]
        ].astype(np.int32, copy=False)

        if to_infectious.size > 0:
            self.state[to_infectious] = 2
            self.time_in_state[to_infectious] = -1
            self.infectious_onset_day[to_infectious] = day

            self.infectious_periods[to_infectious] = (
                self.assign_infectious_period(to_infectious)
            )

            symptomatic_nodes = to_infectious[
                self.will_be_symptomatic[to_infectious]
            ]

            if symptomatic_nodes.size:
                self.symptom_onset_day[symptomatic_nodes] = (
                    day
                    + self.symptom_onset_delays[
                        symptomatic_nodes
                    ]
                )

            for node in to_infectious:
                self.record_event(
                    day=day,
                    event_type="infectious_onset",
                    node_id=int(node),
                    disease_stage="infectious",
                )

        # 4. I -> R
        infectious = np.where(self.state == 2)[0]
        self.time_in_state[infectious] += 1

        to_recovered = infectious[
            self.time_in_state[infectious]
            >= self.infectious_periods[infectious]
        ].astype(np.int32, copy=False)

        if to_recovered.size > 0:
            self.state[to_recovered] = 3
            self.time_in_state[to_recovered] = 0
            self.recovery_day[to_recovered] = day

            for node in to_recovered:
                self.record_event(
                    day=day,
                    event_type="recovery",
                    node_id=int(node),
                    disease_stage="recovered",
                )

        # Recovered nodes remain recovered for the early-outbreak horizon.
        recovered = np.where(self.state == 3)[0]
        self.time_in_state[recovered] += 1

        # 5. Symptom onset
        symptom_candidates = np.where(
            (self.state == 2)
            & self.will_be_symptomatic
            & (~self.is_symptomatic)
            & (self.symptom_onset_day >= 0)
            & (self.symptom_onset_day <= day)
        )[0].astype(np.int32, copy=False)

        if symptom_candidates.size > 0:
            self.is_symptomatic[symptom_candidates] = True

            for node in symptom_candidates:
                self.record_event(
                    day=day,
                    event_type="symptom_onset",
                    node_id=int(node),
                    disease_stage="infectious",
                    symptomatic=True,
                )

        # 6. Save truth state
        susceptible_nodes = np.where(self.state == 0)[0]
        exposed_nodes = np.where(self.state == 1)[0]
        infectious_nodes = np.where(self.state == 2)[0]
        recovered_nodes = np.where(self.state == 3)[0]

        self.states_over_time.append(
            [
                susceptible_nodes.tolist(),
                exposed_nodes.tolist(),
                infectious_nodes.tolist(),
                recovered_nodes.tolist(),
            ]
        )
        self.new_exposures.append(newly_exposed.copy())
        self.new_infections.append(to_infectious.copy())

        epi_state = {
            "new_pre_ids": newly_exposed.copy(),
            "new_inf_ids": to_infectious.copy(),
            "pre_ids": exposed_nodes.astype(
                np.int32,
                copy=False,
            ),
            "inf_ids": infectious_nodes.astype(
                np.int32,
                copy=False,
            ),
            "symptomatic_ids": np.where(
                self.is_symptomatic
                & (self.state == 2)
            )[0].astype(np.int32, copy=False),
            "imported_ids": self.imported_ids.copy(),
            "infection_generation":
                self.infection_generation.copy(),
        }

        return (
            trans_src,
            newly_exposed,
            trans_ct,
            epi_state,
            int(symptom_candidates.size),
        )

    def simulate(self):
        for run in range(self.n_replicates):
            transmissions_over_time = []
            surveillance_over_time = []

            self._initialize_replicate(run)

            self.all_vax_status[run] = (
                self.is_vaccinated.copy()
            )

            day = 0

            while day < self.Tmax:
                day += 1
                self.current_time = day

                (
                    trans_src,
                    trans_tgt,
                    trans_ct,
                    epi_state,
                    new_symptom_onsets,
                ) = self.step()

                transmissions_over_time.append(
                    {
                        "time": np.int32(day),
                        "src": trans_src.copy(),
                        "tgt": trans_tgt.copy(),
                        "ct": trans_ct.copy(),
                    }
                )

                batch = self.lhd.step(
                    t=day,
                    epi_state=epi_state,
                )
                surveillance_over_time.append(batch)

                self._append_daily_epi(
                    day=day,
                    batch=batch,
                    new_local_exposures=int(
                        trans_tgt.size
                    ),
                    new_infectious_onsets=int(
                        epi_state["new_inf_ids"].size
                    ),
                    new_symptom_onsets=int(
                        new_symptom_onsets
                    ),
                )

                _, exposed, infectious, _ = (
                    self.states_over_time[-1]
                )

                if not exposed and not infectious:
                    self.simulation_end_day = day
                    self.stochastic_dieout = True
                    break

            self.all_transmissions[run] = (
                transmissions_over_time
            )
            self.all_states_over_time[run] = [
                states.copy()
                for states in self.states_over_time
            ]
            self.all_stochastic_dieout[run] = (
                self.stochastic_dieout
            )
            self.all_end_days[run] = (
                self.simulation_end_day
            )
            self.all_new_exposures[run] = [
                exposures.copy()
                for exposures in self.new_exposures
            ]
            self.all_surveillance_batches[run] = (
                surveillance_over_time
            )
            self.all_lhd_results[run] = (
                self.lhd.results_to_df()
            )
            self.all_lhd_daily_logs[run] = (
                self.lhd.lhd_daily_log_to_df()
            )
            self.all_lhd_tasks[run] = (
                self.lhd.tasks_to_df()
            )
            self.all_daily_epi[run] = pd.DataFrame(
                self.daily_epi_rows
            )

            self.all_infection_generation[run] = (
                self.infection_generation.copy()
            )
            self.all_infector[run] = (
                self.infector.copy()
            )
            self.all_infection_day[run] = (
                self.infection_day.copy()
            )
            self.all_infectious_onset_day[run] = (
                self.infectious_onset_day.copy()
            )
            self.all_symptom_onset_day[run] = (
                self.symptom_onset_day.copy()
            )
            self.all_recovery_day[run] = (
                self.recovery_day.copy()
            )
            self.all_will_be_symptomatic[run] = (
                self.will_be_symptomatic.copy()
            )

    def _append_daily_epi(
        self,
        *,
        day: int,
        batch: Optional[dict],
        new_local_exposures: int,
        new_infectious_onsets: int,
        new_symptom_onsets: int,
    ) -> None:
        """
        Append one raw daily epidemiological summary row.
        """
        local_mask = self.infection_generation > 0
        imported_mask = self.infection_generation == 0

        local_exposed = int(
            np.count_nonzero(
                (self.state == 1) & local_mask
            )
        )
        local_infectious = int(
            np.count_nonzero(
                (self.state == 2) & local_mask
            )
        )
        local_recovered = int(
            np.count_nonzero(
                (self.state == 3) & local_mask
            )
        )

        imported_infectious = int(
            np.count_nonzero(
                (self.state == 2) & imported_mask
            )
        )

        cumulative_local_infections = int(
            np.count_nonzero(local_mask)
        )

        currently_symptomatic = int(
            np.count_nonzero(
                self.is_symptomatic
                & (self.state == 2)
            )
        )

        if batch is None:
            new_detections = 0
            new_reports = 0
            new_tests_completed = 0
            new_positive_tests = 0
        else:
            new_detections = int(
                np.asarray(
                    batch.get(
                        "detected_cases",
                        np.empty(0, np.int32),
                    )
                ).size
            )
            new_reports = int(
                np.asarray(
                    batch.get(
                        "reported_cases",
                        np.empty(0, np.int32),
                    )
                ).size
            )
            new_tests_completed = int(
                np.asarray(
                    batch.get(
                        "tests_completed",
                        np.empty(0, np.int32),
                    )
                ).size
            )
            new_positive_tests = int(
                np.count_nonzero(
                    np.asarray(
                        batch.get(
                            "test_results_positive",
                            np.empty(0, dtype=bool),
                        ),
                        dtype=bool,
                    )
                )
            )

        row = {
            "run_number": int(self.replicate_ind),
            "day": int(day),
            "susceptible": int(
                np.count_nonzero(self.state == 0)
            ),
            "local_exposed": local_exposed,
            "local_infectious": local_infectious,
            "local_recovered": local_recovered,
            "imported_infectious": imported_infectious,
            "new_local_exposures": int(
                new_local_exposures
            ),
            "new_local_infectious_onsets": int(
                new_infectious_onsets
            ),
            "cumulative_local_infections":
                cumulative_local_infections,
            "new_symptom_onsets": int(
                new_symptom_onsets
            ),
            "currently_symptomatic":
                currently_symptomatic,
            "new_detections": new_detections,
            "new_reports_received": new_reports,
            "new_tests_completed": new_tests_completed,
            "new_positive_tests": new_positive_tests,
        }

        self.daily_epi_rows.append(row)

    def detection_summary_to_df(self) -> pd.DataFrame:
        """
        Summarize detection by imported/local origin and symptomatic disposition.
        """
        events = self.event_log_to_df()

        detection_events = events[
            events["event_type"] == "detection"
        ].copy()

        rows = []

        for run in range(self.n_replicates):
            generations = self.all_infection_generation[run]
            will_symptomatic = (
                self.all_will_be_symptomatic[run]
            )

            if generations is None:
                continue

            infected = np.where(generations >= 0)[0]
            imported = np.where(generations == 0)[0]
            local = np.where(generations > 0)[0]

            run_detections = detection_events[
                detection_events["run_number"] == run
            ]
            detected_nodes = set(
                run_detections["node_id"]
                .dropna()
                .astype(int)
                .tolist()
            )

            groups = {
                "all_infected": infected,
                "imported": imported,
                "local": local,
                "symptomatic_disposition": infected[
                    will_symptomatic[infected]
                ],
                "asymptomatic_disposition": infected[
                    ~will_symptomatic[infected]
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
                            float(detected_count)
                            / float(total)
                            if total > 0
                            else np.nan
                        ),
                    }
                )

        return pd.DataFrame(rows)

    def lhd_tasks_to_df(self) -> pd.DataFrame:
        frames = [
            frame
            for frame in self.all_lhd_tasks
            if isinstance(frame, pd.DataFrame)
            and not frame.empty
        ]

        if not frames:
            return pd.DataFrame()

        return pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )


    def lhd_daily_to_df(self) -> pd.DataFrame:
        frames = [
            frame
            for frame in self.all_lhd_daily_logs
            if isinstance(frame, pd.DataFrame)
            and not frame.empty
        ]

        if not frames:
            return pd.DataFrame()

        return pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )

    def results_to_df(self, metrics: List[str] = ["peakPrev", "peakTime", "outbreakSize"]) -> pd.DataFrame:
        """
        Writes a dataframe of summary metrics for each run containing specified metrics

        Args: metrics, a list of metrics to output, must match expected metrics allowed for. Currently supports:
        - peakPrev (maximum fraction I/N)
        - peakTime (time of peakPrev)
        - outbreakSize (unique locally acquired infections, excluding imported I0 cases)
        Returns: pandas dataframe of summary outputs
        """
        if metrics is None:
            metrics = ["peakPrev", "peakTime", "outbreakSize"]

        # Check that metrics requested are supported
        metrics_supported = {"peakPrev", "peakTime", "outbreakSize"}
        unknown = [metric for metric in metrics if metric not in metrics_supported]

        if unknown:
            warnings.warn(f"Unknown metric requested: {unknown}. Please select one of {metrics_supported}")

        def _aggregate_exposures(exposures_list):
            """
            helper to take a list of exposures and aggregated exposures
            """
            if not exposures_list:
                return np.empty(0, dtype=np.int32)
            exposures = []
            for arr in exposures_list:
                if arr is None:
                    continue
                try:
                    a = np.asarray(list(arr), dtype = np.int32)
                except Exception:
                    try:
                        a = np.asarray(list(arr), dtype = np.int32)
                    except Exception:
                        continue
                if a.size > 0:
                    exposures.append(a)

            if not exposures:
                return np.empty(0, dtype = np.int32)
            if len(exposures) == 1:
                return exposures[0]
            return np.concatenate(exposures)

        rows = []
        n_runs = self.n_replicates
        for run in range(n_runs):
            row = {"run_number": int(run)}

            states_list = self.all_states_over_time[run] or []

            imported_ids = np.asarray(
            self.all_imported_ids[run]
            if self.all_imported_ids[run] is not None
            else np.empty(0, np.int32),
            dtype=np.int32,
        )
            imported_set = set(imported_ids.tolist())

            # Build prevalence time series
            prevalences = []
            for timestep in states_list:
                if timestep is None:
                    nI = 0
                else:
                    try:
                        # Gather infectious nodes
                        I_nodes = timestep[2]
                        nI = int(sum(1 for node in I_nodes if int(node) not in imported_set))
                    except Exception:
                        nI = 0
                prevalences.append(float(nI) / float(self.N))

            # Metrics:

            # peakPrev & peakTime
            if "peakPrev" in metrics or "peakTime" in metrics:
                if prevalences:
                    arr = np.asarray(prevalences, dtype = float)
                    ind_max = int(np.argmax(arr))
                    peakPrev_val = float(arr[ind_max])
                    peakTime_val = int(ind_max)
                else:
                    peakPrev_val = 0.0
                    peakTime_val = pd.NA
                if "peakPrev" in metrics:
                    row["peakPrev"] = peakPrev_val
                if "peakTime" in metrics:
                    row["peakTime"] = peakTime_val

            # Outbreak size is the number of unique locally acquired infections.
            # Imported cases are fixed introductions and are not counted as outcomes.
            if "outbreakSize" in metrics:
                exposures_list = self.all_new_exposures[run]
                aggregate = _aggregate_exposures(exposures_list)

                if aggregate.size:
                    local_infections = np.setdiff1d(
                        np.unique(aggregate),
                        imported_ids,
                        assume_unique=False,
                    )
                    outbreak_size = int(local_infections.size)
                else:
                    outbreak_size = 0

                row["outbreakSize"] = outbreak_size

            # Add row to rows
            rows.append(row)

        # Build and order dataframe, handle typing
        cols = ["run_number"] + [metric for metric in metrics]
        df = pd.DataFrame(rows)
        if "peakTime" in metrics and "peakTime" in df.columns:
            df["peakTime"] = df["peakTime"].astype("Int64")
        if "peakPrev" in metrics and "peakPrev" in df.columns:
            df["peakPrev"] = df["peakPrev"].astype(float)
        if "outbreakSize" in metrics and "outbreakSize" in df.columns:
            df["outbreakSize"] = df["outbreakSize"].astype("Int64")
        # Put cols in order, NA if requested column wasn't calculated
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA

        return df[cols]

    def timeseries_to_df(
        self,
        type: str = "prevalence",
    ) -> pd.DataFrame:
        """
        Return one wide time-series row for each stochastic replicate.

        Runs that end before Tmax are padded with zeros through the fixed output
        horizon of day 0 through day Tmax.

        Args:
            type:
                "prevalence":
                    Fraction of the population currently infectious, excluding
                    imported cases.

                "incidence":
                    Fraction of the population newly exposed on each day.
                    Imported cases are not counted as incident local infections,
                    so incidence at t_0 is zero.

        Returns:
            DataFrame with one row per stochastic replicate and columns:
            run_number, t_0, t_1, ..., t_Tmax.
        """
        series_type = str(type).strip().lower()

        if series_type not in ("incidence", "prevalence"):
            raise ValueError(
                "type must be 'incidence' or 'prevalence' "
                f"(got {type})"
            )

        n_runs = int(self.n_replicates)

        # Fixed horizon across all runs, including day zero.
        Tmax = int(self.Tmax)
        T = Tmax + 1

        if T <= 0:
            return pd.DataFrame(
                {
                    "run_number": np.arange(
                        n_runs,
                        dtype=np.int64,
                    )
                }
            )

        data = np.zeros((n_runs, T), dtype=np.float32)

        if series_type == "prevalence":
            for run in range(n_runs):
                states_list = self.all_states_over_time[run] or []
                max_t = min(T, len(states_list))

                imported_ids = np.asarray(
                    (
                        self.all_imported_ids[run]
                        if self.all_imported_ids[run] is not None
                        else np.empty(0, dtype=np.int32)
                    ),
                    dtype=np.int32,
                )
                imported_set = set(imported_ids.tolist())

                for t in range(max_t):
                    timestep = states_list[t]

                    try:
                        infectious_nodes = timestep[2]
                    except (IndexError, TypeError):
                        infectious_nodes = []

                    try:
                        n_infectious_local = sum(
                            1
                            for node in infectious_nodes
                            if int(node) not in imported_set
                        )
                    except (TypeError, ValueError):
                        n_infectious_local = 0

                    data[run, t] = (
                        float(n_infectious_local) / float(self.N)
                    )

        else:  # incidence
            for run in range(n_runs):
                exposures_list = self.all_new_exposures[run] or []
                max_t = min(T, len(exposures_list))

                for t in range(max_t):
                    exposures = exposures_list[t]

                    if exposures is None:
                        count = 0
                    else:
                        try:
                            count = int(np.asarray(exposures).size)
                        except (TypeError, ValueError):
                            count = 0

                    # Imported I0 cases are not locally acquired infections.
                    # Consequently, t_0 remains zero.
                    data[run, t] = float(count) / float(self.N)

        column_names = [f"t_{t}" for t in range(T)]
        df = pd.DataFrame(data, columns=column_names)

        df.insert(
            0,
            "run_number",
            np.arange(n_runs, dtype=np.int64),
        )

        return df

    def draw_network(self, t: int, run_number = 0, ax=None, clear: bool =True, saveFile: bool = False, suffix: str = None):
        """
        Draws a network containing outbreak-involved nodes (E,I,R) + neighbors at time t
        """

        # get indices of E, I, R
        S, E, I, R = self.all_states_over_time[run_number][t] # noqa: E741
        affected_inds = set(E) | set(I) | set(R) 

        # neighbors of affected nodes
        neighbors = set()
        for ind in affected_inds:
            for (nbr, w, ct) in self.neighbor_map.get(int(ind), []):
                neighbors.add(int(nbr))

        # combine affected nodes and neighbors
        plot_nodes = sorted(affected_inds | neighbors)

        # build subgraph
        name_to_ind = {name: idx for idx, name in enumerate(plot_nodes)}
        subg = ig.Graph()
        subg.add_vertices(len(plot_nodes))
        subg.vs["name"] = plot_nodes

        # add edges
        edges = []
        for src in plot_nodes:
            for (tgt, w, ct) in self.neighbor_map.get(int(src), []):
                if int(tgt) in name_to_ind:
                    edges.append((name_to_ind[src],name_to_ind[int(tgt)]))

        if edges:
            subg.add_edges(edges)

        # color by state
        color_map = {}
        for v in subg.vs:
            name = v["name"]
            if name in S:
                color_map[name] = "blue"
            elif name in E:
                color_map[name] = "orange"

            elif name in I:
                color_map[name] = "red"

            elif name in R:
                color_map[name] = "green"

            else:
                color_map[name] = "gray"

        colors = [color_map[v["name"]] for v in subg.vs] 

        layout = subg.layout("fr")

        # plot
        if ax is None:
            fig, ax = plt.subplots(figsize = (10, 10))
            show_plot = True
        else:
            show_plot = False
        if clear:
            ax.clear()

        ig.plot(
            subg,
            layout = layout,
            vertex_color = colors,
            vertex_size = 15,
            edge_color = "gray",
            bbox = (600, 600),
            target = ax,
            vertex_label = subg.vs["name"] if len(subg.vs) <= 40 else None
        )
        ax.set_title(f"Network at t = {t}")

        # Save if requested
        if saveFile:
            plotpath = os.path.join(self.results_folder, f"network_at_{str(t)}")
            if suffix:
                plotpath = plotpath + suffix
            plt.savefig(f"{plotpath}.png")
        if show_plot and self.config.sim.display_plots:
            plt.show()
        plt.close()

        return
