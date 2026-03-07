

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import igraph as ig
from typing import Dict, Any, Optional, Callable
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from numba import njit
from numba.typed import List as NumbaList


from scripts.config import ModelConfig
from scripts.recorder.recorder import ExposureEventRecorder
from scripts.lhd.lhd import LocalHealthDepartment
from scripts.graph.graph_utils import GraphData 

#Define numba loop outside of model, called by determine_new_exposures
@njit
def _determine_new_exposures_numba(
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
rel_inf_vax
):
    N = state.shape[0]
    newly_exposed = np.zeros(N, dtype = np.bool_)
    n_ct = len(indptr_list)

    #loop over infectious nodes
    for infected_ind in range(infectious_indices.shape[0]):
        src = infectious_indices[infected_ind]
        for ct_ind in range(n_ct):
            indptr = indptr_list[ct_ind]
            indices = indices_list[ct_ind]
            weights = weights_list[ct_ind]

            start = indptr[src]
            end = indptr[src + 1]
            if end <= start:
                continue

            out_mult_src = out_mat[ct_ind, src]

            #calculate effective transmission weight
            for k in range(start, end):
                nbr = indices[k]
                if state[nbr] != 0:
                    continue
                w = weights[k]
                in_mult_nbr = in_mat[ct_ind, nbr]
                effective_w = w*out_mult_src*in_mult_nbr

                prob = base_prob*effective_w
                if is_vax[src]:
                    prob = prob*rel_inf_vax
                if vax_efficacy != 0.0:
                    if is_vax[nbr]:
                        prob = prob*(1.0-vax_efficacy)
                if ages[nbr] <= 5:
                    prob = prob * susc_under5
                elif ages[nbr] >= 65:
                    prob = prob * susc_elderly

                if prob <= 0.0:
                    continue
                if prob > 1.0:
                    prob = 1.0
                
                if np.random.random() < prob:
                    newly_exposed[nbr] = True
    return newly_exposed



#-----Outbreak Model-------
class NetworkModel:
    #@profilesave_
    def __init__(
        self,  
        config: ModelConfig,
        graphdata: GraphData,
        run_dir: str,
        *,
        rng = None,
        seed: Optional[int] = None,
        lhd_register_defaults: bool = True,
        lhd_algorithm_map: Optional[Dict[str, object]] = None,
        lhd_action_factory_map: Optional[Dict[str, Callable[..., Any]]] = None
        ):
        """
        Unpack config and graphdata
        """
        
        self.config = config
        self.config.validate()
        
        self._assign_graphdata_to_model(graphdata)
        self.Tmax = self.config.sim.simulation_duration

        #choose RNG: explicit rng > seed arg > params['seed] 
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(int(seed))
        else:
            seed_from_params = self.config.sim.seed
            self.rng = np.random.default_rng(int(seed_from_params))

        #run metadata:
        self.county = self.config.sim.county
        self.n_replicates = self.config.sim.n_replicates


        #Set up storage for full run
        self.all_states_over_time = [None]*self.n_replicates
        self.all_new_exposures = [None]*self.n_replicates
        self.exposure_event_log = [[] for _ in range(self.n_replicates)]
        self.all_stochastic_dieout = np.zeros(self.n_replicates, dtype = bool)
        self.all_end_days = np.ones(self.n_replicates, dtype = int)*self.Tmax

        self.all_vax_status = [None] * self.n_replicates
        self.all_lhd_action_logs = [None] * self.n_replicates

        #Instantiate recorder and Local Health Department
        self.recorder_template = ExposureEventRecorder(init_event_cap = 1024, init_node_cap = 4096)

        #model caller sets lhd mappings 
        self.lhd = LocalHealthDepartment(
            model = self,
            rng = self.rng,
            discovery_prob = self.config.lhd.lhd_discovery_prob,
            employees = self.config.lhd.lhd_employees,
            workday_hrs = self.config.lhd.lhd_workday_hrs,
            register_defaults = lhd_register_defaults,
            algorithm_map = lhd_algorithm_map,
            action_factory_map = lhd_action_factory_map
        )

        #Results that are saved go to run_dir
        self.results_folder = run_dir 
        #Create if not created by driver
        if self.config.sim.save_data_files:
                os.makedirs(self.results_folder, exist_ok = True)

   
    def _assign_graphdata_to_model(self, graphdata: GraphData):
    # graphdata assumed to be a full GraphData object returned by build_graph_data
        self.N = graphdata.N
        self.edge_list = graphdata.edge_list
        self.adj_matrix = graphdata.adj_matrix
        self.individual_lookup = graphdata.individual_lookup
        self.ages = graphdata.ages
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


    def _initialize_states(self):
        """
        Set up new SEIR arrays and seed initial infectious individuals
        """
        self.current_time = 0
       #Set vaccinations
        self.is_vaccinated = self.rng.random(self.N) < self.config.epi.vax_uptake

        #State tracking variables; S = 0, E = 1, I = 2, R = 3
        self.state = np.zeros(self.N, dtype = np.int8) 
        self.time_in_state = np.zeros(self.N, dtype = np.float32)
        self.incubation_periods = np.full(self.N, np.nan, dtype = np.float32)
        self.infectious_periods = np.full(self.N, np.nan, dtype = np.float32)

        #Per-run trajectories
        self.states_over_time = [] 
        self.new_exposures = []
        self.new_infections = []

        #pick random I0 if none provided
        initial_infectious = self.config.sim.I0
        if not initial_infectious:
            initial_infectious = [self.rng.integers(0,self.N)]
            self.I0 = initial_infectious
        self.state[initial_infectious] = 2
        self.infectious_periods[initial_infectious] = self.assign_infectious_period(initial_infectious)

        S = list(set(range(self.N)) - set(initial_infectious))
        E = []
        I = initial_infectious  # noqa: E741
        R =  []

        self.states_over_time.append([S,E,I,R])
        self.new_exposures.append(np.empty(0, dtype = np.int32))
        self.new_infections.append(initial_infectious)

        #Reset run flags
        self.simulation_end_day = self.Tmax
        self.stochastic_dieout = False

        #Reset LHD intervention Multipliers
        self.in_multiplier = {ct: np.ones(self.N, dtype = np.float32) for ct in self.contact_types}
        self.out_multiplier = {ct: np.ones(self.N, dtype = np.float32) for ct in self.contact_types}

        if hasattr(self, "lhd"):
            self.lhd.reset_for_run()

        return
       

        
    def assign_incubation_period(self, inds):
        """
        Take a list of newly-assigned exposed indices and assign an incubation period
        """
        inds = np.atleast_1d(inds)
        mean_inc = np.where(self.is_vaccinated[inds], self.config.epi.incubation_period_vax, self.config.epi.incubation_period)
        return self.rng.gamma(shape = self.config.epi.gamma_alpha, scale = mean_inc)
    
    def assign_infectious_period(self, inds):
        """
        Take a list of newly-assigned infectious indices and assign an infectious period
        """
        inds = np.atleast_1d(inds)
        mean_inf = np.where(self.is_vaccinated[inds], self.config.epi.infectious_period_vax, self.config.epi.infectious_period)


        return self.rng.gamma(shape = self.config.epi.gamma_alpha, scale = mean_inf)
    
    def _prepare_numba_structures(self):
        """
        Convert self.csr_by_type and multiplier dicts into Numba-friendly structures
        (typed lists and contiguous arrays). Called once (or at first use).
        """
        contact_types = list(self.contact_types)
        self._numba_contact_types = contact_types
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
                    




    def determine_new_exposures(self, recorder: ExposureEventRecorder = None):
        """
        Wrapper that prepares inputs, calls the numba compiled function to compute newly exposed,and optionally reconstructs per-event metadata for the recorder (if provided).
        """
        infectious_indices = np.where(self.state == 2)[0]
        if infectious_indices.size == 0:
            return np.array([], dtype=np.int32)

        base_prob = float(self.config.epi.base_transmission_prob)
        vax_efficacy = float(self.config.epi.vax_efficacy)
        susc_under5 = float(self.config.epi.susceptibility_multiplier_under_five)
        susc_elderly = float(self.config.epi.susceptibility_multiplier_elderly)
        rel_inf_vax = float(self.config.epi.relative_infectiousness_vax)

        if not getattr(self, "_numba_ready", False):
            self._prepare_numba_structures()

        # update multipliers in-place from dictionaries 
        self._update_multiplier_matrices()

        newly_bool = _determine_new_exposures_numba(
            self.state.astype(np.int8),
            infectious_indices.astype(np.int64),
            self._numba_indptr_list,
            self._numba_indices_list,
            self._numba_weights_list,
            self._numba_out_mat,
            self._numba_in_mat,
            self.is_vaccinated,
            self.ages.astype(np.int32),
            base_prob,
            vax_efficacy,
            susc_under5,
            susc_elderly,
            rel_inf_vax
        )

        
        newly_exposed = np.where(newly_bool)[0].astype(np.int32)


        if recorder is not None:
            contact_types = self._numba_contact_types
            for src in infectious_indices:
                for j, ct in enumerate(contact_types):
                    indptr = self._numba_indptr_list[j]
                    indices = self._numba_indices_list[j]
                    start = indptr[src]
                    end = indptr[src + 1]
                    if end <= start:
                        continue
                    neighbors = indices[start:end]
                    sus_mask = (self.state[neighbors] == 0)
                    if not sus_mask.any():
                        continue
                    sus_neighbors = neighbors[sus_mask]
                    infected_mask = newly_bool[sus_neighbors]
                    if infected_mask.any():
                        type_id = self.ct_to_id[ct]
                        recorder.append_event(self.current_time, int(src), int(type_id), sus_neighbors, infected_mask)
        return newly_exposed

    #@profile
    def step(self, recorder: ExposureEventRecorder = None):
        """
        Takes data from a previous step's self.state, and updates, expanding the graph as appropriate
        """

        #S -> E
        self.exposure_events = [] #record exposure events
        newly_exposed = self.determine_new_exposures(recorder = recorder)

        if newly_exposed.size > 0:
            self.state[newly_exposed] = 1
            self.time_in_state[newly_exposed] = -1 #start at -1 since 1 is immediately added in E-> I 
            self.incubation_periods[newly_exposed] = self.assign_incubation_period(newly_exposed)

        #E -> I
        exposed = np.where(self.state == 1)[0]
        self.time_in_state[exposed] += 1
        to_infectious = exposed[self.time_in_state[exposed] >= self.incubation_periods[exposed]]

        if len(to_infectious) > 0:
            self.state[to_infectious] = 2
            self.time_in_state[to_infectious] = -1 #1 added in next step
            self.infectious_periods[to_infectious] = self.assign_infectious_period(to_infectious)

        #I -> R
        infectious = np.where(self.state == 2)[0]
        self.time_in_state[infectious] += 1
        to_recovered = infectious[self.time_in_state[infectious] >= self.infectious_periods[infectious]]

        if len(to_recovered) > 0:
            self.state[to_recovered] = 3
            self.time_in_state[to_recovered] = -1

        #Advance Recovered
        recovered = np.where(self.state == 3)[0]
        self.time_in_state[recovered] += 1

         #R -> S with waning immunity
        cid = self.config.epi.conferred_immunity_duration
        if cid is not None:
            try:
                cid_val = float(cid)
            except Exception:
                cid_val = None

            if cid_val is not None and cid_val >= 0:
                #Revert those with greater time_in_state to sus
                waned_mask = (self.state == 3) & (self.time_in_state >= cid_val)
                if waned_mask.any():
                    waned_nodes = np.where(waned_mask)[0]
                    #move to sus and reset everything
                    self.state[waned_nodes] = 0
                    self.time_in_state[waned_nodes] = 0
                    self.incubation_periods[waned_nodes] = np.nan
                    self.infectious_periods[waned_nodes] = np.nan
                    #check if there is lasting immunity, and reduce in multiplier
                    lasting = self.config.epi.lasting_partial_immunity
                    if lasting:
                        lasting = np.clip(lasting, 0, 1)
                        reduced_sus = (1-lasting)
                        for ct in self.contact_types:
                            self.in_multiplier[ct][waned_nodes] *= reduced_sus

                            





        #Save timestep data
        S = list(np.where(self.state == 0)[0])
        E = list(np.where(self.state == 1)[0])
        I = list(np.where(self.state == 2)[0])  # noqa: E741
        R = list(np.where(self.state == 3)[0])
        self.states_over_time.append([S,E,I,R])
        self.new_exposures.append(newly_exposed)
        self.new_infections.append(to_infectious)


    #@profile
    def simulate(self):

        for run in range(self.n_replicates):
            self._initialize_states()
            #store vaccination status for run
            self.all_vax_status[run] = self.is_vaccinated.copy()

            t = 0
            while t < self.Tmax:
                t += 1 #day 0 is recorded in initialization
                self.current_time = t

                if self.config.sim.record_exposure_events:
                    recorder = self.recorder_template
                    recorder.reset()
                else:
                    recorder = None

                #advance SEIR and record
                self.step(recorder)

                #either log an exposure event or make a false empty one
                if recorder is not None:
                    snapshot = recorder.snapshot_compact(copy = True) 
                    self.exposure_event_log[run].append(snapshot)
                else:
                    snapshot = {
                        'event_time': np.empty(0, dtype = np.int32),
                        'event_source': np.empty(0, dtype = np.int32),
                        'event_type': np.empty(0, dtype = np.int16),
                        'event_nodes_start': np.empty(0, dtype = np.int64),
                        'event_nodes_len': np.empty(0, dtype = np.int32),
                        'nodes': np.empty(0, dtype = np.int32),
                        'infections': np.empty(0, dtype = bool)
                    }

                #Run LHD step once
                self.lhd.step(self.current_time, snapshot)

                S, E, I, R = self.states_over_time[-1]  # noqa: E741
                if not E and not I:
                    self.simulation_end_day = t 
                    self.stochastic_dieout = True
                    break
                
            #save per run data
            self.all_states_over_time[run] = [states.copy() for states in self.states_over_time]
            self.all_new_exposures[run] = [ne.copy() if hasattr(ne, "copy") else list(ne) for ne in self.new_exposures]
            self.all_stochastic_dieout[run] = self.stochastic_dieout
            self.all_end_days[run] = self.simulation_end_day
            self.all_lhd_action_logs[run] = list(self.lhd.action_log)

    def write_run_results(self, out_dir: Optional[str] = None, prefix: str = "run_results"):
        #TODO NOT WORKING 

        """
    Write three parquet files under out_dir (default self.results_folder):
      - <prefix>_prevalence.parquet : per-run rows, columns t_0..t_{T-1} containing prevalence fraction
      - <prefix>_incidence.parquet  : per-run rows, columns t_0..t_{T-1} containing incidence counts
      - <prefix>_summary.parquet    : per-run rows, columns for scalar summary metrics (peak, time_of_peak, calls, SAR by ct, etc.)
      """

        def _flatten_exposures_safe(exposures_list):
            """
            Helper function to concatenate non-empty arrays 
            """
            if not exposures_list:
                return np.empty(0, dtype=np.int32)

            parts = []
            for arr in exposures_list:
                if arr is None:
                    continue
                try:
                    a = np.asarray(arr, dtype=np.int32)
                except Exception:
                    # fallback: try to coerce by converting to list first
                    try:
                        a = np.asarray(list(arr), dtype=np.int32)
                    except Exception:
                        continue
                if a.size > 0:
                    parts.append(a)
            if not parts:
                return np.empty(0, dtype=np.int32)
            if len(parts) == 1:
                return parts[0]
            return np.concatenate(parts)



        out_dir = out_dir or self.results_folder
        os.makedirs(out_dir, exist_ok=True)

        
        T = 0
        for sts in self.all_states_over_time:
            if sts is not None:
                if len(sts) > T:
                    T = len(sts)
        # Ensure we at least create columns t_0 .. t_{T-1}
        t_cols = [f"t_{i}" for i in range(T)]

        # Build prevalence and incidence wide DataFrames: rows per run
        prevalence_rows = []
        incidence_rows = []

        for run in range(self.n_replicates):
            # states_over_time: list of [S,E,I,R] per time
            states_list = self.all_states_over_time[run] or []
            exposures_list = self.all_new_exposures[run] or []
            flat = _flatten_exposures_safe(exposures_list)

            total_infection_events = int(flat.size)
            if flat.size > 0:
                reinfection_counts = np.bincount(flat, minlength = self.N)
                total_infections_unique = int((reinfection_counts > 0).sum())

            else:
                reinfection_counts = np.zeros(self.N, dtype = int)
                total_infections_unique = 0






            # prevalence: fraction I/N; incidence: counts of newly exposed per time
            prevalences = []
            incidences = []
            for t in range(T):
                # prevalence
                if t < len(states_list) and states_list[t] is not None and len(states_list[t]) > 2:
                    I_nodes = states_list[t][2]
                    nI = int(len(I_nodes))
                else:
                    nI = 0
                prevalences.append(float(nI) / float(self.N) if self.N > 0 else 0.0)

                # incidence
                if t < len(exposures_list):
                    arr = exposures_list[t]
                    try:
                        cnt = int(arr.size) if hasattr(arr, "size") else int(len(arr))
                    except Exception:
                        cnt = int(len(arr)) if hasattr(arr, "__len__") else 0
                    # special-case day-0: if empty but I0 available, use I0
                    if t == 0 and cnt == 0:
                        I0_val = getattr(self, "I0", None)
                        if I0_val is None:
                            I0_val = (self.config.sim.I0 or [])
                        try:
                            cnt = int(len(I0_val)) if I0_val is not None else 0
                        except Exception:
                            cnt = 0
                else:
                    cnt = 0
                incidences.append(int(cnt))

            prevalence_rows.append({"run_number": int(run), **{f"t_{i}": prevalences[i] for i in range(T)}})
            incidence_rows.append({"run_number": int(run), **{f"t_{i}": incidences[i] for i in range(T)}})

        prevalence_df = pd.DataFrame(prevalence_rows, columns=["run_number"] + t_cols)
        incidence_df = pd.DataFrame(incidence_rows, columns=["run_number"] + t_cols)

        # Build summary scalars per run (peak prevalence/time, calls, SAR by contact type, etc.)
        summary_rows = []
        contact_types = list(self.contact_types)

        for run in range(self.n_replicates):
            # compute peak prevalence & time
            states_list = self.all_states_over_time[run] or []
            peak_infections = 0
            time_of_peak = None
            for t, state in enumerate(states_list):
                if state is None or len(state) <= 2:
                    continue
                nI = int(len(state[2]))
                if nI > peak_infections:
                    peak_infections = nI
                    time_of_peak = int(t)
            peak_prev_frac = (float(peak_infections) / float(self.N)) if self.N > 0 else 0.0

            # Calls metrics from LHD action logs recorded per run
            calls_log = (self.all_lhd_action_logs[run] or []) if hasattr(self, "all_lhd_action_logs") else []
            call_entries = [e for e in calls_log if e.get("action_type") == "call"]
            number_of_call_actions = len(call_entries)
            people_called = sum(int(e.get("nodes_count", 0)) for e in call_entries)
            if people_called > 0:
                total_person_days = sum(int(e.get("duration", 0)) * int(e.get("nodes_count", 0)) for e in call_entries)
                avg_days_quarantine = float(total_person_days) / float(people_called)
            else:
                avg_days_quarantine = None

            # SAR: compute event-level secondary attack rates by contact type using exposure_event_log snapshots
            # exposure_event_log[run] is list of snapshots (compact form) per timestep
            snapshots = self.exposure_event_log[run] if run < len(self.exposure_event_log) else []
            total_exposed_by_ct = {ct: 0 for ct in contact_types}
            total_infected_events_by_ct = {ct: 0 for ct in contact_types}

            for snap in (snapshots or []):
                n_events = int(snap.get("event_time", np.empty(0)).shape[0])
                if n_events == 0:
                    continue
                for ei in range(n_events):
                    try:
                        ct_id = int(snap["event_type"][ei]) if snap["event_type"].size > 0 else None
                    except Exception:
                        ct_id = None
                    ct_name = self.id_to_ct.get(ct_id, "unknown") if ct_id is not None else "unknown"
                    s = int(snap["event_nodes_start"][ei])
                    L = int(snap["event_nodes_len"][ei])
                    if L == 0:
                        continue
                    infs = snap["infections"][s : s + L]
                    infected_events_count = int(np.sum(infs))
                    total_exposed_by_ct[ct_name] = total_exposed_by_ct.get(ct_name, 0) + int(L)
                    total_infected_events_by_ct[ct_name] = total_infected_events_by_ct.get(ct_name, 0) + infected_events_count

            sar_by_ct = {}
            for ct in contact_types:
                exposed_ct = total_exposed_by_ct.get(ct, 0)
                infected_ct = total_infected_events_by_ct.get(ct, 0)
                sar_by_ct[f"sar_events_{ct}"] = float(infected_ct / exposed_ct) if exposed_ct > 0 else None
                sar_by_ct[f"exposed_{ct}"] = int(exposed_ct)
                sar_by_ct[f"infected_events_{ct}"] = int(infected_ct)

            # assemble summary row
            row = {
                "run_number": int(run),
                "peak_prevalence": float(peak_prev_frac),
                "time_of_peak": int(time_of_peak) if time_of_peak is not None else None,
                "number_of_call_actions": int(number_of_call_actions),
                "people_called": int(people_called),
                "avg_days_quarantine_imposed": float(avg_days_quarantine) if avg_days_quarantine is not None else None,
                "stochastic_dieout": bool(self.all_stochastic_dieout[run]),
                "epidemic_duration": int(self.all_end_days[run]) if self.all_end_days is not None else int(self.Tmax),
                "total_infection_events": int(total_infection_events),
                "total_infections_unique": int(total_infections_unique)
            }
            row.update(sar_by_ct)
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        # atomic writes to disk (tmp + replace)
        def _atomic_write(df: pd.DataFrame, path: str):
            tmp = path + ".tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)

        preval_path = os.path.join(out_dir, f"{prefix}_prevalence.parquet")
        inc_path = os.path.join(out_dir, f"{prefix}_incidence.parquet")
        summ_path = os.path.join(out_dir, f"{prefix}_summary.parquet")

        _atomic_write(prevalence_df, preval_path)
        _atomic_write(incidence_df, inc_path)
        _atomic_write(summary_df, summ_path)

        return preval_path, inc_path, summ_path
   

    def cumulative_incidence_plot(self, run_number = 0, strata:str = None, time: int = None, suffix: str = None) -> None:
        """
        Plots cumulative incidence over time, optionally stratified
        Currently supports stratification by age, sex, vaccination status

        Args:
            time (int): time range for plot, 0-t, defaults to full timespan
            run_number (int): If model run multiple times, specify which to viz
            strata (str )  stratification factor for plot

        Creates a matplotlib plot
        """

        pop_size = self.N
        exposures = self.all_new_exposures[run_number]
        if len(exposures) > 0 and len(exposures[0]) == 0:
            exposures[0] = list(self.config.sim.I0)

        if time is None: 
            max_time = len(exposures) - 1
        else:
            #don't let time input overshoot
            max_time = min(time - 1, len(exposures) - 1) 
        x_vals = range(max_time + 1)

        if isinstance(strata, str):
            strata = strata.lower()



        #Track cumulatively infections
        ever_exposed = np.zeros(pop_size, dtype = bool)
        overall_cumulative = []
        for t in x_vals:
            ever_exposed[np.array(exposures[t], dtype = int)] = True
            overall_cumulative.append(ever_exposed.sum() / pop_size)
        
        #Stratification Logic
        strata_labels, strata_members, strata_colors, strata_cumulative = None, None, None, None

        if strata in ("age", "sex", "vaccination"):
            

            if strata == "age":
                strata_attr = self.individual_lookup[strata]
                #Age bins directly input here, change as needed
                bins = [0, 6, 19, 35, 65, 200]
                labels = ["0-5", "6-18", "19-34", "35-64", "65+"]
                strata_labels = labels
                strata_vals = pd.cut(strata_attr, bins, right = False, labels = labels)
                strata_colors = [
        "#e41a1c",  # Red for 0-5
        "#377eb8",  # Blue for 6-18
        "#4daf4a",  # Green for 19-34
        "#984ea3",  # Purple for 35-64
        "#ff7f00",  # Orange for 65+
    ]

            elif strata == "sex":
                strata_attr = self.individual_lookup[strata]
                strata_vals = strata_attr
                labels = sorted(strata_attr.unique())
                strata_labels = labels
                label_to_color = {lab: ("red" if lab == "F" else "blue") for lab in labels}
                strata_colors = [label_to_color[lab] for lab in labels]

            elif strata == "vaccination":
                if not hasattr(self, "is_vaccinated"):
                    raise ValueError("vaccination stratification requested but vaccination status not initialized")

                strata_attr = self.is_vaccinated
                strata_labels = ["Vaccinated", "Unvaccinated"]
                strata_vals = np.where(strata_attr, "Vaccinated", "Unvaccinated")
                strata_colors = ["#377eb8", "#e41a1c"] #Blue vax, red unvax



        elif strata and (strata not in self.individual_lookup.columns):
            raise ValueError(f"stratifying factor {strata} is not an attribute of this graph. Check spelling and try again")
        
        else:
             strata = None

        if strata_labels is not None:
            strata_members = {label: np.where(strata_vals == label)[0] for label in strata_labels}
            strata_cumulative = {label: [] for label in strata_labels}
            ever_exposed_stratum = {label: np.zeros(len(strata_members[label]), dtype = bool) for label in strata_labels}

            #update ever_exposed_stratums for each label
            for t in x_vals:
                newly_exposed = np.array(exposures[t], dtype = int)
                for label in strata_labels:
                    members = strata_members[label]
                    #check which belong to this stratum
                    mask = np.isin(members, newly_exposed)
                    ever_exposed_stratum[label][mask] = True
                    group_size = len(members)
                    if group_size > 0:
                        strata_cumulative[label].append(ever_exposed_stratum[label].sum() / group_size)
                    else:
                        strata_cumulative[label].append(0.0)



        
        plt.figure(figsize = (8, 5))
        plt.plot(x_vals, overall_cumulative, color = "black", label = "Total", linewidth = 2, zorder = 3)

        if strata_cumulative:
            bottom = np.zeros(len(x_vals))
            for i, label in enumerate(strata_labels):
                pop_fraction = len(strata_members[label]) / pop_size
                plt.fill_between(x_vals, bottom,
                np.array(bottom) + np.array(strata_cumulative[label])*pop_fraction,
                step = None, color = strata_colors[i], alpha = 0.5, label = str(label))
                bottom += np.array(strata_cumulative[label]) * pop_fraction


        plt.xlabel("Time step (day)")
        plt.ylabel("Cumulative Incidence (fraction of population)")
        if strata:
            plt.title(f"Cumulative Incidence (Stratified by {strata})\nRun {self.config.sim.run_name}")
            legend_handles = [Patch(color = strata_colors[i], label = str(label)) for i, label in enumerate(strata_labels)]
            plt.legend(handles = legend_handles, loc = "upper left")
        else:
            plt.title(f"Cumulative Incidence Over Time for  {self.config.sim.run_name}")
        plt.grid(True, axis = "y", alpha = 0.5)
        plt.tight_layout()
        plotpath = os.path.join(self.results_folder, "cumulative_incidence")
        if suffix:
            plotpath = plotpath + suffix
        if self.config.sim.save_plots:
            if strata:
                plt.savefig(f"{plotpath}_by{strata}.png")
            else: 
                plt.savefig(f"{plotpath}.png")
        if self.config.sim.display_plots:
            plt.show()
        plt.close()

    def cumulative_incidence_spaghetti(self, suffix: str = None):
        """
        Plots cumulative incidence trajectory for every run. Runs with stochastic-dieout are red, others blue.

        Args:
            suffix (str): An optional suffix for the figure filepath
        Returns:
            Nothing
        """

        pop_size = self.N
        n_replicates = self.n_replicates

        #TODO make alpha inversely proportional to n_replicates
        alpha = 0.5

        max_timesteps = max(len(run_exposures) for run_exposures in self.all_new_exposures)
        plt.figure(figsize = (10, 6))

        #Calculate individual curves
        all_curves = []
        for run in range(n_replicates):
            exposures = self.all_new_exposures[run]
            ever_exposed = np.zeros(pop_size, dtype = bool)
            cumulative = []
            for t in range(len(exposures)):
                newly_exposed = np.array(exposures[t], dtype = int)
                ever_exposed[newly_exposed] = True
                cumulative.append(ever_exposed.sum() / pop_size)
            if len(cumulative) < max_timesteps:
                cumulative += [cumulative[-1]] * (max_timesteps - len(cumulative))
            all_curves.append(cumulative)

        #plot spaghetti curves
        for run, curve in enumerate(all_curves):
            color = "red" if self.all_stochastic_dieout[run] else "#3333ff"
            plt.plot(range(max_timesteps), curve, 
            color = color, alpha = alpha, lw = 1)

        #plot mean curve
        mean_curve = np.mean(all_curves, axis = 0)
        plt.plot(range(max_timesteps), mean_curve, 
        color = "black", lw = 2, label = "Mean Trajectory")

        plt.xlabel("Time step (day)")
        plt.ylabel("Cumulative Incidence (fraction of population)")
        plt.title(f"Cumulative Incidence Spaghetti Plot\n{self.n_replicates} runs, red = die-out")
        plt.grid(True, axis = "y", alpha = 0.5)
        plt.legend()
        plt.tight_layout()
        plotpath = os.path.join(self.results_folder, "cumulative_incidence_spaghetti")
        if suffix:
            plotpath = plotpath + suffix
        if self.config.sim.save_plots:
            plt.savefig(f"{plotpath}.png")
        if self.config.sim.display_plots:
            plt.show()
        plt.close()
        
    def draw_network(self, t: int, run_number = 0, ax=None, clear: bool =True, saveFile: bool = False, suffix: str = None):
        """
        Draws a network containing outbreak-involved nodes (E,I,R) + neighbors at time t
        """

        #get indices of E, I, R
        S, E, I, R = self.all_states_over_time[run_number][t] # noqa: E741
        affected_inds = set(E) | set(I) | set(R) 

        #neighbors of affected nodes
        neighbors = set()
        for ind in affected_inds:
            for (nbr, w, ct) in self.neighbor_map.get(int(ind), []):
                neighbors.add(int(nbr))

        #combine affected nodes and neighbors
        plot_nodes = sorted(affected_inds | neighbors)

        #build subgraph
        name_to_ind = {name: idx for idx, name in enumerate(plot_nodes)}
        subg = ig.Graph()
        subg.add_vertices(len(plot_nodes))
        subg.vs["name"] = plot_nodes

        #add edges
        edges = []
        for src in plot_nodes:
            for (tgt, w, ct) in self.neighbor_map.get(int(src), []):
                if int(tgt) in name_to_ind:
                    edges.append((name_to_ind[src],name_to_ind[int(tgt)]))

        if edges:
            subg.add_edges(edges)

        #color by state
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

        #plot
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

        #Save if requested
        if saveFile:
            plotpath = os.path.join(self.results_folder, f"network_at_{str(t)}")
            if suffix:
                plotpath = plotpath + suffix
            plt.savefig(f"{plotpath}.png")
        if show_plot and self.config.sim.display_plots:
            plt.show()
        plt.close()
    

        return

    






