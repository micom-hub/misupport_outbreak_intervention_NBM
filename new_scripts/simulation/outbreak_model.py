

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import igraph as ig
from typing import Dict, Any, Optional, Callable
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from numba import njit
from numba.typed import List as NumbaList


from new_scripts.config import ModelConfig
from new_scripts.recorder.recorder import ExposureEventRecorder
from new_scripts.lhd.lhd import LocalHealthDepartment
from new_scripts.graph.graph_utils import GraphData 


#-----Outbreak Model-------
class NetworkModel:
    #@profile
    def __init__(
        self,  
    config: ModelConfig,
    graphdata: GraphData,
    *,
    rng = None,
    seed: Optional[int] = None,
    results_folder: Optional[str] = None,
    lhd_register_defaults: bool = True,
    lhd_algorithm_map: Optional[Dict[str, object]] = None,
    lhd_action_factory_map: Optional[Dict[str, Callable[..., Any]]] = None):
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
        self.n_runs = self.config.sim.n_runs


        #Set up storage for full run
        self.all_states_over_time = [None]*self.n_runs
        self.all_new_exposures = [None]*self.n_runs
        self.exposure_event_log = [[] for _ in range(self.n_runs)]
        self.all_stochastic_dieout = np.zeros(self.n_runs, dtype = bool)
        self.all_end_days = np.ones(self.n_runs, dtype = int)*self.Tmax

        self.all_vax_status = [None] * self.n_runs
        self.all_lhd_action_logs = [None] * self.n_runs

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

        #set-up result folder if not overriden by driver
        
        self.results_folder = results_folder if results_folder is not None else os.path.join(os.getcwd(), "results", self.config.sim.run_name)
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

    #@profile
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
    Determine who becomes newly exposed (pre-infectious) this timestep.
    If recorder is provided, it will track exposure events for each (src, ct)
    storing: exposure metadata

    Returns: np.ndarray of newly exposed node indices
        """


        N = self.N
        newly_exposed = np.zeros(N, dtype = bool)
        infectious_indices = np.where(self.state == 2)[0]

        if len(infectious_indices) == 0:
            return np.array([], dtype = int)

        #gather factors affecting transmission probability
        base_prob = self.config.epi.base_transmission_prob
        vax_efficacy = self.config.epi.vax_efficacy
        susc_mult_under5 = self.config.epi.susceptibility_multiplier_under_five
        susc_mult_elderly = self.config.epi.susceptibility_multiplier_elderly
        rel_inf_vax = self.config.epi.relative_infectiousness_vax

        #local aliases
        csr_by_type = self.csr_by_type
        is_vax = self.is_vaccinated
        ages = self.ages
        rng = self.rng

        #iterate over exposure events (src,ct) to determine transmission probabilities
        for src in infectious_indices:
            for ct, (indptr, indices, weights) in csr_by_type.items():
                start = indptr[src]
                end = indptr[src+1]
                if end <= start:
                    continue
                #determine susceptible neighbors
                neighbors = indices[start:end]
                w = weights[start:end]
                sus_mask = (self.state[neighbors] == 0)
                if not sus_mask.any():
                    continue
                sus_neighbors = neighbors[sus_mask]
                sus_w = w[sus_mask].astype(np.float32)

                #gather intervention multipliers 
                out_mult = self.out_multiplier[ct][src]
                in_mult = self.in_multiplier[ct][sus_neighbors]

                #calculate effective weight and transmission prob
                effective_w = sus_w * out_mult * in_mult
                prob = base_prob * effective_w

                #vaccination weighted
                if is_vax[src]:
                    prob = prob * rel_inf_vax
                
                vax_tgt = is_vax[sus_neighbors]
                if vax_efficacy != 0:
                    prob = prob * ((1.0 - vax_efficacy)** vax_tgt)

                #age weighted
                under5 = (ages[sus_neighbors] <= 5)
                elderly = (ages[sus_neighbors] >= 65)
                if susc_mult_under5 != 1.0:
                    prob[under5] = prob[under5] * susc_mult_under5
                if susc_mult_elderly != 1.0:
                    prob[elderly] = prob[elderly]*susc_mult_elderly
                
                prob = np.clip(prob, 0.0, 1.0)


                #Determine who becomes infected
                draws = rng.random(prob.shape)
                infected_mask = (draws < prob)
                if infected_mask.any():
                    infected_nodes = sus_neighbors[infected_mask]
                    newly_exposed[infected_nodes] = True
                
                #Record event data
                if recorder is not None:
                    type_id = self.ct_to_id[ct]
                    #pass array and infected mask 
                    recorder.append_event(self.current_time, int(src), int(type_id), sus_neighbors, infected_mask)

        
        result = np.where(newly_exposed)[0].astype(np.int32)
        return result

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

        for run in range(self.n_runs):
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


    def epi_outcomes(self, reduced: bool = False) -> pd.DataFrame:    
        """    
        Calculates epidemic outcomes with a row for each run.
    If reduced == True, returns a compact DataFrame with one row per run containing:
    - run_number
    - cumulative_incidence_percent : percent of population infected at least once
    - cumulative_infections : total number of infection events (may exceed population)
    - peak_prevalence : peak prevalence (fraction of population simultaneously infectious)
    - time_of_peak_prevalence : timestep of peak prevalence (int) or None
    - epidemic_duration : time when infection died out (or Tmax if not died out)

    If reduced == False, returns a detailed DataFrame that includes both unique-person
    and event-level metrics (SAR by contact type, effective R both by events and by unique
    attribution, age-stratified measures, LHD metrics, etc.).
    """
        warnings.filterwarnings("ignore", category = RuntimeWarning, module = "numpy.*")

        N = self.N

        if any(x is None for x in self.all_new_exposures):
            raise ValueError("Model runs not completed; run simulate() before computing outcomes")

        summary = []

        # Age bins (kept for the detailed output path)
        age_bins = [0, 6, 19, 35, 65, 200]
        age_labels = ["0-5", "6-18", "19-34", "35-64", "65+"]

        # stable contact type list
        contact_types = list(self.contact_types)

        for run in range(self.n_runs):
            exposures_raw = self.all_new_exposures[run]
            states_over_time = self.all_states_over_time[run]
            stochastic_dieout = bool(self.all_stochastic_dieout[run])
            epidemic_time = int(self.all_end_days[run]) if self.all_end_days is not None else int(self.Tmax)
            snapshots = self.exposure_event_log[run] if run < len(self.exposure_event_log) else []

            # Normalize exposures into list of numpy int arrays
            daily = []
            for arr in (exposures_raw or []):
                if isinstance(arr, np.ndarray):
                    a = arr.astype(np.int32) if arr.size > 0 else np.empty(0, dtype=np.int32)
                else:
                    try:
                        a = np.array(arr, dtype=np.int32) if len(arr) > 0 else np.empty(0, dtype=np.int32)
                    except Exception:
                        a = np.empty(0, dtype=np.int32)
                daily.append(a)

            # ensure day-0 includes I0 if empty
            if len(daily) > 0 and daily[0].size == 0:
                daily[0] = np.array(self.config.sim.I0, dtype = np.int32)

            # Flatten event-level exposures for run-level event counts and reinfection counts
            nonempty = [d for d in daily if d.size > 0]
            flat = np.concatenate(nonempty).astype(np.int32) if nonempty else np.empty(0, dtype=np.int32)

            # Unique persons infected (ever) and event counts per person
            if flat.size > 0:
                reinfection_counts = np.bincount(flat, minlength=N)
                total_infection_events = int(flat.size)
                total_infections_unique = int((reinfection_counts > 0).sum())
            else:
                reinfection_counts = np.zeros(N, dtype=int)
                total_infection_events = 0
                total_infections_unique = 0

            # Reduced summary path
            if reduced:
                # cumulative incidence percent (unique persons)
                cumulative_incidence_percent = (float(total_infections_unique) / float(N) * 100.0) if N > 0 else np.nan

                # peak prevalence (fraction) and time of peak
                peak_infections = 0
                time_of_peak = None
                if states_over_time:
                    for t, state in enumerate(states_over_time):
                        I_nodes = np.array(state[2], dtype=int) if len(state) > 2 else np.empty(0, dtype=int)
                        nI = int(I_nodes.size)
                        if nI > peak_infections:
                            peak_infections = nI
                            time_of_peak = int(t)
                peak_prevalence = (peak_infections / float(N)) if N > 0 else np.nan

                # Compute event-level effective R by contact type
                total_infected_events_by_ct = {ct: 0 for ct in contact_types}
                unique_sources_per_ct = {ct: set() for ct in contact_types}

                for snap in (snapshots or []):
                    n_events = int(snap["event_time"].shape[0])
                    if n_events == 0:
                        continue
                    for ei in range(n_events):
                        src = int(snap["event_source"][ei])
                        # safe extraction of contact-type id/name
                        try:
                            ct_id = int(snap["event_type"][ei])
                            ct_name = self.id_to_ct.get(ct_id, None)
                        except Exception:
                            ct_name = None

                        s = int(snap["event_nodes_start"][ei])
                        L = int(snap["event_nodes_len"][ei])
                        if L == 0:
                            continue
                        infs = snap["infections"][s : s + L]
                        infected_events_count = int(np.sum(infs))
                        ct_name = ct_name if ct_name is not None else "unknown"

                        # ensure ct appears in our dictionaries
                        if ct_name not in total_infected_events_by_ct:
                            total_infected_events_by_ct[ct_name] = 0
                            unique_sources_per_ct[ct_name] = set()

                        total_infected_events_by_ct[ct_name] = total_infected_events_by_ct.get(ct_name, 0) + infected_events_count
                        unique_sources_per_ct[ct_name].add(src)

                # compute per-contact-type mean secondary infections (event-level)
                eff_R_events_by_ct = {}
                for ct in contact_types:
                    n_src = len(unique_sources_per_ct.get(ct, set()))
                    tot_inf = total_infected_events_by_ct.get(ct, 0)
                    eff_R_events_by_ct[ct] = (tot_inf / float(n_src)) if n_src > 0 else np.nan

                # assemble reduced row and include eff_R by contact type
                row = {
                    "run_number": int(run),
                    "cumulative_incidence_percent": float(cumulative_incidence_percent) if not np.isnan(cumulative_incidence_percent) else np.nan,
                    "cumulative_infections": int(total_infection_events),
                    "peak_prevalence": float(peak_prevalence) if not np.isnan(peak_prevalence) else np.nan,
                    "time_of_peak_prevalence": time_of_peak,
                    "epidemic_duration": int(epidemic_time),
                }

                # append one column per known contact type, named eff_R_{ct}
                for ct in contact_types:
                    val = eff_R_events_by_ct.get(ct, np.nan)
                    row[f"eff_R_{ct}"] = float(val) if not np.isnan(val) else np.nan

                summary.append(row)
                continue  # next run

            # Detailed summary path (unchanged from previous full version)
            # Build at-risk mask, anyone who ever had infectious neighbor
            at_risk = np.zeros(N, dtype = bool)
            neighbor_map = self.neighbor_map
            if states_over_time:
                for state in states_over_time:
                    infectious_nodes = state[2]
                    if not infectious_nodes:
                        continue
                    for src in infectious_nodes:
                        for nbr_tuple in neighbor_map.get(int(src), []):
                            nbr = int(nbr_tuple[0])
                            at_risk[nbr] = True
            n_at_risk = int(at_risk.sum())
            infected_at_risk = int(((reinfection_counts > 0) & at_risk).sum())
            attack_rate_at_risk = (infected_at_risk / float(n_at_risk)) if n_at_risk > 0 else np.nan

            # Age-stratified attack rates among individuals ever at-risk
            ages = self.ages
            attack_by_age = {}
            age_members = {}
            for i, label in enumerate(age_labels):
                a_min, a_max = age_bins[i], age_bins[i+1]
                members = np.where((ages >= a_min) & (ages < a_max))[0]
                age_members[label] = members
                if members.size == 0:
                    attack_by_age[label] = np.nan
                else:
                    at_risk_mask = at_risk[members]
                    n_risk_members = int(at_risk_mask.sum())
                    if n_risk_members == 0:
                        attack_by_age[label] = np.nan
                    else:
                        infected_risk_members = int(((reinfection_counts > 0)[members][at_risk_mask]).sum())
                        attack_by_age[label] = infected_risk_members / float(n_risk_members)

            # Vaccination-stratified attack rates among at-risk (vax status at start)
            vax_status = np.asarray(self.all_vax_status[run], dtype = bool) if self.all_vax_status[run] is not None else np.zeros(N, dtype=bool)
            ar_vax_mask = at_risk & vax_status
            ar_unvax_mask = at_risk & (~vax_status)
            n_at_risk_vax = int(ar_vax_mask.sum())
            n_at_risk_unvax = int(ar_unvax_mask.sum())
            attack_rate_vax = ( ((reinfection_counts > 0)[ar_vax_mask].sum()) / n_at_risk_vax ) if n_at_risk_vax > 9 else np.nan
            attack_rate_unvax = ( ((reinfection_counts > 0)[ar_unvax_mask].sum()) / n_at_risk_unvax ) if n_at_risk_unvax > 9 else np.nan

            # Peak incidence (event-level and unique-person-per-day)
            daily_counts_events = [int(d.size) for d in daily] if daily else []
            peak_incidence_events = int(max(daily_counts_events)) if daily_counts_events else 0
            time_of_peak_incidence_events = int(np.argmax(daily_counts_events)) if daily_counts_events else None

            daily_counts_unique = [int(np.unique(d).size) for d in daily] if daily else []
            peak_incidence_unique = int(max(daily_counts_unique)) if daily_counts_unique else 0
            time_of_peak_incidence_unique = int(np.argmax(daily_counts_unique)) if daily_counts_unique else None

            # Peak prevalence + time
            peak_prev = 0.0
            peak_infections = 0
            time_of_peak = None
            for t, state in enumerate(states_over_time):
                I_nodes = np.array(state[2], dtype = int)
                nI = int(I_nodes.size)
                if nI > peak_infections:
                    peak_infections = nI
                    peak_prev = (nI / float(N)) if N > 0 else np.nan
                    time_of_peak = int(t)

            # Cumulative incidence overall and stratified (unique persons)
            cumulative_incidence_by_age = {}
            for label in age_labels:
                members = age_members[label]
                if members.size == 0:
                    cumulative_incidence_by_age[label] = np.nan
                else:
                    cumulative_incidence_by_age[label] = float((reinfection_counts > 0)[members].sum()) / members.size

            n_vax = int(vax_status.sum())
            n_unvax = int((~vax_status).sum())
            cumulative_incidence_vax = (float((reinfection_counts > 0)[vax_status].sum()) / n_vax) if n_vax > 0 else np.nan
            cumulative_incidence_unvax = (float((reinfection_counts > 0)[~vax_status].sum()) / n_unvax) if n_unvax > 0 else np.nan

            # Secondary Attack Rate by contact type and Effective R (both event-level and unique attribution)
            total_exposed_by_ct = {ct: 0 for ct in contact_types}
            total_infected_events_by_ct = {ct: 0 for ct in contact_types}
            total_infected_unique_by_ct = {ct: 0 for ct in contact_types}

            assigned = np.zeros(N, dtype=bool)  # attribute unique infection to first observed source
            source_secondary_events = defaultdict(int)
            source_secondary_unique = defaultdict(int)
            unique_sources = set()

            for snap in (snapshots or []):
                n_events = int(snap["event_time"].shape[0])
                if n_events == 0:
                    continue
                for ei in range(n_events):
                    src = int(snap["event_source"][ei])
                    unique_sources.add(src)
                    ct_id = int(snap["event_type"][ei]) if snap["event_type"].size > 0 else None
                    ct_name = self.id_to_ct.get(ct_id, None) if ct_id is not None else None
                    s = int(snap["event_nodes_start"][ei])
                    L = int(snap["event_nodes_len"][ei])
                    if L == 0:
                        continue
                    nodes = snap["nodes"][s : s + L].astype(int)
                    infs = snap["infections"][s : s + L].astype(bool)
                    ct_name = ct_name if ct_name is not None else "unknown"

                    total_exposed_by_ct[ct_name] = total_exposed_by_ct.get(ct_name, 0) + int(L)
                    infected_events_count = int(np.sum(infs))
                    total_infected_events_by_ct[ct_name] = total_infected_events_by_ct.get(ct_name, 0) + infected_events_count
                    source_secondary_events[src] += infected_events_count

                    # unique attribution: first time a node is seen infected in the run
                    for j in range(L):
                        node = int(nodes[j])
                        if bool(infs[j]) and not assigned[node]:
                            assigned[node] = True
                            total_infected_unique_by_ct[ct_name] = total_infected_unique_by_ct.get(ct_name, 0) + 1
                            source_secondary_unique[src] += 1

            # compute SAR per ct (both event-level and unique-person-level)
            sar_events_by_ct = {}
            sar_unique_by_ct = {}
            for ct in contact_types:
                exposed_ct = total_exposed_by_ct.get(ct, 0)
                infected_events_ct = total_infected_events_by_ct.get(ct, 0)
                infected_unique_ct = total_infected_unique_by_ct.get(ct, 0)
                sar_events_by_ct[ct] = (infected_events_ct / float(exposed_ct)) if exposed_ct > 0 else np.nan
                sar_unique_by_ct[ct] = (infected_unique_ct / float(exposed_ct)) if exposed_ct > 0 else np.nan

            total_secondary_events = sum(source_secondary_events.values())
            total_secondary_unique = sum(source_secondary_unique.values())
            n_sources = len(unique_sources)

            eff_R_events = (total_secondary_events / float(n_sources)) if n_sources > 0 else np.nan
            eff_R_events_std = np.std(list(source_secondary_events.values())) if n_sources > 0 else np.nan
            eff_R_events_median = np.median(list(source_secondary_events.values())) if n_sources > 0 else np.nan

            eff_R_unique = (total_secondary_unique / float(n_sources)) if n_sources > 0 else np.nan
            eff_R_unique_std = np.std(list(source_secondary_unique.values())) if n_sources > 0 else np.nan
            eff_R_unique_median = np.median(list(source_secondary_unique.values())) if n_sources > 0 else np.nan

            # LHD Metrics - calls and quarantine
            lhd_log = self.all_lhd_action_logs[run] or []
            call_entries = [e for e in lhd_log if e.get("action_type") == "call"]
            number_of_call_actions = len(call_entries)
            people_called = sum(int(e.get("nodes_count", 0)) for e in call_entries)
            if people_called > 0:
                total_person_days = sum(int(e.get("duration", 0)) * int(e.get("nodes_count", 0)) for e in call_entries)
                avg_days_quarantine = total_person_days / float(people_called)
            else:
                avg_days_quarantine = np.nan

            # Assemble row dictionary for detailed output
            row = {
                "run_number": int(run),
                "total_infections_unique": int(total_infections_unique),
                "total_infection_events": int(total_infection_events),
                "n_reinfected_people": int((reinfection_counts > 1).sum()),
                "mean_reinfections_per_infected": float(reinfection_counts[reinfection_counts > 0].mean()) if (reinfection_counts > 0).any() else np.nan,
                "n_at_risk": int(n_at_risk),
                "infected_at_risk_unique": int(infected_at_risk),
                "attack_rate_at_risk": float(attack_rate_at_risk) if not np.isnan(attack_rate_at_risk) else np.nan,
                "peak_incidence_events": int(peak_incidence_events),
                "time_of_peak_incidence_events": time_of_peak_incidence_events,
                "peak_incidence_unique": int(peak_incidence_unique),
                "time_of_peak_incidence_unique": time_of_peak_incidence_unique,
                "peak_infections": int(peak_infections),
                "time_of_peak_infections": time_of_peak,
                "peak_prevalence_overall": float(peak_prev),
                "attack_rate_vaccinated_at_risk": float(attack_rate_vax) if not np.isnan(attack_rate_vax) else np.nan,
                "attack_rate_unvaccinated_at_risk": float(attack_rate_unvax) if not np.isnan(attack_rate_unvax) else np.nan,
                # event-level R
                "effective_reproduction_number_events": float(eff_R_events) if not np.isnan(eff_R_events) else np.nan,
                "effective_reproduction_number_events_std": float(eff_R_events_std) if not np.isnan(eff_R_events_std) else np.nan,
                "effective_reproduction_number_events_median": float(eff_R_events_median) if not np.isnan(eff_R_events_median) else np.nan,
                # unique-person attribution R
                "effective_reproduction_number_unique": float(eff_R_unique) if not np.isnan(eff_R_unique) else np.nan,
                "effective_reproduction_number_unique_std": float(eff_R_unique_std) if not np.isnan(eff_R_unique_std) else np.nan,
                "effective_reproduction_number_unique_median": float(eff_R_unique_median) if not np.isnan(eff_R_unique_median) else np.nan,
                "number_of_call_actions": int(number_of_call_actions),
                "people_called": int(people_called),
                "avg_days_quarantine_imposed": float(avg_days_quarantine) if not np.isnan(avg_days_quarantine) else np.nan,
                "stochastic_dieout": stochastic_dieout,
                "epidemic_duration": int(epidemic_time),
            }

            # attach age-specific attack rates among at-risk and cumulative incidence by age
            for label in age_labels:
                row[f"attack_rate_age_{label}_at_risk"] = attack_by_age[label]
                row[f"cumulative_incidence_age_{label}"] = cumulative_incidence_by_age[label]

            # vaccination cumulative incidence (unique-person)
            row["cumulative_incidence_vaccinated"] = cumulative_incidence_vax
            row["cumulative_incidence_unvaccinated"] = cumulative_incidence_unvax

            # attach SAR per contact type and raw exposed/infected counts (both event and unique)
            for ct in contact_types:
                row[f"sar_events_{ct}"] = sar_events_by_ct.get(ct, np.nan)
                row[f"sar_unique_{ct}"] = sar_unique_by_ct.get(ct, np.nan)
                row[f"exposed_{ct}"] = int(total_exposed_by_ct.get(ct, 0))
                row[f"infected_events_{ct}"] = int(total_infected_events_by_ct.get(ct, 0))
                row[f"infected_unique_{ct}"] = int(total_infected_unique_by_ct.get(ct, 0))

            summary.append(row)

        warnings.filterwarnings("default", category = RuntimeWarning, module = "numpy.*")
        return pd.DataFrame(summary)

        
    def epi_curve(self, run_number = 0, suffix: str = None):
        """
        Build an epi curve for a given run of the model

        Args:
            run_number (int): If model run multiple times, specify which curve to build
            suffix (str): An optional suffix to add to the plot name
        """

        exposures = self.all_new_exposures[run_number]
        counts = [len(exposed) for exposed in exposures]
        counts[0] = len(self.config.sim.I0) #add initial infections
        plt.figure(figsize=((8,4)))
        plt.bar(range(len(counts)), counts, color = 'orange', label = "Infections Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Number of Infectious Contacts Made")
        plt.grid(axis = 'y', alpha = 0.5)
        plt.tight_layout()
        if self.config.sim.save_plots:
            plotpath = os.path.join(self.results_folder,"epi_curve")
            if suffix:
                plotpath = plotpath + suffix
            

            plt.savefig(f"{plotpath}.png")

        if self.config.sim.display_plots:
            plt.show()
        plt.close()

        return

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
        n_runs = self.n_runs

        #TODO make alpha inversely proportional to n_runs
        alpha = 0.5

        max_timesteps = max(len(run_exposures) for run_exposures in self.all_new_exposures)
        plt.figure(figsize = (10, 6))

        #Calculate individual curves
        all_curves = []
        for run in range(n_runs):
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
        plt.title(f"Cumulative Incidence Spaghetti Plot\n{self.n_runs} runs, red = die-out")
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

    






