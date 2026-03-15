

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import igraph as ig
from typing import Dict, Any, Optional, Callable, List
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from numba import njit
from numba.typed import List as NumbaList
import warnings
from line_profiler import profile


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
        self.I0 = initial_infectious
        if isinstance(initial_infectious, int):
            initial_infectious = self.rng.integers(low = 0, high = self.N, size = initial_infectious).tolist()
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
        #shape is gamma_alpha, scale is mean/shape
        return self.rng.gamma(shape = self.config.epi.gamma_alpha, scale = mean_inc/self.config.epi.gamma_alpha)
    
    def assign_infectious_period(self, inds):
        """
        Take a list of newly-assigned infectious indices and assign an infectious period
        """
        inds = np.atleast_1d(inds)
        mean_inf = np.where(self.is_vaccinated[inds], self.config.epi.infectious_period_vax, self.config.epi.infectious_period)


        return self.rng.gamma(shape = self.config.epi.gamma_alpha, scale = mean_inf/self.config.epi.gamma_alpha)
    
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

    @profile     
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

    @profile
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


    def results_to_df(self, metrics: List[str] = ["peakPrev", "peakTime", "outbreakSize"]) -> pd.DataFrame:
        """
        Writes a dataframe of summary metrics for each run containing specified metrics

        Args: metrics, a list of metrics to output, must match expected metrics allowed for. Currently supports:
        - peakPrev (maximum fraction I/N)
        - peakTime (time of peakPrev)
        - outbreakSize (unique number of nodes infected, including I0)

        Returns: pandas dataframe of summary outputs
        """
        if metrics is None:
            metrics = ["peakPrev", "peakTime", "outbreakSize"]

        #Check that metrics requested are supported
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

            states_list = self.all_states_over_time[run]
            
            #Build prevalence time series
            prevalences = []
            for timestep in states_list:
                if timestep is None:
                    nI = 0
                else:
                    try:
                        #Gather infectious nodes
                        I_nodes = timestep[2]
                        nI = int(len(I_nodes))
                    except Exception:
                        nI = 0
                prevalences.append(float(nI) / float(self.N))
            
            #Metrics:

            #peakPrev & peakTime
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

            #Outbreak size -- unique infected nodes (I0 plus all ever exposed)
            if "outbreakSize" in metrics:
                exposures_list = self.all_new_exposures[run]

                aggregate = _aggregate_exposures(exposures_list)

                #Combine I0 with all exposures for outbreak size
                init_I0 = np.array(self.I0, dtype = int)
                
                unique_infectious = np.unique(np.concatenate([init_I0, aggregate]))
                outbreakSize = int(len(unique_infectious))
                row["outbreakSize"] = outbreakSize

            
            #Add row to rows
            rows.append(row)

        
        #Build and order dataframe, handle typing
        cols = ["run_number"] + [metric for metric in metrics]
        df = pd.DataFrame(rows)
        if "peakTime" in metrics and "peakTime" in df.columns:
            df["peakTime"] = df["peakTime"].astype("int64")
        if "peakPrev" in metrics and "peakPrev" in df.columns:
            df["peakPrev"] = df["peakPrev"].astype(float)
        if "outbreakSize" in metrics and "outbreakSize" in df.columns:
            df["outbreakSize"] = df["outbreakSize"].astype("int64")
        #Put cols in order, NA if requested column wasn't calculated
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA

        return df[cols]
        
    def timeseries_to_df(self, type: str = "prevalence") -> pd.DataFrame:
        """
        Returns a wide pandas dataframe with a time eries for each run
        
        Args: 
            type: "incidence" or "prevalence" to provide specified timeseries
        """
        series_type = str(type).strip().lower()
        if series_type not in ("incidence", "prevalence"):
            raise ValueError(f"type must be 'incidence' or 'prevalence' (got {type})")
        n_runs = self.n_replicates

        #Prebuild arrays - should be same either way
        if series_type == "prevalence":
            lengths = [len(self.all_states_over_time[run]) for run in range(n_runs)]
        else:
            lengths = [len(self.all_new_exposures[run]) for run in range(n_runs)]

        T = max(lengths) if lengths else 0
        if T == 0:
            return pd.DataFrame({"run_number":np.arange(n_runs)})
        
        #Build prevalence arrays
        if series_type == "prevalence":
            data = np.zeros((n_runs, T), dtype = np.float32)
            for run in range(n_runs):
                states_list = self.all_states_over_time[run]
                max_t = min(T, len(states_list))
                for t in range(max_t):
                    timestep = states_list[t]
                    try:
                        nI = int(len(timestep[2]))
                    except Exception:
                        #just in case they are saved as arrays
                        try:
                            nI = int(np.asarray(timestep[2]).size)
                        except Exception:
                            nI = 0
                    data[run, t] = float(nI) / float(self.N)
        #incidence arrays
        else:
            data = np.zeros((n_runs, T), dtype = np.float32)
            for run in range(n_runs):
                exposures_list = self.all_new_exposures[run]
                max_t = min(T, len(exposures_list))
                for t in range(max_t):
                    arr = exposures_list[t]
                    count=0
                    try:
                        if hasattr(arr, "size"):
                            count = int(arr.size)
                        else:
                            count = int(len(arr))
                    except Exception:
                        count = 0
                    if t == 0 and count == 0:
                        I0 = getattr(self, "I0", None)
                        count = len(I0)

                    frac = count/self.N
                    data[run, t] = frac
        
        #build dataframe
        col_names = [f"t_{i}" for i in range(T)]
        df = pd.DataFrame(data, columns = col_names)
        df.insert(0, "run_number", np.arange(n_runs))

        return df

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

    






