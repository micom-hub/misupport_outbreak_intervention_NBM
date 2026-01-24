from __future__ import annotations
import os
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.stats import truncnorm
from typing import TypedDict, List, Dict, Any
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.animation as animation


from scripts.SynthDataProcessing import build_edge_list, build_individual_lookup

class ModelParameters(TypedDict):
    # Epi Params
    base_transmission_prob: float
    incubation_period: float
    infectious_period: float
    incubation_period_vax: float
    infectious_period_vax: float
    gamma_alpha: float #alpha value for gamma distribution of inc/inf periods

    relative_infectiousness_vax: float
    vax_efficacy: float
    vax_uptake: float
    susceptibility_multiplier_under_five: float #increase in susceptibility if age <= 5

    # Population Params
    hh_contacts: int
    wp_contacts: int
    sch_contacts: int
    gq_contacts: int
    cas_contacts: int
    hh_weight: float
    wp_weight: float
    sch_weight: float
    gq_weight: float
    cas_weight: float

    #LHD Params
    mean_compliance: float 
    lhd_employees: int
    lhd_discovery_prob: float
    lhd_workday_hrs: int
    lhd_default_int_reduction: float
    lhd_default_call_duration: float
    lhd_default_int_duration: int

    # Simulation Settings
    n_runs: int
    run_name: str #prefix for model run
    overwrite_edge_list: bool #Try to reload previously generated edge list for this run to save time
    simulation_duration: int  # days
    I0: List[int]
    seed: int
    county: str #county to run on
    state: str #state to run on
    record_exposure_events: bool
    save_plots: bool
    save_data_files: bool
    make_movie: bool
    display_plots: bool




DefaultModelParams: ModelParameters = {
    #Epdiemic Parameters
    "base_transmission_prob": 1,
    "incubation_period": 10.5,
    "infectious_period": 5,
    "gamma_alpha": 20,
    "incubation_period_vax": 10.5,
    "infectious_period_vax": 5,
    "relative_infectiousness_vax": 0.05,
    "vax_efficacy": .997,
    "vax_uptake": 0.85, 
    "susceptibility_multiplier_under_five": 2.0,

    #Contact Parameters
    "wp_contacts": 10,
    "sch_contacts": 10,
    "gq_contacts": 10,
    "cas_contacts": 5,
    "hh_weight": 1,
    "wp_weight": .5,
    "sch_weight": .6,
    "gq_weight": .3,
    "cas_weight": .1,

    #LHD Params
    "mean_compliance": 1,
    "lhd_employees": 10,
    "lhd_workday_hrs": 8,
    "lhd_discovery_prob": .25,
    "lhd_default_call_duration": 0.1,#in hours
    "lhd_default_int_reduction": 0.8,
    "lhd_default_int_duration": 10, #in days

    #Simulation Settings
    "n_runs": 5,
    "run_name" : "test_run",
    "overwrite_edge_list": False,
    "simulation_duration": 45,
    "I0": [906],
    "seed": 2026,
    "county": "Keweenaw",
    "state": "Michigan",
    "record_exposure_events": True, #Necessary for LHD Dynamics
    "save_plots": True,
    "save_data_files": True,
    "make_movie": False,
    "display_plots": False

}

#-----Outbreak Model-------
class NetworkModel:
    #@profile
    def __init__(self, contacts_df, params = DefaultModelParams):
        """
        Unpack parameters, set-up storage variables, and initialize model
        """
        #Unpack params and model settings
        self.params = params
        self.contacts_df = contacts_df
        self.N = self.contacts_df.shape[0]
        self.Tmax = self.params["simulation_duration"] #how long it could run
        self.rng = np.random.default_rng(self.params["seed"])
        self.county = self.params["county"]
        self.n_runs = self.params["n_runs"]

        #One-time computation of network structures
        self._compute_network_structures()

        #Set up storage for full run
        self.all_states_over_time = [None]*self.n_runs
        self.all_new_exposures = [None]*self.n_runs
        self.exposure_event_log = []
        for _ in range(self.n_runs):
            self.exposure_event_log.append([])

        self.all_stochastic_dieout = np.zeros(self.n_runs, dtype = bool)
        self.all_end_days = np.ones(self.n_runs, dtype = int)*self.Tmax


        #Instantiate recorder and Local Health Department
        self.recorder_template = ExposureEventRecorder(init_event_cap = 1024, init_node_cap = 4096)

        self.lhd = LocalHealthDepartment(model = self, rng = self.rng, discovery_prob = self.params["lhd_discovery_prob"], employees = self.params["lhd_employees"], workday_hrs = self.params["lhd_workday_hrs"], algorithm = None)

        self.lhd.algorithm = EqualPriority()



#Set-up results folder
        
        self.results_folder = os.path.join(os.getcwd(), 
            "results", self.params["run_name"])
        if self.params["save_data_files"]:
            if not os.path.exists(self.results_folder):
                os.mkdir(self.results_folder)

    #Helper functions to convert between
    #Name: an individual's number (index of contact_df/individual_lookup))
    #Ind: the index of the individual's vertex in the model

    #@profile
    def _compute_network_structures(self):
        """
        Compute expensive network structures that are preserved across runs:
        - edge list
        - adj matrix
        - individual lookup
        - neighbor_map and fast neighbor map
        - full node list
        - layout node names
        -layout name to ind
        """
        #Create edge list
        if (not self.params["overwrite_edge_list"]) and os.path.isfile(
            os.path.join(os.getcwd(), 
            "data", 
            self.county, 
            (self.params["run_name"]+ "_edgeList.parquet")
            )):
            print(f"Edge list found for {self.params['run_name']}, reading...")
            self.edge_list = pd.read_parquet(os.path.join(os.getcwd(), 
            "data", 
            self.county, 
            (self.params["run_name"]+ "_edgeList.parquet")
            ))
        else:
            self.edge_list = build_edge_list(
                contacts_df = self.contacts_df,
                params = self.params, 
                rng = self.rng, 
                save = self.params["save_data_files"],
                county = self.county)

        #Create full adjacency matrix
        src = self.edge_list['source'].to_numpy(dtype = np.int32)
        tgt = self.edge_list['target'].to_numpy(dtype = np.int32)
        weights = self.edge_list['weight'].to_numpy(dtype = np.float32)
        N_nodes = self.N

        row = np.concatenate([src, tgt])
        col = np.concatenate([tgt, src])
        dat = np.concatenate([weights, weights])
        self.adj_matrix = csr_matrix(
            (dat, (row, col)), shape = (N_nodes, N_nodes)
            )

        #Create individual lookup table
        self.individual_lookup = build_individual_lookup(self.contacts_df)

        #Initialize individual vectors for quicker lookup
        self.ages = self.individual_lookup["age"].to_numpy()
        self.sexes = self.individual_lookup["sex"].to_numpy()

        #Set individual compliances array
        avg_compliance = np.clip(self.params["mean_compliance"], 0, 1) #0-1
        sd = 0.15 #sd for bounded normal distribution
        #calculate truncnorm params
        a = (0 - avg_compliance) / sd
        b = (1 - avg_compliance) / sd
        self.compliances = truncnorm.rvs(a,b,
        loc = avg_compliance, scale = sd,
        size = self.N)


        #Create neighbor_map and fast_neighbor_map
        self.neighbor_map = self._build_neighbor_map()

        self.fast_neighbor_map = {} #preprocess dict of dicts to speed up later
        for src, neighbor_list in self.neighbor_map.items():
            self.fast_neighbor_map[src] = {
                tgt: (weight, ct) for tgt, weight, ct in neighbor_list
                }

        #Create adjacency sparses by contact type 
        #lookup indptr, indices, weights = csr_by_type['ct']
        self.csr_by_type = self._build_type_csr()
        self.contact_types = sorted(self.csr_by_type.keys()) #stable ct order
        self.ct_to_id = {ct: i for i, ct in enumerate(self.contact_types)}
        self.id_to_ct = {i: ct for ct, i in self.ct_to_id.items()}




        #Create full node list
        self.full_node_list = sorted(
            set(self.edge_list['source']).union(
                set(self.edge_list['target'])
                ))

        #Create node dict and full layout ind
        name_to_ind_dict = {name:ind for ind, name in enumerate(self.full_node_list)}

        #Create a full graph

        g_full = ig.Graph()
        g_full.add_vertices(len(self.full_node_list))
        #Node names in the graph correspond to model indices
        g_full.vs['name'] = self.full_node_list 

        full_edges = list(zip(
            [name_to_ind_dict[src] for src in self.edge_list['source']],
            [name_to_ind_dict[tgt] for tgt in self.edge_list['target']]
        ))
        g_full.add_edges(full_edges)
        self.fixed_layout = g_full.layout('grid') #set a layout for the full graph
        self.layout_node_names = self.full_node_list
        self.layout_name_to_ind = {name:i for i, name in enumerate(self.layout_node_names)}
        self.g_full = g_full.copy()

    def _initialize_states(self):
        """
        Set up new SEIR arrays and seed initial infectious individuals
        """
       #Set vaccinations
        self.is_vaccinated = self.rng.random(self.N) < self.params["vax_uptake"]

        #State tracking variables; S = 0, E = 1, I = 2, R = 3
        self.state = np.zeros(self.N, dtype = np.int8) 
        self.time_in_state = np.zeros(self.N, dtype = np.float32)
        self.incubation_periods = np.full(self.N, np.nan, dtype = np.float32)
        self.infectious_periods = np.full(self.N, np.nan, dtype = np.float32)

        #Per-run trajectories
        self.states_over_time = [] 
        self.new_exposures = []
        self.new_infections = []

        initial_infectious = self.params["I0"]
        self.state[initial_infectious] = 2
        self.infectious_periods[initial_infectious] = self.assign_infectious_period(initial_infectious)

        S = list(set(range(self.N)) - set(initial_infectious))
        E = []
        I = initial_infectious  # noqa: E741
        R =  []

        self.states_over_time.append([S,E,I,R])
        self.new_exposures.append([])
        self.new_infections.append(initial_infectious)

        #Reset run flags
        self.simulation_end_day = self.Tmax
        self.stochastic_dieout = False

        #Reset LHD intervention Multipliers
        self.in_multiplier = {ct: np.ones(self.N, dtype = np.float32) for ct in self.contact_types}
        self.out_multiplier = {ct: np.ones(self.N, dtype = np.float32) for ct in self.contact_types}
       
        
    def name_to_ind(self, g: ig.Graph, names):
        """Convert individual's names to igraph vertex indices

        Args:
            g (ig.Graph): a graph with named nodes
            names (int/list/np.array): a list of individual names 

        Returns: graph indices of the same datatype as names
        """
        name_to_index = {int(v["name"]): v.index for v in g.vs}
        if isinstance(names, (int, np.integer)):
            return name_to_index[names]
        elif isinstance(names, (list, np.ndarray)):
            inds = [name_to_index[n] for n in names]
            return np.array(inds, dtype = int)
        else:
            raise TypeError("names must be int, list, or numpy array.")
        
    def ind_to_name(self, g: ig.Graph, inds):
        """Convert nodes of an igraph to their corresponding names

        Args:
            g (ig.Graph): an igraph object with named nodes
            inds (int/list/np.array/range): a list of indices present in g

        Returns: Names of graph indices as np.array
        """
        if isinstance(inds, (int, np.integer)):
            return g.vs[inds]["name"]
        elif isinstance(inds, (list, np.ndarray, range)):
            names = [int(g.vs[i]["name"]) for i in inds]
            return np.array(names, dtype = int)

        else:
            raise TypeError("inds must be int, list, or numpy array.")
            
    #@profile
    def _build_neighbor_map(self):
        """
        Build a mapping from node to list of (neighbor, weight) (make neighbor queries O(1))
        """
        #build a mapping of src: (tgt, weight, ct)
        neighbor_map = {}
        for row in self.edge_list.itertuples(index = False):
            src, tgt, w, ct = row.source, row.target, row.weight, row.contact_type
            if src not in neighbor_map:
                neighbor_map[src] = []
            neighbor_map[src].append((tgt, w, ct))
            #symmetrize here, since edge_list is undirected from i, j where i<j
            if tgt not in neighbor_map:
                neighbor_map[tgt] = []
            neighbor_map[tgt].append((src, w, ct))



        return neighbor_map

    def _build_type_csr(self):
        """
        Build an csr sparses by contact type that map a source node to a list of neighbors by type
        """

        neighbor_map = self.neighbor_map
        N = self.N
        total_edges = int(2*self.edge_list.shape[0])

    #collect global arrays

        src_array = np.empty(total_edges, dtype = np.int32)
        tgt_array = np.empty(total_edges, dtype = np.int32)
        wt_array = np.empty(total_edges, dtype = np.float32)

        ct_to_id = {}
        id_to_ct = []
        ct_array = np.empty(total_edges, dtype = np.int16)

        pos = 0
        for src, neighbors in neighbor_map.items():
            for tgt, wt, ct in neighbors:
                src_array[pos] = src
                tgt_array[pos] = tgt
                wt_array[pos] = wt
                if ct not in ct_to_id:
                    ct_to_id[ct] = len(id_to_ct)
                    id_to_ct.append(ct)
                ct_array[pos] = ct_to_id[ct]
                pos += 1

        #slice to actual length
        if pos < total_edges:
            src_array = src_array[:pos]
            tgt_array = tgt_array[:pos]
            wt_array = wt_array[:pos]
            ct_array = ct_array[:pos]

    #sort edges by (ct, src) so they are contiguous and sources are grouped

        order = np.lexsort((src_array, ct_array)) #primary key ct, secondary src
        src_s = src_array[order]
        tgt_s = tgt_array[order]
        wt_s = wt_array[order]
        ct_s = ct_array[order]

    #split by type and build a csr for each type
        csr_by_type = {}
        unique_cts, ct_starts = np.unique(ct_s, return_index = True)
        ct_starts = list(ct_starts) + [len(ct_s)]
        for k, ct_id in enumerate(unique_cts):
            start = ct_starts[k]
            end = ct_starts[k+1]
            srcs_ct = src_s[start:end]
            tgts_ct = tgt_s[start:end]
            wts_ct = wt_s[start:end]

            if srcs_ct.size == 0:
                indptr = np.zeros(N+1, dtype = np.int64)
                indices = np.empty(0, dtype = np.int32)
                weights = np.empty(0, dtype = np.float32)
            else:
                counts = np.bincount(srcs_ct, minlength = N)
                indptr = np.empty(N+1, dtype = np.int64)
                indptr[0] = 0
                np.cumsum(counts, out = indptr[1:])
                indices = tgts_ct.astype(np.int32, copy = True)
                weights = wts_ct.astype(np.float32, copy = True)

            csr_by_type[id_to_ct[int(ct_id)]] = (indptr, indices, weights)

        return csr_by_type

    
    def assign_node_attribute(self, attr_name: str, vals: np.ndarray, indices: np.ndarray, g: ig.Graph) -> ig.Graph:
        """Assigns a node attribute to specified nodes in an igraph Graph object

        Args:
            attr_name (str): The name of the attribute to be assigned
            vals (np.ndarray): An array of values to be assigned
            indices (np.ndarray): An array of MODEL INDICES (node names) for assignment
            g (ig.Graph): an igraph containing the indices provided

        Returns:
            ig.Graph: The provided igraph with added attribute names

        DO NOT USE for assigning node names
        """

        if len(vals) != len(indices):
            raise ValueError("Length of values must match length of indices")

        available_names = set(g.vs["name"])
        if (set(indices) - available_names):
            raise ValueError("Indices not present in the graph cannot have attributes added")

        
        name_to_ind = {v["name"]: v.index for v in g.vs}

        node_ind = [name_to_ind[ind] for ind in indices]
        g.vs[node_ind][attr_name] = vals
        return g

        
    def assign_incubation_period(self, inds):
        """
        Take a list of newly-assigned exposed indices and assign an incubation period
        """
        inds = np.atleast_1d(inds)
        mean_inc = np.where(self.is_vaccinated[inds], self.params["incubation_period_vax"], self.params["incubation_period"])/self.params["gamma_alpha"]
        return self.rng.gamma(shape = self.params["gamma_alpha"], scale = mean_inc)
    
    def assign_infectious_period(self, inds):
        """
        Take a list of newly-assigned infectious indices and assign an infectious period
        """
        inds = np.atleast_1d(inds)
        mean_inf = np.where(self.is_vaccinated[inds], self.params["infectious_period_vax"], self.params["infectious_period"])/self.params["gamma_alpha"]


        return self.rng.gamma(shape = self.params["gamma_alpha"], scale = mean_inf)

    #@profile
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
        base_prob = self.params['base_transmission_prob']
        vax_efficacy = self.params['vax_efficacy']
        susc_mult_under5 = self.params['susceptibility_multiplier_under_five']
        rel_inf_vax = self.params['relative_infectiousness_vax']

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
                if susc_mult_under5 != 1.0:
                    prob[under5] = prob[under5] * susc_mult_under5
                
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
            self.time_in_state[to_recovered] = 0

        #TODO R -> S with waning immunity


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
            print(f"Running model run {run + 1} of {self.n_runs}...")
            self._initialize_states()

            t = 0
            while t < self.Tmax:
                t += 1 #day 0 is recorded in initialization
                self.current_time = t

                if self.params.get("record_exposure_events"):
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
                

            self.all_states_over_time[run] = [states.copy() for states in self.states_over_time]
            self.all_new_exposures[run] = [ne.copy() if hasattr(ne, "copy") else list(ne) for ne in self.new_exposures]
            self.all_stochastic_dieout[run] = self.stochastic_dieout
            self.all_end_days[run] = self.simulation_end_day

    def epi_summary(self) -> pd.DataFrame:
        """
        Computes summary statistics for. each run on the model

        Returns a pandas DataFrame with a row for each run, and columns:
        - Total_infections
        - epidemic_duration
        - peak_infections
        - peak_prevalence
        - time_of_peak_infection
        - percent_unvax_infected
        - percent_vax_infected
        - stochastic_dieout
        """
        N = self.N
        try: 
            vax_status = self.is_vaccinated
        except AttributeError:
            raise ValueError("Model states not initialized; run at least one simulation before computing summary statistics")
        
        summary = []
        for run in range(self.n_runs):
            exposures = self.all_new_exposures[run]
            states_over_time = self.all_states_over_time[run]
            stochastic_dieout = bool(self.all_stochastic_dieout[run])
            epidemic_timne = self.all_end_days[run]

            #Determine unique infections
            ever_exposed = np.zeros(N, dtype = bool)
            for exp in exposures:
                ever_exposed[np.array(exp, dtype = int)] = True
            total_infections = int(ever_exposed.sum())

            #Peak infections and timing
            infectious_counts = [len(state[2]) for state in states_over_time]
            if infectious_counts:
                peak_infectious = int(max(infectious_counts))
                time_of_peak = int(np.argmax(infectious_counts))
            else:
                peak_infectious = 0
                time_of_peak = None

            peak_prevalence = peak_infectious / N

            #Vaccination-stratified infection rates
            n_unvax = int((~vax_status).sum())
            n_vax = int(vax_status.sum())

            pct_unvax_infected = (ever_exposed[~vax_status].sum() / n_unvax) if n_unvax else np.nan
            pct_unvax_infected = (ever_exposed[vax_status].sum() / n_vax) if n_vax else np.nan

            summary.append({
                "run_number": run,
                "total_infections": total_infections,
                "epidemic_duration": epidemic_timne,
                "peak_infections": peak_infectious,
                "peak_prevalence": peak_prevalence,
                "time_of_peak_infection": time_of_peak,
                "pct_unvax_infected": pct_unvax_infected,
                "pct_vax_infected": pct_unvax_infected,
                "stochastic_dieout": stochastic_dieout
            })
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
        counts[0] = len(self.params["I0"]) #add initial infections
        plt.figure(figsize=((8,4)))
        plt.bar(range(len(counts)), counts, color = 'orange', label = "Infections Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Number of Infectious Contacts Made")
        plt.grid(axis = 'y', alpha = 0.5)
        plt.tight_layout()
        if self.params["save_plots"]:
            plotpath = os.path.join(self.results_folder,"epi_curve")
            if suffix:
                plotpath = plotpath + suffix
            

            plt.savefig(f"{plotpath}.png")

        if self.params["display_plots"]:
            plt.show()
        plt.close()

        return

    def cumulative_incidence_plot(self, run_number = 0, strata:str = None, time: int = None, suffix: str = None) -> None:
        """
        Plots cumulative incidence over time, optionally stratified

        Args:
            time (int): time range for plot, 0-t, defaults to full timespan
            run_number (int): If model run multiple times, specify which to viz
            strata (str )  stratification factor for plot

        Creates a matplotlib plot
        """
        pop_size = self.N
        exposures = self.all_new_exposures[run_number]
        if len(exposures) > 0 and len(exposures[0]) == 0:
            exposures[0] = list(self.params["I0"])

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

        if strata in ("age", "sex"):
            strata_attr = self.individual_lookup[strata]

            if strata == "age":
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
                strata_vals = strata_attr
                labels = sorted(strata_attr.unique())
                strata_labels = labels
                label_to_color = {lab: ("red" if lab == "F" else "blue") for lab in labels}
                strata_colors = [label_to_color[lab] for lab in labels]


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
            plt.title(f"Cumulative Incidence (Stratified by {strata})\nRun {self.params['run_name']}")
            legend_handles = [Patch(color = strata_colors[i], label = str(label)) for i, label in enumerate(strata_labels)]
            plt.legend(handles = legend_handles, loc = "upper left")
        else:
            plt.title(f"Cumulative Incidence Over Time for  {self.params['run_name']}")
        plt.grid(True, axis = "y", alpha = 0.5)
        plt.tight_layout()
        plotpath = os.path.join(self.results_folder, "cumulative_incidence")
        if suffix:
            plotpath = plotpath + suffix
        if self.params["save_plots"]:
            if strata:
                plt.savefig(f"{plotpath}_by{strata}.png")
            else: 
                plt.savefig(f"{plotpath}.png")
        if self.params["display_plots"]:
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
        if self.params["save_plots"]:
            plt.savefig(f"{plotpath}.png")
        if self.params["display_plots"]:
            plt.show()
        plt.close()
        
    def draw_network(self, t: int, run_number = 0, ax=None, clear: bool =True, saveFile: bool = False, suffix: str = None):
        """Visualizes the active subnetwork at a time t, that consists of all "active nodes" (E, I, or R) and all of their susceptible neighbors

        Args:
            t (int): timestep to plot
            run_number: run to visualize, if multiple model runs
            ax (matplotlib axis, optional): axis for plotting
            clear (bool): whether to clear axis
            saveFile (bool): save to file
            suffix (str): filename suffix
        """

        #get indices of E, I, R
        S, E, I, R = self.all_states_over_time[run_number][t] # noqa: E741
        affected_inds = set(E) | set(I) | set(R) 

        #get direct neighbors of affected nodes
        neighbors_set = set()
        for node in affected_inds:
            node_index = self.layout_name_to_ind[node]
            nbr_indices = self.g_full.neighbors(node_index)
            nbr_names = [self.g_full.vs[nbr]["name"] for nbr in nbr_indices]
            neighbors_set.update(nbr_names)
        #combine affected nodes and neighbors
        plot_nodes = sorted(affected_inds | neighbors_set)

        #create a subgraph from self.g_full with. these nodes
        node_name_to_index = {v["name"]: v.index for v in self.g_full.vs}
        subgraph_indices = [node_name_to_index[name] for name in plot_nodes]
        subgraph = self.g_full.subgraph(subgraph_indices)

        #assign colors by state
        #note that subgraph reindexes nodes so need to remap
        color_map = {}
        for v in subgraph.vs:
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

        colors = [color_map[v["name"]] for v in subgraph.vs] 

        #Use full graph layout
        indices_in_full = [self.layout_name_to_ind[v["name"]] for v in subgraph.vs]
        layout = [self.fixed_layout[i] for i in indices_in_full]
        node_labels = subgraph.vs["name"]

        #plot
        if ax is None:
            fig, ax = plt.subplots(figsize = (10, 10))
            show_plot = True
        else:
            show_plot = False
        if clear:
            ax.clear()
        
        ig.plot(
            subgraph,
            layout = layout,
            vertex_color = colors,
            vertex_size = 15,
            edge_color = "gray",
            bbox = (600, 600),
            target = ax,
            vertex_label = node_labels if len(subgraph.vs) <= 40 else None
        )
        ax.set_title(f"Network at t = {t}")

        #Save if requested
        if saveFile:
            plotpath = os.path.join(self.results_folder, f"network_at_{str(t)}")
            if suffix:
                plotpath = plotpath + suffix
            plt.savefig(f"{plotpath}.png")
        if show_plot and self.params["display_plots"]:
            plt.show()
        plt.close()
    

        return

    def make_movie(self, dt: int = 1, run_number = 0, filename: str = "network_outbreak.mp4", fps: int = 3):
        """_summary_

        Args:
            run_number: which run to create a movie for if multiple model runs
            dt (int, optional): Time interval between network visualizations Defaults to 1.
            filename (str, optional): Name to save to Defaults to "network_outbreak.mp4".
            fps (int, optional): Defaults to 3.

        Creates a .mp4 file of the network evolution saved in results/run_name
        """
        if not self.params["save_data_files"]:
            raise Exception("save_data_files is false, but a network movie was requested")
        out_path = os.path.join(self.results_folder, filename)

        fig, ax = plt.subplots(figsize = (8,8))

        #get timesteps
        timesteps = list(range(0, len(self.all_states_over_time[run_number]), dt))
        if (len(self.states_over_time)-1) not in timesteps:
            timesteps.append(len(self.states_over_time)-1) #always show last step
        
        def update(frame):
            t = timesteps[frame]
            ax.clear()
            self.draw_network(t, ax = ax, clear = False)
            ax.set_title(f"{self.params['run_name']} | Day {t}")
            return ax
        
        ani = animation.FuncAnimation(fig, update, frames = len(timesteps), interval = 1000/fps, blit = False, repeat = False)

        print("Saving outbreak movie...")
        #ffmpeg MUST BE INSTALLED
        try:
            ani.save(out_path, writer = 'ffmpeg', fps = fps)
        except(FileNotFoundError, ValueError) as e:
            print("\nERROR: FFmpag is needed to save mp4 movies, but wasn't found")
            print("Install ffmpeg and ensure it is in your system path")
            print(f"Error Message: {e}")
        finally:
            plt.close(fig)

        print(f"Outbreak movie saved to {out_path}...")

        return

    def make_graphml_file(self, t: int, run_number:int = 0, suffix: str = None):
        """ Visualize the network at a given timestep using grephi

        Args:
            t (int): a timestep of the network to visualize
            run_number(int): a run number to produce the graph for
        Returns:
            Saves a .graphml of the network to open externally
        """

        #Get neighbors of affected nodes from g_full
        S, E, I, R = self.all_states_over_time[run_number][t]  # noqa: E741
        affected_inds = set(E) | set(I) | set(R)

        neighbors_set = set()
        for node in affected_inds:
            node_index = self.layout_name_to_ind[node]
            nbr_indices = self.g_full.neighbors(node_index)
            nbr_names = [self.g_full.vs[nbr]["name"] for nbr in nbr_indices]
            neighbors_set.update(nbr_names)
        plot_nodes = sorted(affected_inds | neighbors_set)

        node_name_to_index = {v["name"]: v.index for v in self.g_full.vs}
        subgraph_indices = [node_name_to_index[name] for name in plot_nodes]
        subgraph = self.g_full.subgraph(subgraph_indices)


        def igraph_to_networkx(g: ig.Graph) -> nx.Graph:

            """
            Takes an igraph graph object and converts it to a networkX graph to be visualized with grephi

            Args:
                g (ig.Graph): an igraph Graph

            Returns:
                nx.Graph: a networkX graph object
            """
          
            G = nx.Graph()
            #add attributes
            for v in g.vs:
                node_id = v["name"]
                attrs = v.attributes().copy()
                attrs.pop("name", None) #don't duplicate name
                G.add_node(node_id, **attrs)
            for e in g.es:
                u = g.vs[e.source]["name"]
                v = g.vs[e.target]["name"]
                attrs = e.attributes().copy()
                G.add_edge(u, v, **attrs)
            return G

        nx_g = igraph_to_networkx(subgraph)
        self.nx_g = nx_g
        if self.params["save_data_files"]:
            netpath = os.path.join(self.results_folder, "networkfile")
            if suffix:
                netpath = netpath + suffix
            nx.write_graphml(nx_g, (netpath + ".graphml"))

        return


#------ Exposure Event Tracking --------
class ExposureEventRecorder:
    """
    Object to record exposure events for a single timestep or across timesteps if reset. Stores:
    - exposure event metadata arrays(time, source, type_id, start, length)
    - nodes (concatenated node ids)
    - infections (concatenated bool that align with nodes)
    Methods:
    - append_event(time, source, type_id, nodes_arr, infected_mask)
    - reset() to reuse the recorder
    - snapshot_compact(copy = True) -> dict of numpy arrays (sliced to used lengths)
    - to_dataframe(id_to_ct) -> pandas DF with nodes and infected as arrays
    """
    def __init__(self, init_event_cap = 1024, init_node_cap = 4096):
        self.event_cap = int(init_event_cap) 
        self.node_cap = int(init_node_cap)
        self._alloc_arrays()
        self.reset()

    def _alloc_arrays(self):
        self.event_time = np.empty(self.event_cap, dtype = np.int32)
        self.event_source = np.empty(self.event_cap, dtype = np.int32)
        self.event_type = np.empty(self.event_cap, dtype = np.int16)
        self.event_nodes_start = np.empty(self.event_cap, dtype = np.int64)
        self.event_nodes_len = np.empty(self.event_cap, dtype = np.int32)
        self.nodes = np.empty(self.node_cap, dtype = np.int32)
        self.infections = np.empty(self.node_cap, dtype = np.bool_)

    def reset(self):
        self.n_events = 0
        self.n_nodes = 0

    def _grow_events(self, min_extra = 1):
        if self.n_events + min_extra <= self.event_cap:
            return
        newcap = max(self.event_cap * 2, self.n_events + min_extra)
        def grow(arr):
            new = np.empty(newcap, dtype = arr.dtype)
            new[: self.n_events] = arr[:self.n_events]
            return new
        
        self.event_time = grow(self.event_time)
        self.event_source = grow(self.event_source)
        self.event_type = grow(self.event_type)
        self.event_nodes_start = grow(self.event_nodes_start)
        self.event_nodes_len = grow(self.event_nodes_len)
        self.event_cap = newcap

    def _grow_nodes(self, min_extra = 1):
        if self.n_nodes + min_extra <= self.node_cap:
            return
        newcap = max(self.node_cap * 2, self.n_nodes + min_extra)
        new_nodes = np.empty(newcap, dtype = np.int32)
        new_nodes[: self.n_nodes] = self.nodes[: self.n_nodes]
        new_inf = np.empty(newcap, dtype = np.bool_)
        new_inf[: self.n_nodes] = self.infections[: self.n_nodes]
        self.nodes = new_nodes
        self.infections = new_inf
        self.node_cap = newcap

    def append_event(self, time: int, source: int, type_id: int, nodes_arr: np.ndarray, infected_mask: np.ndarray):
        nodes_arr = np.asarray(nodes_arr, dtype = np.int32)
        infected_mask = np.asarray(infected_mask, dtype = np.bool_)
        if not nodes_arr.shape[0] == infected_mask.shape[0]:
            raise ValueError("Recorder event has mismatched arrays")

        L = nodes_arr.shape[0]

        self._grow_events(1)
        if L:
            self._grow_nodes(L)
            start = self.n_nodes
            self.nodes[start:start + L] = nodes_arr
            self.infections[start:start+L] = infected_mask
        else:
            start = self.n_nodes
        
        #gather metadata
        i = self.n_events
        self.event_time[i] = np.int32(time)
        self.event_source[i] = np.int32(source)
        self.event_type[i] = np.int16(type_id)
        self.event_nodes_start[i] = np.int64(start)
        self.event_nodes_len[i] = np.int32(L)

        self.n_nodes += L
        self.n_events += 1

    def snapshot_compact(self, copy:bool = True) -> Dict[str, np.ndarray]:
        """
        Returns a dict of np.arrays sliced to used lengths

        Args:
            copy (bool): If true, arrays are copied and saved even if recorder is reused
        """
        event = slice(0, self.n_events)
        node = slice(0, self.n_nodes)
        if copy:
            return{
                'event_time': self.event_time[event].copy(),
                'event_source': self.event_source[event].copy(),
                'event_type': self.event_type[event].copy(),
                'event_nodes_start': self.event_nodes_start[event].copy(),
                'event_nodes_len': self.event_nodes_len[event].copy(),
                'nodes': self.nodes[node].copy(),
                'infections': self.infections[node].copy(),
            }
        else:
            return {
                'event_time': self.event_time[event],
                'event_source': self.event_source[event],
                'event_type': self.event_type[event],
                'event_nodes_start': self.event_nodes_start[event],
                'event_nodes_len': self.event_nodes_len[event],
                'nodes': self.nodes[node],
                'infections': self.infections[node],
            }

    def to_dataframe(self, id_to_ct: Dict[int, str]) -> pd.DataFrame:
        """
        Convert compact snapshot to a pandas dataframe. Computationally expensive

        Args:
            id_to_ct (Dict[int, str]): index to contact type mapping

        """
        rows = []
        for i in range(self.n_events):
            s = int(self.event_nodes_start[i])
            L = int(self.event_nodes_len[i])
            nodes = self.nodes[s:s+L].copy()
            infs = self.infections[s:s+L].copy()
            n_infected = sum(infs)
            rows.append((
                int(self.event_time[i]),
                int(self.event_source[i]),
                id_to_ct[int(self.event_type[i])],
                nodes,
                infs,
                L,
                n_infected
               
            ))
        df = pd.DataFrame(rows, columns = ['time','source','contact_type','nodes','infected', 'n_exposed', 'n_infected'])
        return(df)


#------LHD Stuff --------
class Action:
    """
    Generic action to be performed by the local health department 
    """
    def __init__(self, kind: str, nodes: np.ndarray, contact_types: List[str],
    reduction: float, duration: int, reason: str = None, meta: dict = None):
        self.kind = kind
        self.nodes = np.asarray(nodes, dtype = np.int32)
        self.contact_types = list(contact_types)
        self.reduction = float(reduction)
        self.duration = int(duration)
        self.reason = reason
        self.meta = meta or {}

class AlgorithmBase:
    """
    An interface for designing calling algorithms for the local health department.
    
Algorithms have one method, generate_candidates, which takes:
    a snapshot of exposure events
    a NetworkModel object 
    the indices of discovered events
    
    and returns arrays:
         nodes
         relative priority prioritization
         contact type
         projected call duration (call cost)
    """
    def generate_candidates(self, recorder_snapshot: Dict[str, np.ndarray], model, discovered_event_ind) -> Dict[str, Any]:
        """

        Args:
            Snapshot of exposure events in an outbreak
        Returns:
            Dict[str, Any]: Dict with keys:
                'nodes' : np.ndarray of node ids
                'priority' np.ndarray of float priorities aligned w/ nodes
                'contact_types': array of ct (per candidate or single)
                other metadata
        """
        raise NotImplementedError

class LocalHealthDepartment:
    def __init__(
        self, model: NetworkModel, rng = None, 
    discovery_prob: float = None,  
    employees: int = None, workday_hrs: float = None,
    algorithm: AlgorithmBase = None
    ):
    #LHD settings
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng()
        self.discovery_prob = discovery_prob if discovery_prob is not None else self.model.params["lhd_discovery_prob"]

    #Determine Capacity
        self.employees = employees if employees is not None else self.model.params["lhd_employees"]
        self.hours_per_employee = float(workday_hrs) if workday_hrs is not None else self.model.params["lhd_workday_hrs"]
        self.daily_personhours = float(self.employees * self.hours_per_employee)

    #LHD Actions and Algorithm assignment
        self.algorithm = algorithm
        self.expiry = {}
        self.action_log = []

        self.min_factor = 1e-6 #to prevent div 0 errors
        self.min_candidate_cost = 1e-4

    #Default action params
        self.default_int_reduction = model.params.get("lhd_default_int_reduction", 0.8)
        self.default_int_duration = model.params.get("lhd_default_int_duration", 7)
        self.default_call_cost = model.params.get("lhd_default_call_duration", 0.083)

    def discover_exposures(self, recorder_snapshot):
        """
        Given recorder snapshot dict, select which events the LHD discovers
        Random based on LHD discovery probability
        """
        n_events = recorder_snapshot['event_time'].shape[0]
        if n_events == 0:
            return np.empty(0, dtype = int)
        
        #bernoulli sample each event
        mask = self.rng.random(n_events) < self.discovery_prob
        return np.where(mask)[0]

    def gather_candidates(self, recorder_snapshot, discovered_event_ind):
        """
        Uses an algorithm to generate candidates and processes results. 
        Deduplicates by node, keeping highest priority call (if you had to call someone due to multiple exp)
        On ties in priority, prefers lower cost
        
        Returns: (unique_nodes, max_prios, sel_cts, sel_costs)
        #Give this a memory from past events
        """
        #if empty-return
        empty_nodes = np.empty(0, dtype = np.int32)
        empty_prios = np.empty(0, dtype = np.float32)
        empty_cts = np.empty(0, dtype = object)
        empty_costs = np.empty(0, dtype = np.float32)

        if len(discovered_event_ind) == 0:
            return empty_nodes, empty_prios, empty_cts, empty_costs

        #Pick algorithm to use and generate candidates
        algo = getattr(self, "algorithm", None)
        if algo is None:
            return empty_nodes, empty_prios, empty_cts, empty_costs
        
        out = algo.generate_candidates(recorder_snapshot, self.model, discovered_event_ind)

        #Initialize arrays with correct types
        nodes = np.asarray(out.get("nodes", empty_nodes), dtype = np.int32)
        prios = np.asarray(out.get("priority", np.ones(nodes.shape[0], dtype = np.float32)), dtype = np.float32)

        #contact types array, normalized to length of nodes 
        # (if algo only prioritizes one contact type)
        cts = out.get("contact_types", None)
        if cts is None:
            default_ct = self.model.contact_types[0] if getattr(self.model, "contact_types", ) else ""
            sel_cts_arr = np.array([default_ct]*nodes.shape[0], dtype = object)
        elif isinstance(cts, (str, bytes)):
            sel_cts_arr = np.array([cts] * nodes.shape[0], dtype = object)
        else:
            sel_cts_arr = np.asarray(cts)
            if sel_cts_arr.shape[0] != nodes.shape[0]:
                if sel_cts_arr.size == 1: #if algo returns np.array of one ct
                    sel_cts_arr = np.repeat(sel_cts_arr[0], nodes.shape[0]).astype(object)
                else:
                    raise ValueError("contact_types length from algo must match nodes length or be a single value")

        #Do the same for costs
        costs = out.get("costs", None)
        if costs is None:
            default_cost = self.default_call_cost
            sel_costs_arr = np.full(nodes.shape[0], default_cost, dtype = np.float32) 
        else:
            sel_costs_arr = np.asarray(costs, dtype = np.float32)
            if sel_costs_arr.shape[0] != nodes.shape[0]:
                if sel_costs_arr.size == 1: #if algo returns np.array of one ct
                    sel_costs_arr = np.repeat(sel_costs_arr[0], nodes.shape[0]).astype(np.float32)
                else:
                    raise ValueError("costs length from algo must match nodes length or be a single value")

        if nodes.size == 0:
            return empty_nodes, empty_prios, empty_cts, empty_costs


        #deduplicate by node, choosing highest priority occurrence and tie-breaking with lower cost
        unique_nodes, inverse = np.unique(nodes, return_inverse = True)
        max_prios = np.full(unique_nodes.shape[0], -np.inf, dtype = np.float32)
        chosen_cts = np.empty(unique_nodes.shape[0], dtype = object)
        chosen_costs = np.full(unique_nodes.shape[0], np.inf, dtype = np.float32)

        #scan across occurrences, taking max priority
        for occ_ind in range(nodes.shape[0]):
            u = inverse[occ_ind] 
            p = prios[occ_ind]
            c = float(sel_costs_arr[occ_ind])

            if p > max_prios[u]:
                max_prios[u] = p
                chosen_cts[u] = sel_cts_arr[occ_ind]
                chosen_costs[u] = c
            elif p == max_prios[u]: 
                #tie break with lowest cost
                if c < chosen_costs[u]:
                    chosen_cts[u] = sel_cts_arr[occ_ind]
                    chosen_costs[u] = c

        #floor on call costs
        chosen_costs = np.maximum(chosen_costs, self.min_candidate_cost).astype(np.float32)


        return unique_nodes, max_prios, chosen_cts, chosen_costs


    def apply_interventions(self, nodes: np.ndarray, contact_types: List[str], reduction: float, duration: int):
        """
        Apply a multiplicative reduction to nodes incoming/outgoing contact weight, then schedule this reduction to expire
        """

        if nodes.size == 0:
            return
        t_now = self.model.current_time
        expiration_time = t_now + duration

        #compute per-node factor where factor = 1 - reduction*compliance
        #TODO consider whether compliance should scale multiplicatively or determine in a binary if it happens at all
        compliances = self.model.compliances[nodes]
        factor = 1.0 - (reduction * compliances)
        #clip to avoid zeros
        clipped_factor = np.clip(factor, self.min_factor, 1.0).astype(np.float32)

        for ct in contact_types:
            #multiply outgoing and incoming multipliers 
            #currently assume LHD reduces both directions
            self.model.out_multiplier[ct][nodes] *= clipped_factor
            self.model.in_multiplier[ct][nodes] *= clipped_factor

            #schedule reversal, storing node/factor to divide later
            self.expiry.setdefault(expiration_time, []).append((nodes.copy(), ct, clipped_factor.copy()))


        #log actions taken
        self.action_log.append({
            'time': t_now, 'nodes_count': int(nodes.size),
            'contact_types': contact_types, 'reduction': reduction, 
            'duration': duration
        })

    def process_expirations(self, current_time):
        """
        Reverts multipliers for interventions expiring at current time
        """
        entries = self.expiry.pop(current_time, [])
        for nodes, ct, factor in entries:
            #divide for vectorized undo,
            safe_factor = np.maximum(factor, self.min_factor) #if zeroed before, don't div by zero now
            self.model.out_multiplier[ct][nodes] /= safe_factor
            self.model.in_multiplier[ct][nodes] /= safe_factor

    def step(self, current_time: int, recorder_snapshot: Dict[str, np.ndarray]):
        """
        One step for the LHD where it discovers events, builds candidates, selects calls, and applies interventions
        """

        #1 Expire old interventions
        self.process_expirations(current_time)

        #2 discover new events
        discovered_ind = self.discover_exposures(recorder_snapshot)

        #3 gather candidates through algorithms
        nodes, prios, cts, costs = self.gather_candidates(recorder_snapshot, discovered_ind)
        if nodes.size == 0:
            return

        #Greedy call prioritization (prio / cost)
        hours_available = self.daily_personhours 
        costs = np.maximum(costs, self.min_candidate_cost)

        value_per_hour = prios/costs
        order = np.argsort(-value_per_hour)

        #allocate calls until hours are exhausted
        hours_spent = 0.0
        selected_indices = []
        for ind in order:
            c = float(costs[ind])
            if hours_spent + c <= hours_available:
                hours_spent += c
                selected_indices.append(ind)
            else:
                continue

        selected_nodes = nodes[selected_indices]
        selected_cts = cts[selected_indices]
        selected_costs = costs[selected_indices]

        #group by contact type and apply interventions
        grouped = defaultdict(list)
        for n, ct in zip(selected_nodes, selected_cts):
            grouped[ct].append(int(n))
        
        #apply interventions and schedule to expire
        for ct, node_list in grouped.items():
            self.apply_interventions(np.asarray(node_list, dtype = np.int32),
            [ct], 
            reduction = self.default_int_reduction,
            duration = self.default_int_duration)

        #log action and resource consumption
        self.action_log.append({
            'time':int(current_time),
            'hours_available': hours_available,
            'hours_used': float(np.sum(selected_costs)),
            'n_called': int(len(selected_nodes))
        })


    
        
#Possible LHD strategies:
class EqualPriority(AlgorithmBase):
    def generate_candidates(self, recorder_snapshot, model, discovered_event_ind):
        #if no discovered events, no candidates
        if len(discovered_event_ind) == 0:
            return {
                    'nodes':np.empty(0, dtype = np.int32), 
                    'priority':np.empty(0, dtype = np.float32),
                    'contact_types': np.empty(0, dtype = object),
                    'costs': np.empty(0, dtype = np.float32)
                }

        #Initialize lists to return
        nodes_list, prios_list, cts_list, cost_list = [], [], [], []
        default_cost = model.params.get('lhd_default_call_duration', 0.083)

        #gather data for each contact from each event, assigning priorities, ct, and call durations
        for event in discovered_event_ind:
            s = int(recorder_snapshot['event_nodes_start'][event])
            L = int(recorder_snapshot['event_nodes_len'][event])
            #Skip if no nodes exposed in this event
            if L == 0: 
                continue

            
            nodes = recorder_snapshot['nodes'][s:s+L]
            nodes_list.append(nodes)
            prios_list.append(np.ones(nodes.shape[0], dtype = np.float32))
            ct_name = model.id_to_ct[int(recorder_snapshot['event_type'][event])]
            cts_list.append(np.array([ct_name]*nodes.shape[0], dtype = object))
            cost_list.append(np.full(nodes.shape[0], default_cost, dtype = np.float32))

        if not nodes_list:
            return {
                'nodes':np.empty(0, dtype = np.int32), 
                'priority':np.empty(0, dtype = np.float32),
                'contact_types': np.empty(0, dtype = object),
                'costs': np.empty(0, dtype = np.float32)
            }

        #aggregate lists across events and return
        return {
            'nodes': np.concatenate(nodes_list),
            'priority': np.concatenate(prios_list),
            'contact_types': np.concatenate(cts_list),
            'costs': np.concatenate(cost_list)
        }















        









#Test run on a really small population
if __name__ == "__main__":
    Keweenaw_contacts = pd.read_parquet("./data/Keweenaw/contacts.parquet")

    testModel = NetworkModel(contacts_df = Keweenaw_contacts)
    print("Running Model on Keweenaw County...")
    testModel.simulate()

    print("Plotting outcomes...")
    if testModel.n_runs < 10:
        for run in range(testModel.n_runs):
            testModel.epi_curve(run_number= run, suffix = f"run_{run+1}")
            testModel.cumulative_incidence_plot(run_number= run, 
                suffix = f"run_{run+1}", strata = "age")
            testModel.cumulative_incidence_plot(run_number= run, 
                suffix = f"run_{run+1}", strata = "sex")
    testModel.cumulative_incidence_spaghetti()
    results = testModel.epi_summary()

    






