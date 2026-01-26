"""
network_model.py

Contains:
- NetworkModel, which runs a outbreak simulation on a network structure
- ExposureEventRecorder, which tracks exposure events
- ActionToken, a data tracking class for actions taken
- ActionBase, and all sub-class actions to be performed by LHD
- AlgorithmBase, and all sub-class algorithms that LHD uses to decide who to act upon
- LocalHealthDepartment - an actor that intervenes on the network
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.stats import truncnorm
from typing import TypedDict, List, Dict, Any, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass
import uuid
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.animation as animation


from scripts.synth_data_processing import build_edge_list, build_individual_lookup

class ModelParameters(TypedDict):
    # Epi Params
    base_transmission_prob: float
    incubation_period: float
    infectious_period: float
    incubation_period_vax: float
    infectious_period_vax: float
    conferred_immunity_duration: float
    gamma_alpha: float #alpha value for gamma distribution of inc/inf periods

    relative_infectiousness_vax: float
    vax_efficacy: float
    vax_uptake: float
    susceptibility_multiplier_under_five: float #increase in susceptibility if age <= 5
    susceptibility_multiplier_elderly: float #susc mult age >= 65

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
    I0: List[int] #1 random int if none
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
    "base_transmission_prob": .8,
    "incubation_period": 10.5,
    "infectious_period": 5,
    "gamma_alpha": 20,
    "incubation_period_vax": 10.5,
    "infectious_period_vax": 5,
    "relative_infectiousness_vax": 0.05,
    "vax_efficacy": .997,
    "vax_uptake": 0.85, 
    "susceptibility_multiplier_under_five": 1,
    "susceptibility_multiplier_elderly": 1,

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
    "n_runs": 50,
    "run_name" : "test_run",
    "overwrite_edge_list": True,
    "simulation_duration": 45,
    "I0": [1000], #randomized if None
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
    def __init__(
        self, 
    contacts_df, 
    params: ModelParameters = DefaultModelParams,
    *,
    rng = None,
    seed: Optional[int] = None,
    results_folder: Optional[str] = None,
    lhd_register_defaults: bool = True,
    lhd_algorithm_map: Optional[Dict[str, object]] = None,
    lhd_action_factory_map: Optional[Dict[str, Callable[..., Any]]] = None):
        """
        Unpack parameters, set-up storage variables, and initialize model

        kwargs:
        - rng: optional np.random.Generator for randomness
        - seed: optional int seed if rng is None
        - results_folder: optional path to store outputs overriding run_name default
        - lhd_register_defaults: if True, LHD calls random exposed individuals
        - lhd_algorithm_map / lhd_action_factory_map: maps to pass LHD 
        """
        #Unpack params and model settings
        self.params = dict(params)
        self.contacts_df = contacts_df
        self.N = int(self.contacts_df.shape[0])
        self.Tmax = int(self.params.get("simulation_duration", 100))

        #choose RNG: explicit rng > seed arg > params['seed] 
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(int(seed))
        else:
            seed_from_params = self.params["seed"]
            self.rng = np.random.default_rng(int(seed_from_params))

        #run metadata:
        self.county = self.params.get("county", "")
        self.n_runs = int(self.params.get("n_runs", 1))

        #One-time computation of network structures
        self._compute_network_structures()

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
            discovery_prob = self.params.get("lhd_discovery_prob"),
            employees = self.params.get("lhd_employees"),
            workday_hrs = self.params.get("lhd_workday_hrs"),
            register_defaults = lhd_register_defaults,
            algorithm_map = lhd_algorithm_map,
            action_factory_map = lhd_action_factory_map
        )

        #set-up result folder if not overriden by driver
        
        self.results_folder = results_folder if results_folder is not None else os.path.join(os.getcwd(), "results", self.params.get("run_name", "run"))
        if self.params["save_data_files"]:
                os.makedirs(self.results_folder, exist_ok = True)

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
        self.is_vaccinated = self.rng.random(self.N) < self.params["vax_uptake"]

        #Set individual compliances array
        avg_compliance = np.clip(self.params["mean_compliance"], 0, 1) #0-1
        sd = 0.15 #sd for bounded normal distribution
        #calculate truncnorm params
        a = (0 - avg_compliance) / sd
        b = (1 - avg_compliance) / sd
        self.compliances = truncnorm.rvs(a,b,
        loc = avg_compliance, scale = sd,
        size = self.N, random_state = self.rng)

        

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

        #Initialize multipliers for LHD interaction
        self.in_multiplier = {
            ct: np.ones(self.N, dtype=np.float32) for ct in self.contact_types
            }  
        self.out_multiplier = {
            ct: np.ones(self.N, dtype=np.float32) for ct in self.contact_types
            }  

    def _initialize_states(self):
        """
        Set up new SEIR arrays and seed initial infectious individuals
        """
        self.current_time = 0
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

        #pick random I0 if none provided
        initial_infectious = self.params.get("I0", [self.rng.randint(0,self.N)])
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

        if hasattr(self, "lhd"):
            self.lhd.reset_for_run()
       
        
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
        susc_mult_elderly = self.params["susceptibility_multiplier_elderly"]
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
            self._initialize_states()
            #store vaccination status for run
            self.all_vax_status[run] = self.is_vaccinated.copy()

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
                
            #save per run data
            self.all_states_over_time[run] = [states.copy() for states in self.states_over_time]
            self.all_new_exposures[run] = [ne.copy() if hasattr(ne, "copy") else list(ne) for ne in self.new_exposures]
            self.all_stochastic_dieout[run] = self.stochastic_dieout
            self.all_end_days[run] = self.simulation_end_day
            self.all_lhd_action_logs[run] = list(self.lhd.action_log)


    def epi_outcomes(self) -> pd.DataFrame:
        """
    Calculates epidemic outcomes with a row for each run, and columns:
    - total_infections
    attack_rate
    -attack_rate_by_age_{bins}
    - attack_rate_vaccinated, attack_rate_unvaccinated
    - peak_incidence 
    - peak_prevalence 
    - time_of_peak_infection
    -secondary attack rate by contact type (sar_{ct})
    - effective_reproduction_number (mean R)
    - number_of_calls
    - people_called
    avg_days_quarantine_imposed
    - stochastic_dieout
    - epidemic_duration
        """
        #supress numpy division warnings
        #TODO look into whether this affects validity
        warnings.filterwarnings("ignore", category = RuntimeWarning, module = "numpy.*")


        
        N = self.N

        #require simulations have been run
        if any(x is None for x in self.all_new_exposures):
            raise ValueError("Model runs not completed; run simulate() before computing outcomes")

        summary = []

        #age bins 
        age_bins = [0, 6, 19, 35, 65,200]
        age_labels = ["0-5", "6-18", "19-34", "35-64", "65+"]

        for run in range(self.n_runs):
            exposures = self.all_new_exposures[run]
            states_over_time = self.all_states_over_time[run]
            stochastic_dieout = bool(self.all_stochastic_dieout[run])
            epidemic_time = int(self.all_end_days[run])
            snapshots = self.exposure_event_log[run]

            #determine ever exposed
            ever_exposed = np.zeros(N, dtype=bool)
            for exp in exposures:
                if len(exp) == 0:
                    continue
                ever_exposed[np.array(exp, dtype=int)] = True
            total_infections = int(ever_exposed.sum()) #total_infections does not include imported infections

            #build at-risk mask, anyone who ever had infectious neighbor
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
            infected_at_risk = int((ever_exposed & at_risk).sum())
            attack_rate_at_risk = (infected_at_risk / float(n_at_risk)) if n_at_risk > 0 else np.nan

            #Age-stratified attack rates among individuals ever at-risk
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
                        infected_risk_members = int(ever_exposed[members][at_risk_mask].sum())
                        attack_by_age[label] = infected_risk_members / float(n_risk_members)

            #Vaccination stratified attack rates among at-risk
            #For vaccinated AT START of outbreak
            vax_status = np.asarray(self.all_vax_status[run], dtype = bool)
            ar_vax_mask = at_risk & vax_status
            ar_unvax_mask = at_risk & (~vax_status)
            n_at_risk_vax = int(ar_vax_mask.sum())
            n_at_risk_unvax = int(ar_unvax_mask.sum())
            attack_rate_vax = (ever_exposed[ar_vax_mask].sum() / n_at_risk_vax) if n_at_risk_vax > 9 else np.nan
            attack_rate_unvax = (ever_exposed[ar_unvax_mask].sum() / n_at_risk_unvax) if n_at_risk_unvax > 9 else np.nan

            #Peak incidence, with day 0 including imported inf.
            daily = [np.array(e, dtype=int) for e in exposures]
            if len(daily) > 0 and daily[0].size == 0:
                daily[0] = np.array(self.params.get("I0", []), dtype=int)
            daily_counts = [int(arr.size) for arr in daily] if len(daily) else []
            peak_incidence = int(max(daily_counts)) if daily_counts else 0
            time_of_peak_incidence = int(np.argmax(daily_counts)) if daily_counts else None

            #Peak prevalence + time
            peak_infections = 0
            time_of_peak = None
            for t, state in enumerate(states_over_time):
                I_nodes = np.array(state[2], dtype = int)
                nI = int(I_nodes.size)
                if nI > peak_infections:
                    peak_infections = nI
                    peak_prev = (nI / float(N))
                    time_of_peak = int(t)

            #Cumulative incidence overall and stratified
            cumulative_incidence_by_age = {}
            for label in age_labels:
                members = age_members[label]
                if members.size == 0:
                    cumulative_incidence_by_age[label] = np.nan
                else:
                    cumulative_incidence_by_age[label] = float(ever_exposed[members].sum()) / members.size

            if vax_status is not None:
                n_vax = int(vax_status.sum())
                n_unvax = int((~vax_status).sum())
                cumulative_incidence_vax = (float(ever_exposed[vax_status].sum()) / n_vax) if n_vax > 0 else np.nan
                cumulative_incidence_unvax = (float(ever_exposed[~vax_status].sum()) / n_unvax) if n_unvax > 0 else np.nan
            else:
                cumulative_incidence_vax = np.nan
                cumulative_incidence_unvax = np.nan


            #Secondary Attack Rate by contact type and Effective R0
            #SAR - proportion of contacts infected
            #Mean, sd, and median effective R0
            total_exposed_by_ct = {ct: 0 for ct in self.contact_types}
            total_infected_by_ct = {ct: 0 for ct in self.contact_types}
            assigned = np.zeros(N, dtype=bool)  #attribute infection to src
            source_secondary = defaultdict(int)
            unique_sources = set()

            for snap in snapshots:
                n_events = int(snap["event_time"].shape[0])
                if n_events == 0:
                    continue
                for ei in range(n_events):
                    src = int(snap["event_source"][ei])
                    unique_sources.add(src)
                    ct_id = int(snap["event_type"][ei])
                    ct_name = self.id_to_ct.get(ct_id, None)
                    s = int(snap["event_nodes_start"][ei])
                    L = int(snap["event_nodes_len"][ei])
                    if L == 0:
                        continue
                    nodes = snap["nodes"][s : s + L]
                    infs = snap["infections"][s : s + L]
                    ct_name = ct_name if ct_name is not None else "unknown"
                    total_exposed_by_ct[ct_name] = total_exposed_by_ct.get(ct_name, 0) + int(L)
                    for j in range(L):
                        node = int(nodes[j])
                        if bool(infs[j]) and not assigned[node]:
                            assigned[node] = True
                            total_infected_by_ct[ct_name] = total_infected_by_ct.get(ct_name, 0) + 1
                            source_secondary[src] += 1

            sar_by_ct = {}
            for ct in self.contact_types:
                exposed_ct = total_exposed_by_ct.get(ct, 0)
                infected_ct = total_infected_by_ct.get(ct, 0)
                sar_by_ct[ct] = (infected_ct / float(exposed_ct)) if exposed_ct > 0 else np.nan
            
            total_secondary = sum(source_secondary.values())
            n_sources = len(unique_sources)
            eff_R = (total_secondary / float(n_sources)) if n_sources > 0 else np.nan
            eff_R_std = np.std(list(source_secondary.values())) if n_sources > 0 else np.nan
            eff_R_median = np.median(list(source_secondary.values())) if  n_sources > 0 else np.nan

            #LHD Metrics - quarantine and calls
            lhd_log = self.all_lhd_action_logs[run]
            call_entries = [e for e in lhd_log if e.get("action_type") == "call"] 
            number_of_call_actions = len(call_entries)
            people_called = sum(int(e.get("nodes_count", 0)) for e in call_entries)
            if people_called > 0:
                total_person_days = sum(int(e.get("duration", 0)) * int(e.get("nodes_count", 0)) for e in call_entries)
                avg_days_quarantine = total_person_days / float(people_called)

            else:
                avg_days_quarantine = np.nan

                # Assemble row
            row = {
                "run_number": int(run),
                "total_infections": int(total_infections),
                "n_at_risk": int(n_at_risk),
                "infected_at_risk": int(infected_at_risk),
                "attack_rate_at_risk": float(attack_rate_at_risk) if not np.isnan(attack_rate_at_risk) else np.nan,
                "peak_incidence": int(peak_incidence),
                "time_of_peak_incidence": time_of_peak_incidence,
                "peak_infections": int(peak_infections),
                "time_of_peak_infections": time_of_peak,
                "peak_prevalence_overall": float(peak_prev),
                "attack_rate_vaccinated_at_risk": float(attack_rate_vax) if not np.isnan(attack_rate_vax) else np.nan,
                "attack_rate_unvaccinated_at_risk": float(attack_rate_unvax) if not np.isnan(attack_rate_unvax) else np.nan,
                "effective_reproduction_number": float(eff_R) if not np.isnan(eff_R) else np.nan,
                "effective_reproduction_number_std": float(eff_R_std) if not np.isnan(eff_R_std) else np.nan,
                "effective_reproduction_number_median": float(eff_R_median) if not np.isnan(eff_R_median) else np.nan,
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

            # vaccination cumulative incidence
            row["cumulative_incidence_vaccinated"] = cumulative_incidence_vax
            row["cumulative_incidence_unvaccinated"] = cumulative_incidence_unvax

            # attach SAR per contact type and raw exposed/infected counts
            for ct in self.contact_types:
                row[f"sar_{ct}"] = sar_by_ct.get(ct, np.nan)
                row[f"exposed_{ct}"] = int(total_exposed_by_ct.get(ct, 0))
                row[f"infected_{ct}"] = int(total_infected_by_ct.get(ct, 0))

            summary.append(row)


        #re-enable warnings
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


####### Actions - LHD Interacting with the model

#Token produced by action.apply and stored in expiry schedule
@dataclass
class ActionToken:
    action_id: str
    action_type: str #e.g. "calling", "quarantining", etc.
    contact_type: Optional[str] #e.g. 'cas', 'hh' or None
    nodes: np.ndarray #integer array of nodes affected by action
    factor: Optional[np.ndarray] #factor applied to contact reductions or None
    reversible: bool = True #if action can be undone
    meta: Optional[Dict[str, Any]] = None #any relevant metadata

class ActionBase:
    """
    Abstract action. Subclasses implement apply() to alter the model, returning ActionToken(s)
    """
    def __init__(self, action_type: str, duration:int = 0, kind: Optional[str] = None):
        self.id = uuid.uuid4().hex
        self.action_type = action_type
        self.duration = int(duration)
        self.kind = kind or action_type #more descriptive, sub-type label
        self.reversible = True #subclasses can override

    def apply(self, model: NetworkModel, current_time: int) -> List[ActionToken]:
        """
        Apply action to the model, return a list of ActionTokens describing state changes
        """
        raise NotImplementedError

    def revert_token(self, model: NetworkModel, token: ActionToken) -> None:
        """
        Reverts a previously-applied token for a reversible action
        """

#Possible Actions
class CallIndividualsAction(ActionBase):
    """
    Calls individuals to inform them of their exposure, applying a multiplicative reduction to their contact weight for school, workplace, and casual contacts

    nodes: list of model node indices
    contact_types: list or iterable of contact types ('cas', 'sch', etc.)
    reduction: fraction (0-1) representing reduction (to be mult by compliance)
    duration: days individual asked to reduce contacts
    call_cost: hours spent per call
    min_factor: minimum allowed reduction (just to avoid divzero errors)
    """
    def __init__(self, nodes: np.ndarray, contact_types: List[str], reduction: float, duration: int, call_cost: float, min_factor: float = 1e-6):
        super().__init__(action_type = "call", duration = duration, kind = "call_individuals")
        self.nodes = np.asarray(nodes, dtype = np.int32)
        self.contact_types = list(contact_types)
        self.reduction = float(reduction)
        self.call_cost = float(call_cost)
        self.min_factor = float(min_factor)
        self.reversible = True

    def apply(self, model: NetworkModel, current_time: int) -> List[ActionToken]:
        #per-node factor, scaled by compliance
        comps = model.compliances[self.nodes] 
        raw = 1.0 - (self.reduction * comps)
        factor = np.clip(raw, self.min_factor, 1.0).astype(np.float32)

        tokens: List[ActionToken] = []
        for ct in self.contact_types:
            #apply reductions
            model.out_multiplier[ct][self.nodes] *= factor
            model.in_multiplier[ct][self.nodes] *= factor

            token = ActionToken(
                action_id = self.id,
                action_type = self.action_type,
                contact_type = ct,
                nodes = self.nodes.copy(),
                factor = factor.copy(),
                reversible = True,
                meta = {'call_cost': self.call_cost}
            )
            tokens.append(token)
        
        return tokens

    def revert_token(self, model: NetworkModel, token: ActionToken) -> None:
        #undo the multiplicative factor by dividing out 
        if token.factor is None:
            return
        f = token.factor
        #safeguard against zeros
        f_safe = np.maximum(f, self.min_factor)
        model.out_multiplier[token.contact_type][token.nodes] /= f_safe
        model.in_multiplier[token.contact_type][token.nodes] /= f_safe




####### Algorithms - LHD taking info from the outbreak and making decisions
class AlgorithmBase:
    """
    An interface for designing calling algorithms for the local health department.
    
Algorithms have one method, generate_candidates, which takes:
    recorder_snapshot: 
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

## Possible Algorithms
class EqualPriority(AlgorithmBase):
    """
    Returns each susceptible individual as equal priority candidates 
    """
    def generate_candidates(self, recorder_snapshot:Dict[str, np.ndarray], model, discovered_event_ind) -> Dict[str, Any]:
        #if no events discovered, return nobody
        if len(discovered_event_ind) == 0:
            return {
                'nodes': np.empty(0, dtype = np.int32),
                'priority': np.empty(0, dtype = np.float32),
                'contact_types': np.empty(0, dtype = object),
                'costs': np.empty(0, dtype = np.float32),
                'params': []
            }

        nodes_list = []
        prios_list = []
        cts_list = []
        costs_list = []

        default_cost = float(model.params.get('lhd_default_call_duration', 0.083))
        for event in discovered_event_ind:
            s = int(recorder_snapshot['event_nodes_start'][event])
            L = int(recorder_snapshot["event_nodes_len"][event])
            if L == 0:
                continue

            nodes = np.asarray(recorder_snapshot['nodes'][s:s+L], dtype = np.int32)
            nodes_list.append(nodes)
            prios_list.append(np.ones(nodes.shape[0], dtype = np.float32))
            ct_name = model.id_to_ct[int(recorder_snapshot['event_type'][event])] if 'event_type' in recorder_snapshot else None
            cts_list.append(np.array([ct_name]*nodes.shape[0], dtype = object))
            costs_list.append(np.full(nodes.shape[0], default_cost, dtype = np.float32))

        if not nodes_list:
            return {
                'nodes': np.empty(0, dtype = np.int32),
                'priority': np.empty(0, dtype = np.float32),
                'contact_types': np.empty(0, dtype = object),
                'costs': np.empty(0, dtype = np.float32),
                'params': []
            }
        return { 
            'nodes': np.concatenate(nodes_list),
            'priority': np.concatenate(prios_list),
            'contact_types': np.concatenate(cts_list),
            'costs': np.concatenate(costs_list),
            'params': None
        }

class RandomPriority(AlgorithmBase):
    """
    Assigns each exposed candidate a random priority in [0-1)
    """
    def generate_candidates(self, recorder_snapshot: Dict[str, np.ndarray], model, discovered_event_ind) -> Dict[str, Any]:
        #if no events discovered, return nobody
        if len(discovered_event_ind) == 0:
            return {
                'nodes': np.empty(0, dtype = np.int32),
                'priority': np.empty(0, dtype = np.float32),
                'contact_types': np.empty(0, dtype = object),
                'costs': np.empty(0, dtype = np.float32),
                'params': []
            }

        nodes_list = []
        prios_list = []
        cts_list = []
        costs_list = []
        default_cost = float(model.params.get('lhd_default_call_duration', 0.083))
        for event in discovered_event_ind:
            s = int(recorder_snapshot['event_nodes_start'][event])
            L = int(recorder_snapshot["event_nodes_len"][event])
            if L == 0:
                continue

            nodes = np.asarray(recorder_snapshot['nodes'][s:s+L], dtype = np.int32)
            nodes_list.append(nodes)

            #random priority using model.rng
            prios_list.append(model.rng.random(size = nodes.shape[0]).astype(np.float32))
            ct_name = model.id_to_ct[int(recorder_snapshot['event_type'][event])] if 'event_type' in recorder_snapshot else None
            cts_list.append(np.array([ct_name] * nodes.shape[0], dtype = object))
            costs_list.append(np.full(nodes.shape[0], default_cost, dtype = np.float32))

        if not nodes_list:
            return {
                'nodes': np.empty(0, dtype = np.int32),
                'priority': np.empty(0, dtype = np.float32),
                'contact_types': np.empty(0, dtype = object),
                'costs': np.empty(0, dtype = np.float32),
                'params': []
            }

        return { 
            'nodes': np.concatenate(nodes_list),
            'priority': np.concatenate(prios_list),
            'contact_types': np.concatenate(cts_list),
            'costs': np.concatenate(costs_list),
            'params': None
            }

class PrioritizeElders(AlgorithmBase):
    """
    Priority boost for exposed nodes aged 65+. Priority = base + boost
    Assume elders take longer to speak with on the phone 
    """
    def __init__(self, base_priority: float = 1.0, elder_boost: float = 4.0, elder_cost: float = 0.1 ):
        self.base_priority = float(base_priority)
        self.elder_boost = float(elder_boost)
        self.elder_cost = float(elder_cost)

    def generate_candidates(self, recorder_snapshot: Dict[str, np.ndarray], model, discovered_event_ind) -> Dict[str, Any]:
        #if no events discovered, return nobody
        if len(discovered_event_ind) == 0:
            return {
                'nodes': np.empty(0, dtype = np.int32),
                'priority': np.empty(0, dtype = np.float32),
                'contact_types': np.empty(0, dtype = object),
                'costs': np.empty(0, dtype = np.float32),
                'params': []
            }

        nodes_list = []
        prios_list = []
        cts_list = []
        costs_list = []

        default_cost = float(model.params.get('lhd_default_call_duration', 0.083))
        for event in discovered_event_ind:
            s = int(recorder_snapshot['event_nodes_start'][event])
            L = int(recorder_snapshot["event_nodes_len"][event])
            if L == 0:
                continue

            nodes = np.asarray(recorder_snapshot['nodes'][s:s+L], dtype = np.int32)
            nodes_list.append(nodes)

            ages_targets = model.ages[nodes]
            pr = np.where(ages_targets >= 65, self.base_priority + self.elder_boost, self.base_priority).astype(np.float32)
            prios_list.append(pr)

            ct_name = model.id_to_ct[int(recorder_snapshot['event_type'][event])] if 'event_type' in recorder_snapshot else None
            cts_list.append(np.array([ct_name] * nodes.shape[0], dtype = object))

            costs = np.where(ages_targets >= 65, self.elder_cost, default_cost).astype(np.float32)
            costs_list.append(costs)

        if not nodes_list:
            return {
                'nodes': np.empty(0, dtype = np.int32),
                'priority': np.empty(0, dtype = np.float32),
                'contact_types': np.empty(0, dtype = object),
                'costs': np.empty(0, dtype = np.float32),
                'params': []
            }

        return { 
            'nodes': np.concatenate(nodes_list),
            'priority': np.concatenate(prios_list),
            'contact_types': np.concatenate(cts_list),
            'costs': np.concatenate(costs_list),
            'params': None
            }

####### LHD Class - Local Health Department agent that interacts with outbreak
class LocalHealthDepartment:
    def __init__(
        self, 
        model: NetworkModel, 
        rng = None, 
        discovery_prob: float = None,  
        employees: int = None, 
        workday_hrs: float = None,
        register_defaults: bool = True,
        algorithm_map: Optional[Dict[str, object]] = None,
        action_factory_map: Optional[Dict[str, Callable[..., ActionBase]]] = None
    ):
    #LHD settings
        self.model = model
        self.rng = rng if rng is not None else getattr(model, "rng", np.random.default_rng())
        self.discovery_prob = discovery_prob if discovery_prob is not None else self.model.params["lhd_discovery_prob"]

    #LHD Capacity
        self.employees = employees if employees is not None else self.model.params["lhd_employees"]
        self.hours_per_employee = float(workday_hrs) if workday_hrs is not None else self.model.params["lhd_workday_hrs"]
        self.daily_personhours = float(self.employees * self.hours_per_employee)

    #Algorithm -> algorithm instance
        self.algorithms: Dict[str, AlgorithmBase] = {}
    #action factories: action_type -> callable to return ActionBase
        self.action_factories: Dict[str, Callable[..., ActionBase]] = {}


    #trackers for action objects and token counts
        self.expiry: Dict[int, List[ActionToken]] = {}
        self.action_log: List[Dict[str, Any]] = []
        # action id -> action instance
        self._active_actions: Dict[str, ActionBase] = {}
        #action id -> number outstanding tokens
        self._action_token_counts: Dict[str, int] = {}

        self.min_factor = 1e-6 #to prevent div 0 errors
        self.min_candidate_cost = 1e-4

    #Default action params
        self.default_int_reduction = model.params.get("lhd_default_int_reduction", 0.8)
        self.default_int_duration = model.params.get("lhd_default_int_duration", 7)
        self.default_call_cost = model.params.get("lhd_default_call_duration", 0.083)


        #Register default actions if requested:
        if register_defaults:
            self.register_algorithm('call', RandomPriority())

            def default_call_factory(nodes, contact_type, prio, cost, params = None):
                return CallIndividualsAction(
                nodes = nodes,
            contact_types = [contact_type] if contact_type is not None else
            ['cas', 'sch', 'wp'],
            reduction = params.get('reduction', self.default_int_reduction) if params else self.default_int_reduction,
            duration = int(params.get('duration', self.default_int_duration)) if params else self.default_int_duration,
            call_cost = float(cost) if cost is not None else self.default_call_cost,
            min_factor = self.min_factor
        )
            self.register_action_factory('call', default_call_factory)

        #Register mappings provided by call
        if algorithm_map:
            for atype, alg in algorithm_map.items():
                self.register_algorithm(atype, alg, overwrite = True)
        if action_factory_map:
            for atype, factory in action_factory_map.items():
                self.register_action_factory(atype, factory, overwrite = True)
            


    ##registration helpers
    # map algorithms action_type -> algorithm
    def register_algorithm(self, action_type: str, algorithm: AlgorithmBase, overwrite: bool = False) -> None:
        """
        Assign each action with an algorithm that is used to decide who that action should be done to. 
        """
        if action_type in self.algorithms and not overwrite:
            raise ValueError(f"Algorithm already registered for action '{action_type}'. Only one allowed.")
        if action_type in self.algorithms and overwrite:
            warnings.warn(f"Overwriting existing algorithm for action '{action_type}'")
        self.algorithms[action_type] = algorithm

    #map action_type -> factory to create action objects
    #expects 
    def register_action_factory(self, action_type: str, factory: Callable[..., ActionBase], overwrite: bool = False) -> None:
        #factory signature expected: (nodes, contact_type, prio, cost, params) -> ActionBase
        if action_type in self.action_factories and not overwrite:
            raise ValueError(f"Action factory already registered for action '{action_type}'. Only one allowed.")
        if action_type in self.action_factories and overwrite:
            warnings.warn(f"Overwriting existing action factory for action '{action_type}'")
        self.action_factories[action_type] = factory

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
        Returns flattened candidate arrays with parallel arrays:
        action_types (st array), nodes (int), priority (float), 
        contact_types (object), costs (float)
        """
        
        #lists to fill
        action_types_list = []
        nodes_list = []
        prios_list = []
        cts_list = []
        costs_list = []
        params_list = []

        #use algorithms to generate action candidates
        for action_type, algo in self.algorithms.items():
            out = algo.generate_candidates(recorder_snapshot, self.model, discovered_event_ind) or {}
            nodes = np.asarray(out.get('nodes', np.empty(0, dtype=np.int32)), dtype=np.int32)
            prios = np.asarray(out.get('priority', np.ones(nodes.shape[0], dtype=np.float32)), dtype=np.float32)

            #raw cts can be a single str or array of size len(nodes)
            raw_cts = out.get('contact_types', None)
            if raw_cts is None:
                cts = np.array([None] * nodes.shape[0], dtype = object)
            elif isinstance(raw_cts, (str, bytes)):
                cts =  np.array([raw_cts] * nodes.shape[0], dtype = object)
            else:
                cts = np.asarray(raw_cts, dtype = object)
                if cts.shape[0] != nodes.shape[0]:
                    if cts.size == 1:
                        cts = np.repeat(cts[0], nodes.shape[0]).astype(object)
                    else:
                        raise ValueError("contact_types must be scalar or match nodes length")
            
            #costs must be a single float or array of size len(nodes)
            costs = out.get('costs', None)
            if costs is None:
                costs = np.array([None] * nodes.shape[0], dtype = np.float32)
            else:
                costs = np.asarray(costs, dtype = object)
                if costs.shape[0] != nodes.shape[0]:
                    if costs.size == 1:
                        costs = np.repeat(costs[0], nodes.shape[0]).astype(object)
                    else:
                        raise ValueError("costs must be scalar or match nodes length")

            params = out.get('params', None)
            #for each action type, deduplicate candidates
            if nodes.size > 0:
                unique_nodes, inverse = np.unique(nodes, return_inverse = True)
                best_prios = np.full(unique_nodes.shape[0], -np.inf, dtype = np.float32)
                best_cts = np.empty(unique_nodes.shape[0], dtype = object)
                best_costs = np.full(unique_nodes.shape[0], np.inf, dtype = np.float32)
                best_params = [None] * unique_nodes.shape[0]

                for occ in range(nodes.shape[0]):
                    uid = inverse[occ]
                    p = float(prios[occ])
                    c = float(costs[occ])
                    #keep highest priority, or lowest cost if tie
                    if p > best_prios[uid] or (p==best_prios[uid] and c < best_costs[uid]):
                        best_prios[uid] = p
                        best_cts[uid] = cts[occ]
                        best_costs[uid] = c
                        if params is not None:
                            #params can be per-occurrance or scalar
                            try:
                                best_params[uid] = params[occ]
                            except Exception:
                                best_params[uid] = params
                    
                #append best occurrences to global lists with action_type label
                for i, u in enumerate(unique_nodes):
                    action_types_list.append(action_type)
                    nodes_list.append(int(u))
                    prios_list.append(best_prios[i])
                    cts_list.append(best_cts[i])
                    costs_list.append(best_costs[i])
                    params_list.append(best_params[i])

        #if no nodes to gather, return empties
        if not nodes_list:
            return(
                np.empty(0, dtype = object),
                np.empty(0, dtype = np.int32), 
                np.empty(0, dtype = np.float32),
                np.empty(0, dtype = object),
                np.empty(0, dtype = np.float32), 
                []
            )

        #else, gather results and return
        action_types_arr = np.array(action_types_list, dtype = object)
        nodes_arr = np.array(nodes_list, dtype = np.int32)
        prios_arr = np.array(prios_list, dtype = np.float32)
        contact_types_arr = np.array(cts_list, dtype = object)
        costs_arr = np.array(costs_list, dtype = np.float32)
        params_arr = params_list

        return action_types_arr, nodes_arr, prios_arr, contact_types_arr, costs_arr, params_arr

    def schedule_action(self, action: ActionBase, current_time: int, resource_cost: float):
        """
        Apply action and schedule tokens for expiry if duration > 0.
        Registers Action instance for process_expirations to call reversion
        """
        #Apply all actions, get a list of actions performed
        tokens = action.apply(self.model, current_time)

        for t in tokens:
            if getattr(t, "action_id", None) != action.id:
                raise ValueError(f"Token.action_id {getattr(t, 'action_id', None)} does not match action.id {action.id}")

        #partition to reversible and nonreversible
        reversible_tokens = [t for t in tokens if getattr(t, "reversible", True)]
        nonreversible_tokens = [t for t in tokens if not getattr(t, "reversible", True)]

        #if there are reversible tokens and duration, schedule


        #register reversible tokens only, schedule expiry
        if reversible_tokens and (action.duration and action.duration > 0):
            expiry_time = int(current_time + action.duration)
            self.expiry.setdefault(expiry_time, []).extend(reversible_tokens)

            self._active_actions[action.id] = action
            self._action_token_counts[action.id] = self._action_token_counts.get(action.id, 0) + len(reversible_tokens)

        #if duration > 0, but not reversible, warning as nothing to revert
        if action.duration and action.duration > 0 and not reversible_tokens:
            warnings.warn(f"Action {action.id} has duration but produced no reversible tokens, will not be automatically reverted")

        #log metadata
        self.action_log.append({
            'time': int(current_time),
            'action_id': action.id,
            'action_type': action.action_type,
            'kind': getattr(action, "kind", action.action_type),
            'nodes_count': int(getattr(action, "nodes", np.empty(0)).size),
            'hours_used': float(resource_cost),
            'duration': int(action.duration),
            'reversible_tokens': len(reversible_tokens),
            'nonreversible_tokens': len(nonreversible_tokens)
        })

    def process_expirations(self, current_time):
        """
        Revert ActionTokens scheduled for current_time using action reversion methods, and remove from active actions
        """
        tokens_due = self.expiry.pop(int(current_time), [])
        for token in tokens_due:
            action = self._active_actions.get(token.action_id)
            if action is not None:
                #delegate reversion to action's method
                try:
                    action.revert_token(self.model, token)
                except Exception as exc:
                    warnings.warn(f"Action.revert_token failed for action {token.action_id}: {exc}")

                #decrement token counters and clean mappings 
                self._action_token_counts[token.action_id] -= 1
                if self._action_token_counts[token.action_id] <= 0:
                    #if no more tokens on this action, remove from active 
                    del self._action_token_counts[token.action_id]
                    del self._active_actions[token.action_id]
            else:
                warnings.warn(f"No registered action object for token.action_id {token.action_id}")

    def step(self, current_time: int, recorder_snapshot: Dict[str, np.ndarray]):
        """
        One step for the LHD where it discovers events, builds candidates, selects calls, and applies interventions
        """

        #1 expire old interventions
        self.process_expirations(current_time)

        #2 discover new events
        discovered_event_ind = self.discover_exposures(recorder_snapshot)

        #3 gather action candidates through algorithms
        (action_types_arr, nodes_arr, prios_arr, contact_types_arr, costs_arr, params_arr) = self.gather_candidates(recorder_snapshot, discovered_event_ind)
        if nodes_arr.size == 0:
            return

        #4 select actions maximizing value/hour (prio / cost)
        costs_arr = np.maximum(costs_arr, self.min_candidate_cost)
        value_per_hour = prios_arr/costs_arr
        order = np.argsort(-value_per_hour)

        #allocate actions by value until hours are exhausted 
        hours_available = float(self.daily_personhours)
        hours_spent = 0.0
        selected_indices = []
        for ind in order:
            c = float(costs_arr[ind])
            if hours_spent + c <= hours_available:
                hours_spent += c
                selected_indices.append(ind)
            else:
                continue

        #5 group selected actions by action_type, contact_type and schedule
        grouped = defaultdict(list)
        grouped_costs = defaultdict(float)
        grouped_params = defaultdict(list)
        for ind in selected_indices:
            atype = action_types_arr[ind]
            ctype = contact_types_arr[ind]
            key = (atype, ctype)
            grouped[key].append(int(nodes_arr[ind]))
            grouped_costs[key] += float(costs_arr[ind])
            grouped_params[key].append(params_arr[ind] if params_arr is not None else None)

        #create an action for each group and schedule
        for (atype, ctype), nodes in grouped.items():
            factory = self.action_factories.get(atype)
            if factory is None:
                #skip if no action factory registered
                continue
            #choose params and pass merged dict or None
            params_list_group = grouped_params[(atype, ctype)]
            merged_params = None
            for p in params_list_group:
                if isinstance(p, dict):
                    merged_params = merged_params or {}
                    merged_params.update(p)

            #create action instance 
            #use sum of costs or average priority for group
            group_cost = grouped_costs[(atype, ctype)]
            group_prio = float(np.mean([prios_arr[ind] for ind in selected_indices if action_types_arr[ind] == atype and contact_types_arr[ind] == ctype]))
            action = factory(np.asarray(nodes, dtype = np.int32), ctype, group_prio, group_cost, merged_params)
            self.schedule_action(action, current_time, resource_cost = group_cost)

        return
    def reset_for_run(self):
        """
        Reset LHD state for new model run 
        """
        self.expiry = {}
        self.action_log = []
        self._active_actions = {}
        self._action_token_counts = {}


#Test run on population N = 2186
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
            testModel.cumulative_incidence_plot(run_number= run, 
                suffix = f"run_{run+1}", strata = "vaccination")
    testModel.cumulative_incidence_spaghetti()
    results = testModel.epi_outcomes()

    






