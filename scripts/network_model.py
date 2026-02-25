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
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.animation as animation



from scripts.synth_data_processing import build_edge_list, build_individual_lookup

from new_scripts.recorder.recorder import ExposureEventRecorder
from new_scripts.lhd.lhd import LocalHealthDepartment


class ModelParameters(TypedDict):
    # Epi Params
    base_transmission_prob: float
    incubation_period: float
    infectious_period: float
    incubation_period_vax: float
    infectious_period_vax: float
    conferred_immunity_duration: Optional[float]
    lasting_partial_immunity: Optional[float]
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
    "base_transmission_prob": 0.25,
    "incubation_period": 10.5,
    "infectious_period": 5,
    "gamma_alpha": 20,
    "incubation_period_vax": 10.5,
    "infectious_period_vax": 5,
    "relative_infectiousness_vax": 0.05,
    "conferred_immunity_duration": None, #duration where one is completely immune post-infection
    "lasting_partial_immunity": None, #0-1, percent immune relative to naive
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
    "I0": None, #randomized if None
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
    edge_list: Optional[pd.DataFrame] = None,
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

        #edge_list can be passed by driver
        self.external_edge_list = edge_list 
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
        #If the driver has supplied an edge list, use it directly
        if getattr(self, "external_edge_list", None) is not None:
            self.edge_list = self.external_edge_list.copy()
        else:
            # existing behavior: either reuse saved file or call build_edge_list
            edgefile_path = os.path.join(
                os.getcwd(),
                "data",
                self.county,
                (self.params.get("run_name", "run") + "_edgeList.parquet")
            )
            # if a cached edge-list exists and overwrite_edge_list is False, read it
            if (not bool(self.params.get("overwrite_edge_list", False))) and os.path.isfile(edgefile_path):
                self.edge_list = pd.read_parquet(edgefile_path)
            else:
                # build edge list (this may save the file if build_edge_list honors save flag)
                self.edge_list = build_edge_list(
                    contacts_df = self.contacts_df,
                    params = self.params,
                    rng = self.rng,
                    save = bool(self.params.get("save_data_files", False)),
                    county = self.county
                )

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
        initial_infectious = self.params.get("I0", None)
        if not initial_infectious:
            initial_infectious = [self.rng.integers(0,self.N)]
            self.params["I0"] = initial_infectious
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

         #R -> S with waning immunity
        cid = self.params.get("conferred_immunity_duration", None)
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
                    lasting = self.params.get("lasting_partial_immunity", None)
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
                daily[0] = np.array(self.params.get("I0", []), dtype = np.int32)

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

    






