import os
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
from scipy.sparse import csr_matrix
from typing import TypedDict, List
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
    vax_uptake: float
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

    # Simulation Settings
    run_name: str #prefix for model run
    overwrite_edge_list: bool #Try to reload previously generated edge list for this run to save time
    simulation_duration: int  # days
    dt: float  # steps per day
    I0: List[int]
    seed: int
    county: str #county to run on
    state: str #state to run on
    save_plots: bool
    save_data_files: bool
    make_movie: bool
    display_plots: bool



DefaultModelParams: ModelParameters = {
    "base_transmission_prob": 0.8,
    "incubation_period": 10.5,
    "infectious_period": 5,
    "gamma_alpha": 20,
    "incubation_period_vax": 10.5,
    "infectious_period_vax": 5,
    "relative_infectiousness_vax": 0.05,
    "vax_efficacy": 0,
    "vax_uptake": 0.85, 
    "susceptibility_multiplier_under_five": 2.0,

    "wp_contacts": 10,
    "sch_contacts": 10,
    "gq_contacts": 10,
    "cas_contacts": 5,
    "hh_weight": 1,
    "wp_weight": .5,
    "sch_weight": .6,
    "gq_weight": .3,
    "cas_weight": .1,

    "run_name" : "test_run",
    "overwrite_edge_list": True,
    "simulation_duration": 45,
    "dt": 1,
    "I0": [906],
    "seed": 2026,
    "county": "Keweenaw",
    "state": "Michigan",
    "save_plots": True,
    "save_data_files": True,
    "make_movie": False,
    "display_plots": True

}

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

        if (not self.params["overwrite_edge_list"]) and os.path.isfile(
            os.path.join(os.getcwd(), 
            "data", 
            self.county, 
            (self.params["run_name"]+ "_edgeList.parquet")
            )):
            print(f"Edge list found for {self.params["run_name"]}, reading...")
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

        #try speeding things up with a sparse
        src = self.edge_list['source'].to_numpy(dtype = np.int32)
        tgt = self.edge_list['target'].to_numpy(dtype = np.int32)
        weights = self.edge_list['weight'].to_numpy(dtype = np.float32)
        N_nodes = self.N

        row = np.concatenate([src, tgt])
        col = np.concatenate([tgt, src])
        dat = np.concatenate([weights, weights])
        self.adj_matrix = csr_matrix((dat, (row, col)), shape = (N_nodes, N_nodes))

        self.individual_lookup = build_individual_lookup(self.contacts_df)
        self.neighbor_map = self.build_neighbor_map()
        self.existing_edge_set= set() # track existing edges
        self.fast_neighbor_map = {} #preprocess dict of dicts to speed up later
        for src, nbr_list in self.neighbor_map.items():
            self.fast_neighbor_map[src] = {tgt: (weight, ct) for tgt, weight, ct in nbr_list}

    #Set a theoretical full with axis positions for visualization purposes
        self.full_node_list = sorted(set(self.edge_list['source']).union(set(self.edge_list['target'])))
        name_to_ind_dict = {name:ind for ind, name in enumerate(self.full_node_list)}
        g_full = ig.Graph()
        g_full.add_vertices(len(self.full_node_list))
        g_full.vs['name'] = self.full_node_list #Node names in the graph correspond to model individual indices

        full_edges = list(zip(
            [name_to_ind_dict[src] for src in self.edge_list['source']],
            [name_to_ind_dict[tgt] for tgt in self.edge_list['target']]
        ))
        g_full.add_edges(full_edges)
        self.fixed_layout = g_full.layout('grid') #set a layout for the full graph
        self.layout_node_names = self.full_node_list
        self.layout_name_to_ind = {name:i for i, name in enumerate(self.layout_node_names)}
        self.g_full = g_full.copy()


#State tracking: 0 = S, 1 = E, 2 = I, 3 = R
        self.is_vaccinated = self.rng.random(self.N) < params["vax_uptake"]

        self.state = np.zeros(self.N, dtype = np.int8) #current state of each ind
        self.time_in_state = np.zeros(self.N, dtype = np.float32) #time since transition
        self.incubation_periods = np.full(self.N, np.nan, dtype = np.float32)
        self.infectious_periods = np.full(self.N, np.nan, dtype = np.float32)

        self.epi_graphs = [] #snapshot of epidemic graph at each timestep
        self.states_over_time = [] #time series of inds in each state
        
        

    #Seed initial Infection
        initial_infectious = params["I0"]
        self.state[initial_infectious] = 2
        
        self.infectious_periods[initial_infectious] = self.assign_infectious_period(initial_infectious)
        
        #Initialize Graph
        self.epi_g = self.initialize_graph(indices = initial_infectious)
        self.epi_graphs.append(self.epi_g.copy())

        #Initial States Variables
        S = list(set(range(self.N)) - set(initial_infectious))
        E = []
        I = initial_infectious  # noqa: E741
        R = []
        self.states_over_time.append([S,E,I,R])

        self.new_exposures = [[]] #track which individuals became exposed at which timestep, which really correlates to pre-infectious in this model

        self.new_infections = [initial_infectious] #track which individuals became infectious at which timestep

#Set Simulation Tags
        self.simulation_end_day = self.Tmax
        self.stochastic_dieout = False
        self.model_has_run = False


#Set-up results folder
        
        self.results_folder = os.path.join(os.getcwd(), 
            "results", self.params["run_name"])
        if self.params["save_data_files"]:
            if not os.path.exists(self.results_folder):
                os.mkdir(self.results_folder)

    #Helper functions to convert between
    #Name: an individual's number (index of contact_df/individual_lookup))
    #Ind: the index of the individual's vertex in the model
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
            raise TypeError("inds myst be int, list, or numpy array.")
            

    #@profile
    def build_neighbor_map(self):
        """
        Build a mapping from node to list of (neighbor, weight) (make neighbor queries O(1))
        """
        neighbor_map = {}
        for row in self.edge_list.itertuples(index = False):
            src, tgt, w, ct = row.source, row.target, row.weight, row.contact_type
            if src not in neighbor_map:
                neighbor_map[src] = []
            neighbor_map[src].append((tgt, w, ct))
        return neighbor_map

    #@profile
    def initialize_graph(self, indices):
        """
        Initialize an igraph with the provided index/indices and all their neighbors, with weighted edges
        """
            # Make sure all indices are ints and unique
        indices = [int(i) for i in (indices if isinstance(indices, (list, np.ndarray)) else [indices])]
        indices = np.unique(np.array(indices, dtype = int))
        node_set = set(indices)

        # Get all required nodes
        for ind in indices:
            neighbors = self.neighbor_map.get(ind, [])
            node_set.update([int(neighbor) for neighbor, _, _ in neighbors])
        node_list = sorted(node_set)
        node_name_to_index = {name:ind for ind, name in enumerate(node_list)}
        N_nodes = len(node_list)



        # Gather all edges
        edge_data = []

        for ind in node_list:
            for neighbor, weight, ct in self.neighbor_map.get(ind, []):
                neighbor = int(neighbor)
                if neighbor in node_set and ind != neighbor:
                    ind1, ind2 = node_name_to_index[ind], node_name_to_index[neighbor]
                    #avoid duplicated edges by only adding this direction
                    if ind1 < ind2:
                        edge_data.append((ind1, ind2, weight, ct)) 
        #double check for no duplicate edges
        edge_data = list(set(edge_data))

        #convert to np arrays for efficiency and gather edge data
        edge_array = np.array([(e[0],e[1]) for e in edge_data], dtype = int)
        weight_array = np.array([e[2] for e in edge_data], dtype = float)
        type_array = np.array([e[3] for e in edge_data], dtype = object)


        g = ig.Graph()
        g.add_vertices(N_nodes)
        g.vs["name"] = node_list

        if edge_array.size > 0:
            g.add_edges(edge_array.tolist())
            g.es["weight"] = weight_array.tolist()
            g.es["contact_type"] = type_array.tolist()
        
        #assign node attributes with df subsetting
        lookup_sub = self.individual_lookup.loc[node_list]
        g.vs["age"] = lookup_sub["age"].to_numpy()
        g.vs["sex"] = lookup_sub["sex"].to_numpy()


        #track edges
        self.existing_edge_set_init = set(tuple(sorted((node_list[e[0]], node_list[e[1]]))) for e in edge_array)

        return g
    #@profile
    def add_to_graph(self, g: ig.Graph, indices):
        """
        Quickly add all neighbors of indices to igraph g without duplicates, ensuring all node names are ints, assigns demographic attributes to each new node when added
        """
        # Always treat indices as list of ints
        if isinstance(indices, int):
            indices = [indices]
        indices = [int(i) for i in np.asarray(indices)]
        indices = np.unique(np.asarray(indices, dtype = int))
        existing_names = set(self.ind_to_name(g, range(g.vcount())))

        # Gather current edges as tuples of int
        new_nodes = set()
        edge_data = []

        #gather new nodes and edges
        for ind in indices:
            neighbors_dict = self.fast_neighbor_map.get(ind, {})
            for neighbor, (weight, ct) in neighbors_dict.items():
                neighbor = int(neighbor)
                #only add a node if it isn't present
                if neighbor not in existing_names:
                    new_nodes.add(neighbor)
                #don't duplicate edges or add loops
                if ind != neighbor:
                    node1, node2 = min(ind, neighbor), max(ind, neighbor)
                    if (node1, node2) not in self.existing_edge_set:
                        edge_data.append((node1, node2, weight, ct))

        #add new nodes to graph in batch
        if new_nodes:
            g.add_vertices(len(new_nodes))
            newly_added = sorted(new_nodes)
            g.vs[-len(new_nodes):]["name"] = newly_added

            #assign node attributes
            lookup_sub = self.individual_lookup.loc[newly_added]
            g.vs[-len(new_nodes):]["age"] = lookup_sub["age"].to_numpy()
            g.vs[-len(new_nodes):]["sex"] = lookup_sub["sex"].to_numpy()

            existing_names.update(newly_added)



        #deduplicate and add new edges to graph in batch 
        name_to_node_ind = {int(v["name"]): v.index for v in g.vs}
        edge_tuples = set((e[0], e[1]) for e in edge_data)

        #prepare arrays for igraph
        edges_array = np.array([(name_to_node_ind[i], name_to_node_ind[j]) for i, j in edge_tuples], dtype = int)
        weights_array = np.array([next(e[2] for e in edge_data if (e[0], e[1]) == (i, j)) for i, j in edge_tuples], dtype=float)
        types_array = np.array([next(e[3] for e in edge_data if (e[0], e[1]) == (i, j)) for i, j in edge_tuples], dtype=object)

        if len(edges_array) > 0:
            g.add_edges(edges_array.tolist())
            g.es[-len(edges_array):]["weight"] = weights_array.tolist()
            g.es[-len(edges_array):]["contact_type"] = types_array.tolist()

            self.existing_edge_set.update(edge_tuples)
    
        return g

    
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
    def determine_new_exposures(self):
        """
    Uses the current epi_g, self.neighbor_map, and self.individual_lookup to determine who is newly exposed in a given step
        """
        N = self.N
        newly_exposed = np.zeros(N, dtype = bool)

        infectious_indices = np.where(self.state == 2)[0]
        if len(infectious_indices) == 0:
            return np.array([], dtype = int)
        

        for src in infectious_indices:
            neighbors = self.adj_matrix[src].indices
            weights = self.adj_matrix[src].data

            sus_mask = (self.state[neighbors] == 0)
            sus_neighbors = neighbors[sus_mask]
            sus_weights = weights[sus_mask]
        
            vax_src = self.is_vaccinated[src]
            vax_tgt = self.is_vaccinated[sus_neighbors]
            ages_targets = self.individual_lookup.loc[sus_neighbors, "age"].to_numpy()

            vax_efficacy = self.params['vax_efficacy']

            prob = self.params['base_transmission_prob'] * sus_weights
            if vax_src:
                prob *= self.params['relative_infectiousness_vax']
            prob *= ((1- vax_efficacy)**vax_tgt)

            #example implementation of heterogeneous susceptibility
            under_five = ages_targets <= 5
            prob[under_five] *= self.params['susceptibility_multiplier_under_five']

            draws = self.rng.random(prob.shape)
            infected = sus_neighbors[draws < prob]
            newly_exposed[infected] = True

        result = np.where(newly_exposed)[0]


        return result

    #@profile
    def step(self):
        """
        Takes data from a previous step's self.state, and updates, expanding the graph as appropriate
        """

        #S -> E
        newly_exposed = self.determine_new_exposures()

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


        #Save timestep data
        S = list(np.where(self.state == 0)[0])
        E = list(np.where(self.state == 1)[0])
        I = list(np.where(self.state == 2)[0])
        R = list(np.where(self.state == 3)[0])
        self.states_over_time.append([S,E,I,R])
        self.new_exposures.append(newly_exposed)
        self.new_infections.append(to_infectious)


    #@profile
    def simulate(self):
        t = 0
        while t < self.Tmax:
            self.step()
            S, E, I, R = self.states_over_time[-1]  # noqa: E741
            if not E and not I:
                self.simulation_end_day = t + 1 #as it did run another step 
                self.stochastic_dieout = True
                break
            t += 1
        self.model_has_run = True

    def epi_curve(self, suffix: str = None):

        #Produce an epi-curve of the number of individuals infected at each timestep
        counts = [len(exposed) for exposed in self.new_exposures]
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

    def cumulative_incidence_plot(self, strata:str = None, time: int = None, suffix: str = None) -> None:
        """
        Plots cumulative incidence over time, optionally stratified

        Args:
            time (int, optional): time range for plot, 0-t, defaults to full timespan
            strata (str, optional)  stratification factor for plot

        Creates a matplotlib plot
        """
        pop_size = self.N
        exposures = self.new_exposures
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
            plt.title(f"Cumulative Incidence Over Time for  {self.params["run_name"]}")
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



    #@profile
    def draw_network(self, t: int, ax=None, clear: bool =True, saveFile: bool = False, suffix: str = None):
        """Visualizes the active subnetwork at a time t, that consists of all "active nodes" (E, I, or R) and all of their susceptible neighbors

        Args:
            t (int): timestep to plot
            ax (matplotlib axis, optional): axis for plotting
            clear (bool): whether to clear axis
            saveFile (bool): save to file
            suffix (str): filename suffix
        """

        #get indices of E, I, R
        S, E, I, R = self.states_over_time[t]
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

    def make_movie(self, dt: int = 1, filename: str = "network_outbreak.mp4", fps: int = 3) -> None:
        """_summary_

        Args:
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
        timesteps = list(range(0, len(self.states_over_time), dt))
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

    def make_graphml_file(self, t: int, suffix: str = None):
        """ Visualize the network at a given timestep using grephi

        Args:
            t (int): a timestep of the network to visualize

        Returns:
            Saves a .graphml of the network to open externally
        """

        #Get neighbors of affected nodes from g_full
        S, E, I, R = self.states_over_time[t]
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


       


    

#Test run on a really small population
if __name__ == "__main__":
    Keweenaw_contacts = pd.read_parquet("./data/Keweenaw/contacts.parquet")

    testModel = NetworkModel(contacts_df = Keweenaw_contacts)
    print("Running Model on Keweenaw County...")
    testModel.simulate()

    print("Displaying epidemic curve...")
    testModel.epi_curve()

    final_timestep = testModel.simulation_end_day

    # print(f"Drawing Network at timestep {final_timestep}...")
    # testModel.draw_network(final_timestep, saveFile = testModel.params["save_plots"])



    print("Displaying cumulative incidence...")
    testModel.cumulative_incidence_plot()
    






