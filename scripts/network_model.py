import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import igraph as ig
from typing import TypedDict, List

from scripts.syntheticAM import df_to_adjacency

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

    # Population Params
    vax_uptake: float
    hh_contacts: int
    wp_contacts: int
    sch_contacts: int
    gq_contacts: int
    cas_contacts: int


    # Simulation Settings
    simulation_duration: int  # days
    dt: float  # steps per day
    I0: List[int]
    seed: int

DefaultModelParams: ModelParameters = {
    "base_transmission_prob": 0.8,
    "incubation_period": 10.5,
    "infectious_period": 5,
    "gamma_alpha": 20,
    "incubation_period_vax": 10.5,
    "infectious_period_vax": 5,
    "relative_infectiousness_vax": 0.05,
    "vax_efficacy": 0.997,
    "vax_uptake": 0, #.85

    "hh_contacts": 1,
    "wp_contacts": 1,
    "sch_contacts": 1,
    "gq_contacts": 1,
    "cas_contacts": 1,

    "simulation_duration": 45,
    "dt": 1,
    "I0": [0],
    "seed": 2026


}

class NetworkModel:
    def __init__(self, adj_csr, params = DefaultModelParams):
        """
        Unpack parameters, set-up storage variables, and initialize model
        """
#Unpack params and model settings
        
        self.params = params
        self.adj_csr = adj_csr
        self.N = adj_csr.shape[0]
        self.Tmax = self.params["simulation_duration"]
        self.rng = np.random.default_rng(self.params["seed"])


    #set a full graph with positions
        # self.full_graph = ig.Graph.Adjacency((self.adj_csr.toarray() != 0).tolist())
        # self.full_graph.vs["name"] = list(range(self.N))
        # self.fixed_layout = self.full_graph.layout("fr")


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
        I = initial_infectious
        R = []
        self.states_over_time.append([S,E,I,R])

        self.new_exposures = [[]] #track which individuals became exposed at which timestep
        self.new_infections = [initial_infectious] #track which individuals became infected at which timestep

#Set Simulation Tags
        self.simulation_end_day = None
        self.stochastic_dieout = False
        self.model_has_run = False
    
#TODO remove duplicated edges
    def initialize_graph(self, indices):
        """
        Initialize an igraph with a given index and all it's neighbors from self.adj_csr
        """
        if isinstance(indices, int):
            indices = [indices]

        node_set = set(indices)
        node_set.update(int(n) for ind in indices for n in self.adj_csr[ind].indices)
        node_list = list(node_set)

        g = ig.Graph()
        g.add_vertices(len(node_list))
        g.vs["name"] = node_list

        name_to_vertex = {name: vtx for vtx, name in enumerate(node_list)}

        edges = []
        weights = []
        for ind in indices:
            for n in self.adj_csr[ind].indices:
                s = name_to_vertex[ind]
                t = name_to_vertex[n]
                edges.append((s, t))
                weights.append(float(self.adj_csr[ind, n]))
        
        g.add_edges(edges)
        g.es["weight"] = weights

        return g
        

    def add_to_graph(self, g: ig.Graph, indices):
        """
        Quickly add neighbors of 'indices' to an igraph, using adjacency from self.adj_csr

        """
        if isinstance(indices, int):
            indices = [indices]
        indices = np.asarray(indices, dtype = int)

        #names of nodes correspond to indexes in the adjacency matrix
        existing_names = set(g.vs["name"])

        #get all neighbors for all indices
        indptr = self.adj_csr.indptr
        neighbors = np.concatenate([
            self.adj_csr.indices[indptr[ind]:indptr[ind+1]] for ind in indices
        ])
        new_names = set(neighbors) - existing_names
        if new_names:
            g.add_vertices(len(new_names))
            newly_added = list(new_names)
            g.vs[-len(new_names):]["name"] = newly_added
        
        name_to_vertex = {v["name"]: v.index for v in g.vs}

        edges = []
        weights = []
        for ind in indices:
            source = name_to_vertex[ind]
            targets = self.adj_csr.indices[indptr[ind]:indptr[ind+1]]
            for t in targets:
                target = name_to_vertex[t]
                edges.append((source, target))
                weights.append(float(self.adj_csr[ind, t]))
        
        #Remove duplicate edges
        g.add_edges(edges)
        g.es[-len(edges):]["weight"] = weights

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

        

    def step(self):
        """
        Takes data from a previous step's self.state, and updates, expanding the graph as appropriate
        """


        #Determine new exposures
        newly_exposed = []

        current_nodes = np.array([int(v["name"]) for v in self.epi_g.vs])
        infectious_nodes = current_nodes[self.state[current_nodes] == 2]

        for infec_ind in infectious_nodes:
            neighbors_raw = self.epi_g.neighbors(infec_ind)
            neighbors = np.array([int(n) if not isinstance(n, int) else n for n in neighbors_raw], dtype = int)
            sus_neighbors = neighbors[self.state[neighbors] == 0]

            if sus_neighbors.size:
                #get edge weights
                eids = [self.epi_g.get_eid(infec_ind, nbr) for nbr in sus_neighbors]
                weights = np.array([self.epi_g.es[eid]["weight"] for eid in eids])
                #calculate transmission prob for each
                prob = self.params["base_transmission_prob"]*weights
                #adjust for vaccination of i, j
                source_vax = self.is_vaccinated[infec_ind]
                exposed_vax = self.is_vaccinated[sus_neighbors]
                if source_vax:
                    prob *= self.params["relative_infectiousness_vax"]
                prob *= ((1- self.params["vax_efficacy"]) ** exposed_vax)

                draws = self.rng.random(sus_neighbors.shape)
                new_exposed = sus_neighbors[draws < prob]
                newly_exposed.extend(new_exposed.tolist())

        newly_exposed = np.unique(newly_exposed)

        #Determine exposed to infectious
        exposed_nodes = current_nodes[self.state[current_nodes] == 1]
        self.time_in_state[exposed_nodes] += 1
        to_infectious = exposed_nodes[self.time_in_state[exposed_nodes] >= self.incubation_periods[exposed_nodes]]

        #Determine infectious to recovered
        self.time_in_state[infectious_nodes] += 1
        to_recovered = infectious_nodes[self.time_in_state[infectious_nodes] >= self.infectious_periods[infectious_nodes]]

        #Move individuals to new compartments and handle
        if newly_exposed.size:
            self.state[newly_exposed] = 1
            self.time_in_state[newly_exposed] = 0
            self.incubation_periods[newly_exposed] = self.assign_incubation_period(newly_exposed)

        #Expand graph to include newly infectious
        if to_infectious.size:
            self.state[to_infectious] = 2
            self.time_in_state[to_infectious] = 0
            self.infectious_periods[to_infectious] = self.assign_infectious_period(to_infectious)
            self.epi_g = self.add_to_graph(g = self.epi_g, indices = to_infectious)

        if to_recovered.size:
            self.state[to_recovered] = 3
            self.time_in_state[to_recovered] = 0

        inactive_mask = np.logical_or(self.state[current_nodes] == 0, self.state[current_nodes] == 1)
        inactive_nodes = current_nodes[inactive_mask]
        self.time_in_state[inactive_nodes] += 1


        #Save timestep data
        S = current_nodes[self.state[current_nodes] == 0].tolist()
        E = current_nodes[self.state[current_nodes] == 1].tolist()
        I = current_nodes[self.state[current_nodes] == 2].tolist()
        R = current_nodes[self.state[current_nodes] == 3].tolist()

        self.states_over_time.append([S,E,I,R])
        self.epi_graphs.append(self.epi_g.copy())

        self.new_exposures.append(newly_exposed)
        self.new_infections.append(to_infectious)
    


    #FUNCTION NOTES
        #take data from previous step's self.state


        #For each infectious individual, find which of their neighbors are susceptible, and do a probabilistic draw (transmission_prob*contact_weight*vax_efficacy (if exposee is vaccinated) *relative_infectiousness_vax (if infectious is vaccinated)))
        #For individuals who become exposed, add them to a newly_exposed list

        #For all exposed individuals, check if their time in state is greater than their incubation period. If yes, add them to newly_infectious, else, do nothing

        #For all infectious individuals, check if their time-in-state is greater than their infectious period. if yes, add them to newly_recovered, else do nothing.



        #Add 1 to everybody's time in state

        #Add all neighbors of newly-infected I to the graph as susceptible

        #Move all newly exposed, newly infectious, and newly-recovered to their new states, and reset their time in state to 0


    def simulate(self):
        t = 1
        while t < self.Tmax:
            self.step()
            S, E, I, R = self.states_over_time[-1]
            if not E and not I:
                self.simulation_end_day = t
                self.stochastic_dieout = True
                break
            t += 1
        self.model_has_run = True

    



    def epi_curve(self):

        #Produce an epi-curve of the number of individuals infected at each timestep
        counts = [len(exposed) for exposed in self.new_exposures]
        plt.figure(figsize=((8,4)))
        plt.bar(range(len(counts)), counts, color = 'orange', label = "Infections Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Number of Infectious Contacts Made")
        plt.grid(axis = 'y', alpha = 0.5)
        plt.tight_layout()
        plt.show()





    def draw_network(self, t, ax=None, clear=True):

        #Take a timestep t as an input, and return a plot of the graph
        g = self.epi_graphs[t]
        S, E, I, R = self.states_over_time[t]
        color_map = {i: "blue" for i in S}
        color_map.update({i: "orange" for i in E})
        color_map.update({i: "red" for i in I})
        color_map.update({i: "green" for i in R})
        colors = [color_map[v["name"]] for v in g.vs]

        node_ind = [int(v["name"]) for v in g.vs]
        layout = g.layout('fr')
        if ax is None:
            fig, ax = plt.subplots(figsize = (8,8))
            show_plot = True
        else:
            show_plot = False
        if clear:
            ax.clear()
        ig.plot(
            g,
            layout = layout,
            vertex_color = colors,
            vertex_size = 30,
            edge_color = "gray",
            bbox = (600,600),
            target = ax,
            vertex_label = g.vs["name"]
        )
        ax.set_title(f"Network at t = {t}")
        if show_plot:
            plt.show()







    def make_movie(self, dt=1, movie_file="network_outbreak.mp4", fps=3):
        #To be completed later
        return
       



if __name__ == "__main__":
    Keweenaw_adj = df_to_adjacency(county = "Alcona", saveFile = False, sparse = True)
    testModel = NetworkModel(adj_csr = Keweenaw_adj)
    print("Running Model on Keweenaw County...")
    testModel.simulate()

    print("Displaying epidemic curve...")
    testModel.epi_curve()

    final_timestep = testModel.simulation_end_day
    print(f"Drawing Network at timestep {final_timestep}...")
    testModel.draw_network(final_timestep)




