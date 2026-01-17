import os
import numpy as np
import pandas as pd
import igraph as ig
from typing import TypedDict, List
import matplotlib.pyplot as plt
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
    simulation_duration: int  # days
    dt: float  # steps per day
    I0: List[int]
    seed: int
    county: str #county to run on
    state: str #state to run on
    save_data_files: bool
    make_movie: bool



DefaultModelParams: ModelParameters = {
    "base_transmission_prob": 0.8,
    "incubation_period": 10.5,
    "infectious_period": 5,
    "gamma_alpha": 20,
    "incubation_period_vax": 10.5,
    "infectious_period_vax": 5,
    "relative_infectiousness_vax": 0.05,
    "vax_efficacy": 0.997,
    "vax_uptake": 0.85, 

    "wp_contacts": 10,
    "sch_contacts": 10,
    "gq_contacts": 20,
    "cas_contacts": 1,

    "hh_weight": 1,
    "wp_weight": .5,
    "sch_weight": .6,
    "gq_weight": .3,
    "cas_weight": .1,

    "run_name" : "test_run",
    "simulation_duration": 45,
    "dt": 1,
    "I0": [906],
    "seed": 2026,
    "county": "Keweenaw",
    "state": "Michigan",
    "save_data_files": True,
    "make_movie": False

}

class NetworkModel:
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

        if os.path.isfile(
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

        self.individual_lookup = build_individual_lookup(self.contacts_df)

        self.neighbor_map = self.build_neighbor_map()


    #Set a theoretical full with axis positions for visualization purposes
        self.full_node_list = sorted(set(self.edge_list['source']).union(set(self.edge_list['target'])))
        g_full = ig.Graph()
        g_full.add_vertices(len(self.full_node_list))
        g_full.vs['name'] = self.full_node_list

        full_edges = list(zip(
            [self.full_node_list.index(src) for src in self.edge_list['source']],
            [self.full_node_list.index(tgt) for tgt in self.edge_list['target']]
        ))
        g_full.add_edges(full_edges)
        self.fixed_layout = g_full.layout('fr') #set a layout for the full graph
        self.layout_node_names = self.full_node_list


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
        if self.params["save_data_files"]:
            self.results_folder = os.path.join(os.getcwd(), 
            "results", self.params["run_name"]
            )
            if not os.path.exists(self.results_folder):
                os.mkdir(self.results_folder)
    
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

    def initialize_graph(self, indices):
        """
        Initialize an igraph with the provided index/indices and all their neighbors, with weighted edges
        """
        # Make sure all indices are ints and unique
        indices = [int(i) for i in (indices if isinstance(indices, (list, np.ndarray)) else [indices])]
        node_set = set(indices)

        # Add neighbors ensuring all are plain int
        for ind in indices:
            ind = int(ind)
            neighbors = self.neighbor_map.get(ind, [])
            for neighbor, _, _ in neighbors:
                node_set.add(int(neighbor))
        node_list = sorted([int(n) for n in node_set])

        name_to_vertex = {int(name): v for v, name in enumerate(node_list)}

        # Preventing duplicates; all int
        edge_set = set()
        edge_data = {}  # i, j, weight, ct

        induced_nodes = set([int(n) for n in node_list])
        for ind in node_list:
            ind = int(ind)
            for neighbor, weight, ct in self.neighbor_map.get(ind, []):
                neighbor = int(neighbor)
                if neighbor in induced_nodes and ind != neighbor:
                    edge_tuple = tuple(sorted((ind, neighbor)))
                    if edge_tuple not in edge_set:
                        edge_set.add(edge_tuple)
                        edge_data[edge_tuple] = (weight, ct)

        edges, weights, types = [], [], []
        for (i, j), (w, t) in edge_data.items():
            edges.append((name_to_vertex[int(i)], name_to_vertex[int(j)]))
            weights.append(w)
            types.append(t)

        g = ig.Graph()
        g.add_vertices(len(node_list))
        g.vs["name"] = [int(n) for n in node_list]
        if edges:
            g.add_edges(edges)
            g.es["weight"] = weights
            g.es["contact_type"] = types

        return g

    def add_to_graph(self, g: ig.Graph, indices):
        """
        Quickly add all neighbors of indices to igraph g without duplicates, ensuring all node names are ints.
        """
        # Always treat indices as list of ints
        if isinstance(indices, int):
            indices = [indices]
        indices = [int(i) for i in np.asarray(indices)]

        existing_names = set([int(name) for name in g.vs["name"]])

        # Gather current edges as tuples of int
        existing_edge_set = set()
        for e in g.es:
            i, j = int(g.vs[e.source]["name"]), int(g.vs[e.target]["name"])
            edge_tuple = tuple(sorted((i, j)))
            existing_edge_set.add(edge_tuple)

        new_nodes = set()
        edge_set = set()
        edge_data = {}

        for ind in indices:
            ind = int(ind)
            for neighbor, weight, ct in self.neighbor_map.get(ind, []):
                neighbor = int(neighbor)
                if neighbor not in existing_names:
                    new_nodes.add(neighbor)
                if ind != neighbor:
                    edge_tuple = tuple(sorted((ind, neighbor)))
                    if edge_tuple not in existing_edge_set and edge_tuple not in edge_set:
                        edge_set.add(edge_tuple)
                        edge_data[edge_tuple] = (weight, ct)

        if new_nodes:
            g.add_vertices(len(new_nodes))
            newly_added = sorted([int(n) for n in new_nodes])
            g.vs[-len(new_nodes):]["name"] = newly_added
            existing_names.update(newly_added)

        

        # Update mapping, all int
        name_to_vertex = {int(v["name"]): v.index for v in g.vs}

        edges, weights, types = [], [], []
        for (i, j), (w, t) in edge_data.items():
            edges.append((name_to_vertex[int(i)], name_to_vertex[int(j)]))
            weights.append(w)
            types.append(t)
        if edges:
            g.add_edges(edges)
            g.es[-len(edges):]["weight"] = weights
            g.es[-len(edges):]["contact_type"] = types



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
            #note infectious indices correspond to names in the igraph
            node_index = self.epi_g.vs.find(name = infec_ind).index
            neighbor_indices = self.epi_g.neighbors(node_index)
            neighbors = np.array([int(self.epi_g.vs[n]["name"]) for n in neighbor_indices])
            sus_neighbors = neighbors[self.state[neighbors] == 0]

            if sus_neighbors.size:
                #get edge weights
                eids = [self.epi_g.get_eid(
                    self.epi_g.vs.find(name = infec_ind).index,
                    self.epi_g.vs.find(name = nbr).index) 
                    for nbr in sus_neighbors]

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

        #New nodes list, adding new nodes to S
        updated_nodes = np.array([int(v["name"]) for v in self.epi_g.vs])


        #Save timestep data
        S = updated_nodes[self.state[updated_nodes] == 0].tolist()
        E = updated_nodes[self.state[updated_nodes] == 1].tolist()
        I = updated_nodes[self.state[updated_nodes] == 2].tolist()  # noqa: E741
        R = updated_nodes[self.state[updated_nodes] == 3].tolist()

        self.states_over_time.append([S,E,I,R])
        self.epi_graphs.append(self.epi_g.copy())

        self.new_exposures.append(newly_exposed)
        self.new_infections.append(to_infectious)

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

    def epi_curve(self):

        #Produce an epi-curve of the number of individuals infected at each timestep
        counts = [len(exposed) for exposed in self.new_exposures]
        counts[0] = len(self.params["I0"]) #add initial infections
        plt.figure(figsize=((8,4)))
        plt.bar(range(len(counts)), counts, color = 'orange', label = "Infections Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Number of Infectious Contacts Made")
        plt.grid(axis = 'y', alpha = 0.5)
        plt.tight_layout()
        plt.show()

    def cumulative_incidence_plot(self, time: int = None, strata:str = None) -> None:
        """
        Plots cumulative incidence over time, optionally stratified

        Args:
            time (int, optional): time range for plot, 0-t, defaults to full timespan
            strata (str, optional)  stratification factor for plot

        Creates a matplotlib plot
        """
        pop_size = self.N

        if time is None:
            time = len(self.states_over_time)

        #Track cumulatively infected individuals
        exposures = self.new_exposures
        if len(exposures) > 0 and len(exposures[0]) == 0:
            exposures[0] = list(self.params["I0"])

        max_time = len(exposures) - 1 if time is None else min(time, len(exposures) - 1)

        cum_inf = []
        tot = 0
        for t in range(max_time+1):
            tot += len(exposures[t])
            cum_inf.append(tot/pop_size)
        
        plt.figure(figsize = (7, 4))
        plt.plot(range(max_time+ 1), cum_inf, color = "red", linewidth = 2)
        plt.xlabel("Time step (day)")
        plt.ylabel("Cumulative Incidence (fraction of population)")
        plt.title(f"Cumulative Incidence Over Time for  {self.params["run_name"]}")
        plt.grid(True, axis = "y", alpha = 0.5)
        plt.tight_layout()
        plt.show()




    def draw_network(self, t: int, ax=None, clear: bool =True):

        #Take a timestep t as an input, and return a plot of the graph
        g = self.epi_graphs[t]
        S, E, I, R = self.states_over_time[t]  # noqa: E741
        color_map = {i: "blue" for i in S}
        color_map.update({i: "orange" for i in E})
        color_map.update({i: "red" for i in I})
        color_map.update({i: "green" for i in R})
        colors = [color_map[v["name"]] for v in g.vs]

        indices = [self.layout_node_names.index(v["name"]) for v in g.vs]
        layout = [self.fixed_layout[i] for i in indices]

        node_labels = g.vs["name"]
        if len(g.vs) > 40:
            node_labels = None


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
            vertex_label = node_labels
        )
        ax.set_title(f"Network at t = {t}")
        if show_plot:
            plt.show()

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
        timesteps = list(range(0, len(self.epi_graphs), dt))
        if (len(self.epi_graphs)-1) not in timesteps:
            timesteps.append(len(self.epi_graphs)-1) #always show last step
        
        def update(frame):
            t = timesteps[frame]
            ax.clear()
            self.draw_network(t, ax = ax, clear = False)
            ax.set_title(f"{self.params['run_name']} | Day {t}")
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
       


#Test run on a really small population
if __name__ == "__main__":
    Keweenaw_contacts = pd.read_parquet("./data/Keweenaw/contacts.parquet")

    testModel = NetworkModel(contacts_df = Keweenaw_contacts)
    print("Running Model on Keweenaw County...")
    testModel.simulate()

    print("Displaying epidemic curve...")
    testModel.epi_curve()

    final_timestep = testModel.simulation_end_day
    print(f"Drawing Network at timestep {final_timestep}...")
    testModel.draw_network(final_timestep)
    testModel.cumulative_incidence_plot()




