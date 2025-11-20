# %% Imports
import igraph as ig
from igraph import Graph
import numpy as np
from scipy.stats import expon, uniform

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generateAdjacency(N, p, seed=None):

    np.random.seed(seed)
    upper = np.triu(np.random.rand(N, N) < p, k=1)
    adj = upper + upper.T
    return adj.astype(int)


class NetworkSEIR:

    def __init__(self, adj, initial_infected, graph_attrs=None, nodeTrack=True):
        """
        adj: NxN adjacency matrix (scipy.sparse)
        initial_infected: list/array of node indices that import infection
        graph_attrs: population attributes for the adjacency matrix
        """
        self.N = adj.shape[0]
        self.adj = adj
        self.graph_attrs = graph_attrs or {}
        self.nodeTrack = nodeTrack

        self.states = np.zeros(self.N, dtype=np.int8)  # initial pop = S
        self.states[initial_infected] = 2  # set initially infected (S=2)

        # Array of state histories and infected nodes
        self.S_hist = [np.sum(self.states == 0)]
        self.E_hist = [np.sum(self.states == 1)]
        self.I_hist = [np.sum(self.states == 2)]
        self.R_hist = [np.sum(self.states == 3)]
        self.I_nodes = [list(np.where(self.states == 2)[0])]  # must track

        if self.nodeTrack:
            self.S_nodes = [list(np.where(self.states == 0)[0])]
            self.E_nodes = [list(np.where(self.states == 1)[0])]
            self.R_nodes = [list(np.where(self.states == 3)[0])]

    def step_vectorized(self, beta=0.03, sigma=0.2, gamma=0.1, dt=1.0):
        susceptible = np.where(self.states == 0)[0]
        exposed = np.where(self.states == 1)[0]
        infectious = np.where(self.states == 2)[0]

        # S -> E
        infectious_vec = (self.states == 2).astype(int)
        infected_contacts = self.adj[susceptible, :] @ infectious_vec
        p_infect = 1 - np.exp(-beta * infected_contacts * dt)
        rand = np.random.rand(
            len(susceptible)
        )  # make sure EV for this aligns with p_infect
        new_E = susceptible[rand < p_infect]

        # E -> I
        e_draws = np.random.rand(len(exposed))
        new_I = exposed[e_draws < (1 - np.exp(-sigma * dt))]

        # I -> R
        i_draws = np.random.rand(len(infectious))
        new_R = infectious[i_draws < 1 - np.exp(-gamma * dt)]

        self.states[new_E] = 1
        self.states[new_I] = 2
        self.states[new_R] = 3

        self.S_hist.append(np.sum(self.states == 0))
        self.E_hist.append(np.sum(self.states == 1))
        self.I_hist.append(np.sum(self.states == 2))
        self.R_hist.append(np.sum(self.states == 3))
        self.I_nodes.append(np.where(self.states == 2)[0].tolist())

        # Record states
        if self.nodeTrack:
            if self.nodeTrack:
                self.S_nodes.append(list(np.where(self.states == 0)[0]))
                self.E_nodes.append(list(np.where(self.states == 1)[0]))
                self.R_nodes.append(list(np.where(self.states == 3)[0]))

    def run(self, tMax, beta=0.03, sigma=0.2, gamma=0.1, dt=1.0):
        self.tMax = tMax
        for t in range(tMax):
            self.step_vectorized(beta, sigma, gamma, dt)
            if np.sum(self.states == 2) == 0 and np.sum(self.states == 1) == 0:
                break

    def get_output(self):
        return (
            np.array(self.S_hist),
            np.array(self.E_hist),
            np.array(self.I_hist),
            np.array(self.R_hist),
        )

    def visualize_network(
        self, timestep=None, layout_type="fr", figsize=(8, 8), outdir=None, show=True
    ):
        """_summary_

        Args:
            timestep (_type_, optional): _description_. Defaults to None.
            layout_type (str, optional): _description_. Defaults to 'fr'.
            figsize (tuple, optional): _description_. Defaults to (8,8).
            outdir (_type_, optional): _description_. Defaults to None.
            show (bool, optional): _description_. Defaults to True.
        """

        N = self.N
        if timestep is None:
            timestep = (
                len(self.I_nodes) - 1 if hasattr(self, "I_nodes") else self.tMax - 1
            )

        row, col = np.where(self.adj)
        edges = [(int(i), int(j)) for i, j in zip(row, col) if i < j]
        g = ig.Graph()
        g.add_vertices(N)
        g.add_edges(edges)
        layout = g.layout(layout_type)

        colors = ["black"] * N
        if hasattr(self, "E_nodes"):
            for i in self.E_nodes[timestep]:
                colors[i] = "orange"
            for i in self.I_nodes[timestep]:
                colors[i] = "red"
            for i in self.R_nodes[timestep]:
                colors[i] = "green"
        else:
            for i in self.I_nodes[timestep]:
                colors[i] = "red"  # only track infected if not everything

        g.vs["color"] = colors
        fig, ax = plt.subplots(figsize=figsize)
        visual_style = {
            "vertex_size": 18,
            "vertex_color": colors,
            "edge_color": "#777",
            "vertex_label": None,
            "layout": layout,
            "bbox": (600, 600),
            "margin": 20,
        }
        ig.plot(g, **visual_style, target=ax)
        plt.title(f"Network SEIR States (step {timestep})")
        legend_items = [
            mpatches.Patch(color="black", label="Susceptible (S)"),
            mpatches.Patch(color="orange", label="Exposed (E)"),
            mpatches.Patch(color="red", label="Infectious (I)"),
            mpatches.Patch(color="green", label="Recovered (R)"),
        ]
        plt.legend(handles=legend_items, bbox_to_anchor=(1.05, 1))

        if outdir is not None:
            plt.savefig(
                f"{outdir}/network_seir_step_{timestep}.png", bbox_inches="tight"
            )
            plt.close(fig)
        elif show:
            plt.show()
        else:
            plt.close(fig)


N = 20
p = 0.3
aM = generateAdjacency(N, p, 42)
imported_infections = [1]

model = NetworkSEIR(aM, imported_infections)
model.run(tMax=100, beta=0.3, sigma=0.2, gamma=0.1)
S, E, I, R = model.get_output()
for i in range(len(I)):
    model.visualize_network(timestep=i, show=False)
# %%

