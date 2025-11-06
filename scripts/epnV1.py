# %% Start here
import igraph as ig
from igraph import Graph
import numpy as np
from scipy.stats import expon, uniform

import matplotlib
import matplotlib.pyplot as plt

import heapq

print("Initializing Network...")


def contactNetwork(N, nNeighbors):
    G = Graph.Watts_Strogatz(
        1, N, nNeighbors, 0.3
    )  # .5 right now to have a fairly random graph
    return G


MeaslesParameters = {
    # Disease Params
    "R0": 15.0,  # transmission rate
    "incubation_period": 10.5,  # time from t_infected to symptom onset (days)
    "infectious_period": 5,  #
    # Contact Params
    "school_contacts": 5.63424,
    "other_contacts": 2.2823,
    "random_network": True,
    # Model Setting Params:
    "population": 20,  # list of population sizes if running multiple in parallel
    "I0": [1, 5],
    "threshold_values": [10],  # outbreak threshold, only first value used for now
    "is_stochastic": True,  # False for deterministic,
    "simulation_seed": 123456789,
}


# %% EPN Time!
def network2epn(contactNetwork: ig.Graph, params: dict = None) -> ig.Graph:
    """
    Args:
        contactNetwork (ig.Graph): _description_
        params (dict): A dictionary containing SEIR parameters
            params["latent_distr"]
            params["infectious_distr"]
            params["beta"]

    Returns:
        epn: directed graph with edges (i->j, with contact time as an attribute)
    """
    # unpack params

    is_stochastic = params["is_stochastic"]
    if is_stochastic:
        RNG = np.random.Generator(np.random.MT19937(seed=params["simulation_seed"]))

    R0 = params["R0"]
    school_contacts = params["school_contacts"]
    other_contacts = params["other_contacts"]
    avg_infectious_period = params["infectious_period"]
    beta = R0 / ((school_contacts + other_contacts) * avg_infectious_period)

    # Initialize epn based on contact network
    n = contactNetwork.vcount()
    epn = ig.Graph(directed=True)
    epn.add_vertices(n)

    # Assume exponential distributions for now
    incubation_distr = expon(scale=params["incubation_period"])
    infectious_distr = expon(scale=params["infectious_period"])

    # Assign incubation and infectious periods a priori
    incubation_periods = np.array([incubation_distr.rvs() for node in range(n)])
    infectious_periods = np.array([infectious_distr.rvs() for node in range(n)])

    epn.vs["incubation"] = incubation_periods
    epn.vs["infectious"] = infectious_periods
    epn.vs["rel_infectiousness"] = np.ones(n)
    epn.vs["rel_susceptibility"] = np.ones(n)

    # Set time of infection arrays, and assign initial infections

    def tau_ij_relative(
        I_i,
        beta,
        rel_infectiousness_i=1.0,
        rel_susceptibility_j=1.0,
        dist="exponential",
    ):
        """
        For infected individual i, and their neighbor on *contactNetwork* j,
        draw the contact interval (tau_ij), or required time in contact, for transmission from i to j. This is adjusted by the relative infectiousness of i, and relative susceptibility of j


        Args:
            I_i: Infectious period of i
            beta: Per-contact transmission rate
            rel_infectiousness_i (float, optional): relative infectiousness of i
            rel_susceptibility_j (float, optional): relative susceptibility of j
            dist : exponential or uniform draw for tau

        Returns:
        tau_ij: the transmission interval since i becomes infectious, or np.inf if this time is outside infectious period (j not infected by i)
        """
        effective_beta = beta * rel_infectiousness_i * rel_susceptibility_j

        # if no transmission is possible
        if effective_beta <= 0:
            return np.inf

        if dist == "exponential":
            tau = expon(scale=1.0 / effective_beta).rvs()
            return tau if tau < I_i else np.inf

        elif dist == "uniform":
            # use susceptibility and infectiousness to shrink interval of transmission
            # uniform draw on interval to decide if transmission is possible
            chance = rel_infectiousness_i * rel_susceptibility_j
            if np.random.rand() < chance:
                return uniform(loc=0, scale=I_i).rvs()
            else:
                return np.inf
        else:
            raise ValueError("Unknown distribution")

    for i in range(n):
        Ei = incubation_periods[i]
        Ii = infectious_periods[i]
        inf_i = epn.vs[i]["rel_infectiousness"]
        for j in contactNetwork.neighbors(i):
            sus_j = epn.vs[j]["rel_susceptibility"]
            tau_ij = tau_ij_relative(Ii, beta, inf_i, sus_j)
            if i == j:
                continue  # bypass any self-adjacency
            elif tau_ij < np.inf:
                infectious_contact_time = Ei + tau_ij
                epn.add_edge(i, j, infectious_contact_time=infectious_contact_time)

    return epn


def simulate_epidemic(epn: ig.Graph, imported_infections: list):
    n = epn.vcount()
    t_infected = np.full(n, np.inf)
    # set initial infections
    for ind in imported_infections:
        t_infected[ind] = 0.0

    queue = []
    for i in imported_infections:
        for e in epn.es.select(_source=i):
            j = e.target
            t_event = t_infected[i] + e["infectious_contact_time"]
            queue.append((t_event, i, j))
    heapq.heapify(queue)

    while queue:
        t_event, i, j = heapq.heappop(queue)
        # if the event happens,
        if t_event < t_infected[j]:
            t_infected[j] = t_event
            for e in epn.es.select(_source=j):
                k = e.target
                next_t = t_infected[j] + e["infectious_contact_time"]
                if next_t < t_infected[k]:
                    heapq.heappush(queue, (next_t, j, k))

    return t_infected


# TODO Implement some analytic tools here


def epicurve(times):
    itimes = np.sort(times[np.isfinite(times)])
    curve = np.arange(1, len(itimes + 1))
    plt.hist(itimes)
    
def getAttackRate(times):
    attack_rate = np.sum(np.isfinite(times))/ len(times)
    
    return attack_rate


# example run
if __name__ == "__main__":
    G = contactNetwork(100, 2)
    epn = network2epn(G, MeaslesParameters)
    ig.plot(epn)
    out = simulate_epidemic(epn, [2])
    print(out)
    epicurve(out)
    print(getAttackRate(out))


# %%
