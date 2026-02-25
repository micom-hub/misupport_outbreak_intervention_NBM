import numpy as np
from typing import Dict, Any


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
