import numpy as np
import pandas as pd
from typing import Dict

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
