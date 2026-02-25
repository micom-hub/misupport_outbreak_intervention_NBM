import numpy as np
from typing import Dict, Optional, Any, Callable, List
import warnings
from collections import defaultdict


from new_scripts.lhd.actions import ActionBase, ActionToken, CallIndividualsAction
from new_scripts.lhd.algorithms import AlgorithmBase, RandomPriority
from scripts.network_model import NetworkModel


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
                costs = np.full(nodes.shape[0], np.nan, dtype = np.float32)
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
