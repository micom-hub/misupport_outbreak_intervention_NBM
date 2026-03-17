#scripts/lhd/lhd.py
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Any, Callable, List
import warnings
from collections import defaultdict

from scripts.lhd.actions import ActionBase, ActionToken, CallIndividualsAction
from scripts.lhd.state import LHDState
from scripts.lhd.algorithms import AlgorithmBase, RandomPriority
from scripts.lhd.surveillance import SurveillanceModel


if TYPE_CHECKING:
    from scripts.simulation.outbreak_model import NetworkModel


class LocalHealthDepartment:
    def __init__(
        self, 
        seed: int,
        surv_seed: int,
        model: NetworkModel, 
        capacity: Optional[int] = None,
        register_defaults: bool = True,
        algorithm_map: Optional[Dict[str, object]] = None,
        action_factory_map: Optional[Dict[str, Callable[..., ActionBase]]] = None
    ):
    #Unpack LHD settings
        self.model = model
        self.seed = seed
        self.surv_seed = surv_seed
        self.rng = np.random.default_rng(seed)
        self.p_detect_inf = self.model.config.lhd.p_detect_inf
        self.report_delay_days = self.model.config.lhd.report_delay_days
        self.daily_capacity = int(capacity) if capacity is not None else int(self.model.config.lhd.lhd_daily_capacity)
        
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
        self.min_candidate_cost = 1

    #Default action params
        self.default_int_reduction = model.config.lhd.lhd_default_int_reduction
        self.default_int_duration = model.config.lhd.lhd_default_int_duration
        self.default_call_cost = model.config.lhd.lhd_default_call_cost


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
            call_cost = int(cost) if cost is not None else self.default_call_cost,
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

        #Set-up surveillance object
        self.surveillance = SurveillanceModel(
            seed = surv_seed,
            N = self.model.N,
            ages = self.model.ages,
            is_vax = self.model.is_vaccinated,
            p_detect_inf= self.p_detect_inf,
            report_delay_days = self.report_delay_days
        )

        #Set-up LHD knowledge state
        self.state = LHDState(N = self.model.N)

            
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

    def observe(self, t: int, epi_state: Dict[str, np.ndarray], scheduled_actions: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        return self.surveillance.step(t=t, epi_state=epi_state, scheduled_actions=scheduled_actions)


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
            
            #costs must be a single int or array of size len(nodes)
            costs = out.get('costs', None)
            if costs is None:
                costs = np.full(nodes.shape[0], self.min_candidate_cost, dtype = np.int32)

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
                best_costs = np.full(unique_nodes.shape[0], 99999, dtype = np.int32)
                best_params = [None] * unique_nodes.shape[0]

                for occ in range(nodes.shape[0]):
                    uid = inverse[occ]
                    p = float(prios[occ])
                    c = int(costs[occ])
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
        costs_arr = np.array(costs_list, dtype = np.int32)
        params_arr = params_list

        return action_types_arr, nodes_arr, prios_arr, contact_types_arr, costs_arr, params_arr

    def schedule_action(self, action: ActionBase, current_time: int, cost_units: int):
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
            'capacity_used': int(cost_units),
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

    def step(self, *, t: int, epi_state: Dict[str, np.ndarray], scheduled_surv_actions: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:

        """
        One step for the LHD where it:
        1) Processes old interventions
        2) Observes reported cases through surveillance 
        """


        #1 expire old interventions
        self.process_expirations(t)

        #2) Observe events through surveillance
        batch = self.observe(t=t, epi_state = epi_state, scheduled_actions = scheduled_surv_actions) 

        #3 Update knowledge based on observed batch
        self.state.process_batch(batch)

        return batch



    def respond(self, *, t: int, batch: Dict[str, np.ndarray]) -> None:
        """
        Will generate candidates from self.state + batch & schedule actions
        """




        return 

    def reset_for_run(self):
        """
        Reset LHD state for new model run 
        """
        self.expiry = {}
        self.action_log = []
        self._active_actions = {}
        self._action_token_counts = {}
        if hasattr(self, "surveillance") and self.surveillance is not None:
            self.surveillance.reset_for_run(seed=self.surveillance.seed, is_vax = self.model.is_vaccinated)
        if hasattr(self, "state") and self.state is not None:
            self.state.reset_for_run()
