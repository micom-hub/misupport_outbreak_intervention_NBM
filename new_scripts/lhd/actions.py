import numpy as np
from typing import Optional, Dict, Any, List
import uuid
from dataclasses import dataclass
from scripts.network_model import NetworkModel


@dataclass
class ActionToken:
    action_id: str
    action_type: str #"call", "vaccination", etc
    contact_type: Optional[str] 
    nodes: np.ndarray #node indices to act on
    factor: Optional[np.ndarray] #reduction factor 
    reversible: bool = True #if action can be undone
    meta: Optional[Dict[str, Any]] = None #any relevant metadata


class ActionBase:
    """
    Abstract action. Subclasses implement apply() to alter the model, returning ActionToken(s)
    """
    def __init__(self, action_type: str, duration:int = 0, kind: Optional[str] = None):
        self.id = uuid.uuid4().hex
        self.action_type = action_type
        self.duration = int(duration)
        self.kind = kind or action_type #more descriptive, sub-type label
        self.reversible = True #subclasses can override

    def apply(self, model: NetworkModel, current_time: int) -> List[ActionToken]:
        """
        Apply action to the model, return a list of ActionTokens describing state changes
        """
        raise NotImplementedError

    def revert_token(self, model: NetworkModel, token: ActionToken) -> None:
        """
        Reverts a previously-applied token for a reversible action
        """


class CallIndividualsAction(ActionBase):
    """
    Calls individuals to inform them of their exposure, applying a multiplicative reduction to their contact weight for school, workplace, and casual contacts

    nodes: list of model node indices
    contact_types: list or iterable of contact types ('cas', 'sch', etc.)
    reduction: fraction (0-1) representing reduction (to be mult by compliance)
    duration: days individual asked to reduce contacts
    call_cost: hours spent per call
    min_factor: minimum allowed reduction (just to avoid divzero errors)
    """
    def __init__(self, nodes: np.ndarray, contact_types: List[str], reduction: float, duration: int, call_cost: float, min_factor: float = 1e-6):
        super().__init__(action_type = "call", duration = duration, kind = "call_individuals")
        self.nodes = np.asarray(nodes, dtype = np.int32)
        self.contact_types = list(contact_types)
        self.reduction = float(reduction)
        self.call_cost = float(call_cost)
        self.min_factor = float(min_factor)
        self.reversible = True

    def apply(self, model: NetworkModel, current_time: int) -> List[ActionToken]:
        #per-node factor, scaled by compliance
        comps = model.compliances[self.nodes] 
        raw = 1.0 - (self.reduction * comps)
        factor = np.clip(raw, self.min_factor, 1.0).astype(np.float32)

        tokens: List[ActionToken] = []
        for ct in self.contact_types:
            #apply reductions
            model.out_multiplier[ct][self.nodes] *= factor
            model.in_multiplier[ct][self.nodes] *= factor

            token = ActionToken(
                action_id = self.id,
                action_type = self.action_type,
                contact_type = ct,
                nodes = self.nodes.copy(),
                factor = factor.copy(),
                reversible = True,
                meta = {'call_cost': self.call_cost}
            )
            tokens.append(token)
        
        return tokens

    def revert_token(self, model: NetworkModel, token: ActionToken) -> None:
        #undo the multiplicative factor by dividing out 
        if token.factor is None:
            return
        f = token.factor
        #safeguard against zeros
        f_safe = np.maximum(f, self.min_factor)
        model.out_multiplier[token.contact_type][token.nodes] /= f_safe
        model.in_multiplier[token.contact_type][token.nodes] /= f_safe
