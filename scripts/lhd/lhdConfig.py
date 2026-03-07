#scripts/lhd/lhdConfig.py
from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable, Tuple, Type

from scripts.lhd.algorithms import (AlgorithmBase, RandomPriority, EqualPriority, PrioritizeElders
)
from scripts.lhd.actions import (CallIndividualsAction, ActionBase)
from scripts.lhd.lhd import LocalHealthDepartment

@dataclass
class LhdVariant:
    """
    Specs for an LHD variant with:
    - Name (string)
    - algorithm_map that maps action types to their algorithms
    - action_factory_map that creates action factories
    - description to briefly explain the variant
    """
    name: str
    algorithm_map: Dict[str, AlgorithmBase] = field(default_factory=dict)
    action_factory_map: Dict[str, Callable[..., ActionBase]] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class LhdConfig:
    """
    Container for LhdVariants with convenience functions
    """
    variants: List[LhdVariant] = field(default_factory=list)

    def __post_init__(self):
        for var in self.variants:
            try:
                validate_variant(var)
            except Exception as exc:
                msg = f"Variant '{getattr(var, 'name', '<unknown>')}' invalid: {exc}"
                raise ValueError(msg) from exc


    def add_variant(self, variant: LhdVariant) -> None:
        self.variants.append(variant)

    def get_variant(self, name: str) -> LhdVariant:
        for var in self.variants:
            if var.name == name:
                return var
        raise KeyError(f"Variant not found: {name}")

    def get_variant_maps(self, name: str) -> Tuple[Dict[str, AlgorithmBase], Dict[str, Callable[..., ActionBase]]]:
        """
        Fetches variant mappings and returns
        """
        var = self.get_variant(name)
        return dict(var.algorithm_map), dict(var.action_factory_map)

    def instantiate_lhd(self, variant_name: str, model, *, register_defaults: bool = True, **lhd_kwargs) -> LocalHealthDepartment:
        """
        Helper to build a local health department
        """
        alg_map, fac_map = self.get_variant_maps(variant_name)
        lhd = LocalHealthDepartment(
            model=model,
            register_defaults=register_defaults,
            algorithm_map=alg_map,
            action_factory_map=fac_map,
            **lhd_kwargs,
        )
        return lhd



def default_call_factory_builder(
    reduction: Optional[float] = None,    
    duration: Optional[int] = None,    
    call_cost: Optional[float] = None,    
    min_factor: float = 1e-6,) -> Callable[..., ActionBase]: 
    """
    Returns a factory callable with signature:      
        factory(nodes, contact_type, prio, cost, params) -> ActionBase
    """

    def factory(nodes, contact_type, prio, cost, params=None):
        # Use provided parameters if any, otherwise, use builder params
        red = None
        dur = None
        ccost = None
        if params and isinstance(params, dict):
            red = params.get("reduction", reduction)
            dur = params.get("duration", duration)
            ccost = params.get("call_cost", call_cost)
        else:
            red = reduction
            dur = duration
            ccost = call_cost

        reduction_val = float(red) if red is not None else 0.0
        duration_val = int(dur) if dur is not None else 0
        # choose cost: candidate cost (cost arg) preferred, otherwise ccost, otherwise small default
        call_cost_final = float(cost) if (cost is not None and not (isinstance(cost, float) and (np.isnan(cost)))) else (float(ccost) if ccost is not None else 0.1)

        # If contact_type is None, default to casual+school+workplace as used elsewhere
        contact_types_list = [contact_type] if contact_type is not None else ["cas", "sch", "wp"]

        return CallIndividualsAction(
            nodes=nodes,
            contact_types=contact_types_list,
            reduction=reduction_val,
            duration=duration_val,
            call_cost=call_cost_final,
            min_factor=min_factor,
        )

    return factory

def validate_variant(variant: LhdVariant) -> None:
    """
    Helper to validate variant maps and fail if formatted incorrectly
    """
    algorithm_map = variant.algorithm_map
    action_factory_map=variant.action_factory_map
    for k, map in algorithm_map.items():
        if not isinstance(map, AlgorithmBase):
            raise TypeError(f"algorithm_map[{k}] is not an AlgorithmBase instance: {type(map)}")
    for k, fac in action_factory_map.items():
        if not callable(fac):
            raise TypeError(f"action_factory_map[{k}] is not callable: {type(fac)}")

"""
To write variants:

- Select algorithms 
- Select an action factory (default call for calling, etc.) and call params
- 
"""



random_alg_map = {"call": RandomPriority()}
random_fac_map = {"call": default_call_factory_builder(reduction = 0.5, duration = 7, call_cost = 0.1)}
random = LhdVariant(name="random_priority", algorithm_map = random_alg_map, action_factory_map = random_fac_map, description="random priority, 50% reduction")

elder_alg_map = {"call": PrioritizeElders(base_priority=1.0, elder_boost=5.0, elder_cost=0.15)}
elder_fac_map = {"call": default_call_factory_builder(reduction = 0.6, duration = 10, call_cost = 0.15)}
elder = LhdVariant(name="elder_priority", algorithm_map = elder_alg_map, action_factory_map = elder_fac_map)


cfg = LhdConfig(variants=[random, elder])
