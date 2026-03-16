from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import json


@dataclass(frozen=True)
class EpiParams:
    base_transmission_prob: float = 0.5
    incubation_period: float = 10.5
    infectious_period: float = 5.0
    gamma_alpha: float = 20.0
    incubation_period_vax: float = 10.5
    infectious_period_vax: float = 5.0
    relative_infectiousness_vax: float = 0.05
    conferred_immunity_duration: Optional[float] = None
    lasting_partial_immunity: Optional[float] = None
    vax_efficacy: float = 0.3
    vax_uptake: float = 0.85
    susceptibility_multiplier_under_five: float = 1.0
    susceptibility_multiplier_elderly: float = 1.0


@dataclass(frozen=True)
class PopulationParams:
    wp_contacts: int = 10
    sch_contacts: int = 10
    gq_contacts: int = 10
    cas_contacts: int = 10  #up to master_casual_contacts
    hh_weight: float = 1.0
    wp_weight: float = 0.5
    sch_weight: float = 0.6
    gq_weight: float = 0.3
    cas_weight: float = 0.1


@dataclass(frozen=True)
class LHDParams:
    mean_compliance: float = 1.0
    lhd_employees: int = 10
    lhd_discovery_prob: float = 0.25
    lhd_workday_hrs: int = 8
    lhd_default_call_duration: float = 0.1
    lhd_default_int_reduction: float = 0.8
    lhd_default_int_duration: int = 10


@dataclass(frozen=True)
class SimulationParams:
    n_replicates: int = 50 #Stochastic Replicates
    run_name: str = field(default_factory=lambda: "RUN_" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
    overwrite_master: bool = True
    simulation_duration: int = 100
    I0: Union[List[int],int] = 5
    seed: int = 2026
    county: str = "Keweenaw"
    state: str = "Michigan"
    resample_network_per_run: bool = False
    master_casual_candidates: int = 100
    save_master: bool = True
    record_exposure_events: bool = True
    save_plots: bool = True
    save_data_files: bool = True
    make_movie: bool = False
    display_plots: bool = False
    save_run_timeseries: bool = True


@dataclass(frozen=True)
class ModelConfig:
    epi: EpiParams = field(default_factory=EpiParams)
    population: PopulationParams = field(default_factory=PopulationParams)
    lhd: LHDParams = field(default_factory=LHDParams)
    sim: SimulationParams = field(default_factory=SimulationParams)


    #convert ModelConfig to dict
    def to_dict(self) -> Dict[str, Any]:
        return {"epi": asdict(self.epi), "population": asdict(self.population), "lhd": asdict(self.lhd), "sim": asdict(self.sim)}

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    #Build model config object from a json (saved by to_json)
    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r") as fh:
            d = json.load(fh)
        return cls.from_nested_dict(d)

    #build from nested dict, also used by from_json
    @classmethod
    def from_nested_dict(cls, nested: Dict[str, Any]) -> "ModelConfig":
        """
        Build ModelConfig from nested dicts of dataclass structures
        Example:
            {"epi": {...}, "population": {...}, "lhd": {...}, "sim": {...}}
        """
        epi = EpiParams(**(nested.get("epi", {})))
        population = PopulationParams(**(nested.get("population", {})))
        lhd = LHDParams(**(nested.get("lhd", {})))
        sim = SimulationParams(**(nested.get("sim", {})))
        return cls(epi=epi, population=population, lhd=lhd, sim=sim)

    def copy_with(self, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> "ModelConfig":
        """
        Return a new ModelConfig with nested overrides.
        Example overrides: {"sim": {"n_replicates": 30}, "population": {"wp_contacts": 20}}
        """
        if not overrides:
            return self
        new_epi = self.epi
        new_population = self.population
        new_lhd = self.lhd
        new_sim = self.sim
        if "epi" in overrides:
            new_epi = replace(self.epi, **overrides["epi"])
        if "population" in overrides:
            new_population = replace(self.population, **overrides["population"])
        if "lhd" in overrides:
            new_lhd = replace(self.lhd, **overrides["lhd"])
        if "sim" in overrides:
            new_sim = replace(self.sim, **overrides["sim"])
        return ModelConfig(epi=new_epi, population=new_population, lhd=new_lhd, sim=new_sim)

    def validate(self) -> None:
        # Basic checks that configurations make sense
        if not (0.0 <= self.epi.base_transmission_prob <= 1.0):
            raise ValueError("epi.base_transmission_prob must be in [0,1]")
        if not (0.0 <= self.epi.vax_uptake <= 1.0):
            raise ValueError("epi.vax_uptake must be in [0,1]")
        if self.sim.n_replicates < 1:
            raise ValueError("sim.n_replicates must be >= 1")
        if self.lhd.lhd_employees < 0:
            raise ValueError("lhd.lhd_employees must be >= 0")
        if not isinstance(self.sim.I0, (list, int)):
            raise ValueError("sim.I0 must be an integer or list")
        if not (self.sim.master_casual_candidates >= self.population.cas_contacts):
            raise ValueError("sim.master_casual_candidates must exceed population.cas_contacts for sampling purposes")

