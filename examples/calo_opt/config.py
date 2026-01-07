import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple, Union

from aido.logger import logger

@dataclass
class SimulationConfig:
    number_simulations_per_particle: int = 100
    number_stitched_events: int = 200
    mean_number_particles_per_event: int = 5
    min_distance_cut : float = 5.
    minEnergy_GeV: float = 5.
    maxEnergy_GeV: float = 100.
    particle_types: Tuple[str] = (
        "e+",
        "gamma",
        #"pi+",
        #"pi0",
        #"proton",
        #"neutron",
    )


@dataclass
class GraphConfig:
    retrain: bool = False
    inputs: Tuple[str] = (
        "pos",
        "E",
        "layer",
    )
    regression: Tuple[str] = (
        #"E",
        "pos",
    )
    classification: Tuple[str] = (
        #"id",
    )
    loss_factors: Dict[str, float] = field(
        default_factory = lambda : {
            "attraction" : 1.0,
            "repulsion"  : 1.0,
            "beta"       : 10.0,
        #    "E"          : 0.1,
            "pos"        : 10.0,
        #    "id"         : 0.1,
        }
    )
    t_beta: float = 0.1
    t_dist: float = 0.1


#    n_epochs_pre: int = 50
#    n_epochs_main: int = 50
#    batch_size: int = 512
#    early_stopping: int = 50
#    lr_pre: float  = 1e-2
#    lr_main: Tuple[float] = (5e-3, 1e-3, 5e-4)
#    targets: Tuple[str] = (
#        'true_energy',
#    )

@dataclass
class CaloConfig:
    graph : GraphConfig = field(default_factory=GraphConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    @classmethod
    def from_json(cls, file_path: str):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {file_path} not found. Using default configuration.")
            return cls()
        except Exception as e:
            logger.error(f"Error reading config file {file_path}. Using default configuration.\nError: {e}")
            return cls()

        return cls(
            graph = GraphConfig(**data["graph"]),
            simulation = SimulationConfig(**data["simulation"]),
        )

    def to_json(self, file_path: str):
        with open(file_path, "w") as file:
            json.dump(self.as_dict(), file, indent=4)

    def as_dict(self) -> Dict:
        return asdict(self)
