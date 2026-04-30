import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple, Union

from aido.logger import logger

@dataclass
class SimulationConfig:
    number_simulations_per_particle: int = 1000
    number_stitched_events: int = 1000
    mean_number_particles_per_event: int = 5
    min_distance_cut : float = 0.
    minEnergy_GeV: float = 5.
    maxEnergy_GeV: float = 100.
    particle_types: Tuple[str] = (
        "e+",
        "gamma",
        "pi0",
        #"pi+",
        #"proton",
        #"neutron",
    )

@dataclass
class TimingConfig:
    enabled: bool = True
    runs : int = 1
    device : str = 'cpu'
    #device : str = 'cuda:2'

@dataclass
class GraphConfig:
    device: str = 'cuda:2'
    retrain: bool = False
    n_epochs: Tuple[int] = (40,20,10)
    n_batches: Tuple[int] = (500,500,500)
    batch_sizes: Tuple[int] = (256,256,256)
    lrs: Tuple[float] = (1e-3,1e-4,1e-5)
    annealing: Tuple[float] = (20,1.)
    pretrain: Tuple[bool] = (True,False,False)
    inputs: Tuple[str] = (
        "pos",
        "E",
        "layer",
        "cell",
    )
    regression: Tuple[str] = (
        "E",
        "pos",
    )
    classification: Tuple[str] = (
        "id",
    )
    loss_factors: Dict[str, float] = field(
        default_factory = lambda : {
            "attraction" : 1.0,
            "repulsion"  : 1.0,
            "beta"       : 1.0,
            "E"          : 1.0,
            "pos"        : 1.0,
            "id"         : 1.0,
        }
    )
    t_beta: float = 0.1
    t_dist: str = "auto"

@dataclass
class CaloConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)

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
            timing = TimingConfig(**data["timing"]),
        )

    def to_json(self, file_path: str):
        with open(file_path, "w") as file:
            json.dump(self.as_dict(), file, indent=4)

    def as_dict(self) -> Dict:
        return asdict(self)
