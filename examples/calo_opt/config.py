import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from aido.logger import logger

@dataclass
class GraphConfig:
    retrain: bool = False
    inputs: Tuple[str] = (
        "pos",
        "E",
        "layer",
        "parameters",
    )
    reg_outputs : Tuple[str] = (
        "particle_E",
        "particle_pos",
    )
    cls_outputs : Tuple[str] = (
        "particle_type",
    )
    loss_factors: Dict[str, float] = field(
        default_factory = lambda : {
            "attraction"    : 1.0,
            "repulsion"     : 1.0,
            "beta"          : 1.0,
            "particle_E"    : 0.1,
            "particle_pos"  : 0.1,
            "particle_type" : 0.1,
        }
    )
    t_beta: float = 0.1
    t_dist: float = 0.5


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
            graph = ReconstructionConfig(**data["graph"]),
        )

    def to_json(self, file_path: str):
        with open(file_path, "w") as file:
            json.dump(self.as_dict(), file, indent=4)

    def as_dict(self) -> Dict:
        return asdict(self)
