import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

@dataclass
class ReconstructionConfig:
    retrain: bool = False
    momentum: float = 0.01
    n_epochs_pre: int = 50
    n_epochs_main: int = 50
    batch_size: int = 512
    early_stopping: int = 50
    lr_pre: float  = 1e-2
    lr_main: Tuple[float] = (5e-3, 1e-3, 5e-4)
    targets: Tuple[str] = (
        'true_energy',
    )

@dataclass
class ClassificationConfig:
    retrain: bool = False
    momentum: float = 0.01
    n_epochs_pre: int = 20
    n_epochs_main: int = 20
    batch_size: int = 512
    early_stopping: int = 20
    lr_pre: float  = 1e-3
    lr_main: Tuple[float] = (5e-4, 1e-4, 5e-5)
    classes: Tuple[str] = (
        'contains:e+',
        #'contains:gamma',
        #'contains:pi+',
        #'contains:proton',
    )
    multiclass: bool = False
    weight: Tuple[float] = (1.,)

@dataclass
class CaloConfig:
    reconstruction : ReconstructionConfig = field(default_factory=ReconstructionConfig)
    classification : ClassificationConfig = field(default_factory=ClassificationConfig)

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
            reconstruction = ReconstructionConfig(**data["reconstruction"]),
            classification = ClassificationConfig(**data["classification"]),
        )

    def to_json(self, file_path: str):
        with open(file_path, "w") as file:
            json.dump(self.as_dict(), file, indent=4)

    def as_dict(self) -> Dict:
        return asdict(self)
