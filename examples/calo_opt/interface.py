import os
import re
import json
from typing import Dict, Iterable, List

import pandas as pd
import torch
#from calo_opt.reconstruction.model import Reconstruction
#from calo_opt.classification.model import Classification

import aido
from .config import CaloConfig

class CaloOptInterface(aido.UserInterfaceBase):
    """ This class is an example of how to implement the 'AIDOUserInterface' class.

    We use the following container:

        https://hub.docker.com/r/jkiesele/minicalosim

    Download the container and add the file path to the variable 'container_path'
    """

    htc_global_settings = {}
    container_path: str = ...  # place the path for the example container here
    container_extra_flags: str = ""  # place extra flags for singularity here
    verbose: bool = False

    def __init__(self,config):
        self._results_dir = None
        self.config = config

    @property
    def results_dir(self):
        return self._results_dir

    @results_dir.setter
    def results_dir(self,value):
        assert isinstance(value,str)
        assert os.path.exists(value)
        assert self._results_dir is None
        self._results_dir = value
        self.config.to_json(os.path.join(self._results_dir,'calo.json'))

    @property
    def suppress_output(self) -> str:
        return "> /dev/null 2>&1" if not self.verbose else ""

    def simulate(self, parameter_dict_path: str, sim_output_path: str):
        os.system(
            f"singularity exec {self.container_extra_flags} {self.container_path} python3 \
            examples/calo_opt/simulation.py {parameter_dict_path} {sim_output_path} {self.suppress_output}"
        )
        return None

    def merge(
            self,
            parameter_dict_file_paths: List[str],
            simulation_file_paths: List[str],
            reco_input_path: str
            ):
        """ Combines parameter dicts and pd.DataFrames into a large pd.DataFrame which is subsequently saved
        to parquet format.
        """
        if os.path.exists(reco_input_path):
            print (f'File {reco_input_path} already exists, will not merge')
            return None

        # First turn the parameter dict in dict of values saved for later miniframes #
        parameter_values_file_paths = []
        for parameter_dict_file_path in parameter_dict_file_paths:
            parameter_dict = aido.SimulationParameterDictionary.from_json(parameter_dict_file_path)
            parameter_values = parameter_dict.to_df(display_discrete="as_one_hot").iloc[0].to_dict()
            parameter_values_file_path = parameter_dict_file_path.replace('param_dict.json','param_dict_values.json')
            parameter_values_file_paths.append(parameter_values_file_path)
            with open(parameter_values_file_path,'w') as handle:
                json.dump(parameter_values,handle)

        # Calls the merge inside singularity container to use miniframe #
        os.system(
            f"singularity exec {self.container_extra_flags} {self.container_path} python3 \
            examples/calo_opt/merge.py {self.results_dir}/calo.json {reco_input_path} {' '.join(parameter_values_file_paths)} {' '.join(simulation_file_paths)}"
        )
        return None

    def reconstruct(self, reco_input_path: str, reco_output_path: str, is_validation: bool):
        """ Start your reconstruction algorithm from a local container.
        """
        assert self.results_dir is not None
        os.system(
            f"singularity exec --nv {self.container_extra_flags} {self.container_path} \
            python3 examples/calo_opt/train.py \
            {self.results_dir}/calo.json {reco_input_path} {reco_output_path} {is_validation} {self.results_dir}"
        )
        os.system("rm -f *.pkl")
        return None

    def reconstruction_loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return Reconstruction.loss(y, y_pred)

    def classification_loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return Classification.loss(y, y_pred, self.config.classification.multiclass, torch.tensor(self.config.classification.weight))
