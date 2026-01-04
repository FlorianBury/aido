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
from .dataset import CaloGraphDataset, concat_dataset
from .train import train

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
        if os.path.exists(reco_input_path):
            print ('Merged dataset already exists')
            return
        datasets = []
        for parameter_dict_file_path,simulation_file_path in zip(parameter_dict_file_paths,simulation_file_paths):
            parameter_dict = aido.SimulationParameterDictionary.from_json(parameter_dict_file_path)
            params = torch.from_numpy(parameter_dict.to_df(display_discrete="as_one_hot").iloc[0].values)

            dataset = CaloGraphDataset.load(simulation_file_path)
            param_dataset = CaloGraphDataset.from_data_list_and_parameters(
                [data for data in dataset],
                params,
            )
            datasets.append(param_dataset)
        dataset = concat_dataset(datasets)
        print (f'Merged dataset with {len(dataset)} events')
        dataset.save(reco_input_path)

        return None

    def reconstruct(self, reco_input_path: str, reco_output_path: str, is_validation: bool):
        """ Start your reconstruction algorithm from a local container.
        """
        train(
            config_path = f" {self.results_dir}/calo.json",
            input_graph_path = reco_input_path,
            output_graph_path = reco_output_path,
            isVal = is_validation,
            results_dir = self.results_dir,
        )
        return None

#    def reconstruction_loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
#        return Reconstruction.loss(y, y_pred)
#
#    def classification_loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
#        return Classification.loss(y, y_pred, self.config.classification.multiclass, torch.tensor(self.config.classification.weight))
