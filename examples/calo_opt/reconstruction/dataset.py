"""
Dataset for the Reconstruction model. Based on pytorch
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ReconstructionDataset(Dataset):
    def __init__(
        self,
        input_df: pd.DataFrame,
        means: Optional[List[np.float32]] = None,
        stds: Optional[List[np.float32]] = None
    ):
        """Convert the files from the simulation to simple lists.

        Args:
            input_df (pd.DataFrame): Must contain as first level columns:
                ["Parameters", "Inputs", "Targets", "Context"], the names of the further dimensions
                are ignored.

        Returns:
            torch.DataSet
        """
        self.df = input_df
        self.df = self.filter_infs_and_nans(self.df)
        self.parameters = self.df["Parameters"].to_numpy("float32")
        self.inputs = self.df["Inputs"].to_numpy("float32")
        self.targets = self.df["Targets"].to_numpy("float32")
        if "Context" in self.df.columns:
            self.context = self.df["Context"].to_numpy("float32")
        else:
            self.context = torch.empty(self.inputs.shape[0],0)


        self.shape = (
            self.parameters.shape[1],
            self.inputs.shape[1],
            self.targets.shape[1],
            self.context.shape[1]
        )
        if means is None:
            self.means = [
                self.parameters.mean(axis=0),
                self.inputs.mean(axis=0),
                self.targets.mean(axis=0),
                self.context.mean(axis=0),
            ]
        else:
            self.means = means

        if stds is None:
            self.stds = [
                self.parameters.std(axis=0) + 1e-10,
                self.inputs.std(axis=0) + 1e-10,
                self.targets.std(axis=0) + 1e-10,
                self.context.std(axis=0) + 1e-10,
            ]
        else:
            self.stds = stds

        self.inputs = (self.inputs - self.means[1]) / self.stds[1]
        self.targets = (self.targets - self.means[2]) / self.stds[2]
        self.context = (self.context - self.means[3]) / self.stds[3]

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.c_means = [torch.tensor(a).to(dev) for a in self.means]
        self.c_stds = [torch.tensor(a).to(dev) for a in self.stds]

    def filter_infs_and_nans(self, df: pd.DataFrame):
        '''
        Removes all events that contain infs or nans.
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0, ignore_index=True)
        return df

    def filter_empty_events(self, df: pd.DataFrame):
        df = df[df["Inputs"]["sensor_energy_0"] > 0.0]
        df = df.dropna(axis=0, ignore_index=True)
        return df

    def unnormalize_target(self, target: torch.Tensor):
        return target * self.c_stds[2] + self.c_means[2]

    def normalize_target(self, target: torch.Tensor):
        return (target - self.c_means[2]) / self.c_stds[2]

    def unnormalize_detector(self, detector: torch.Tensor):
        return detector * self.c_stds[1] + self.c_means[1]

    def normalize_detector(self, detector: torch.Tensor):
        return (detector - self.c_means[1]) / self.c_stds[1]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.parameters[idx], self.inputs[idx], self.context[idx], self.targets[idx]
