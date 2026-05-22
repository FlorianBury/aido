import os
import sys
import math
import glob
import pathlib
import re
from typing import Iterable
import torch
import torchmetrics

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import pandas as pd

import aido

matplotlib.use("agg")


class Pareto:
    def __init__(self, results_dir: str | os.PathLike) -> None:
        self.results_dir = results_dir
        self.reco_output_paths = {}
        for file_name in glob.glob(f"{results_dir}/task_outputs/iteration=*/validation=False/reco_output_df"):
            iteration = int(re.search(r"iteration=(\d+)", file_name).group(1))
            self.reco_output_paths[iteration] = file_name
        self.simulation_parameter_paths = {
            iteration : [
                json_file
                for json_file in glob.glob(
                    f"{results_dir}/task_outputs/iteration={iteration}/validation=False/simulation_task_id=*/param_dict.json"
                )
            ]
            for iteration in self.reco_output_paths.keys()
        }
        self.optimizer_parameter_paths = {
            iteration : f"{results_dir}/parameters/param_dict_iter_{iteration}.json"
            for iteration in self.reco_output_paths.keys()
        }

    def mask_parameters(self,df,param_dict):
        masks = []
        for param_name, param_value in param_dict.get_current_values(format="dict", types="continuous").items():
            masks.append(
                (abs(df[param_name] - param_value) < 1e-5).values
            )
        for param_name, param_value in param_dict.get_current_values(format="dict", types="discrete").items():
            masks.append(
                df[f'{param_name}_{param_value}'] == 1
            )
        idx = np.where(np.logical_and.reduce(masks))[0]
        assert len(idx) > 0
        return idx

    def plot(self,figpath=None):
        reco_losses = []
        class_losses = []
        iterations = sorted(list(self.reco_output_paths.keys()))
        for iteration in iterations:
            print (f'Iteration {iteration}/{len(iterations)}')
            df = pd.read_parquet(self.reco_output_paths[iteration])
            N = int(df.shape[0] * 0.8)
            #df = df.iloc[:N]
            #df = df.iloc[N:]
            param_dict = aido.SimulationParameterDictionary.from_json(self.optimizer_parameter_paths[iteration])
            idx = self.mask_parameters(df['Parameters'],param_dict)
            reco_losses.append(np.mean(df["Loss"]["Reco_loss"][idx]))
            class_losses.append(np.mean(df["Loss"]["Class_loss"][idx]))

        reco_losses = np.array(reco_losses)
        class_losses = np.array(class_losses)

        if figpath is not None:
            points = np.array([reco_losses, class_losses]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            idx = np.arange(len(reco_losses))

            lc = LineCollection(segments, cmap='viridis')
            lc.set_array(idx)

            fig,ax = plt.subplots(figsize=(6,5))
            ax.add_collection(lc)
            ax.autoscale()
            plt.colorbar(lc, label='Iteration')
            plt.xlabel('Reconstruction Loss')
            plt.ylabel('Classification Loss')
            plt.xscale('log')
            plt.yscale('log')
            fig.savefig(figpath,bbox_inches='tight',transparent=False)

        return reco_losses, class_losses


if __name__ == "__main__":
    results_dir: str = sys.argv[1]

    plt.style.use(pathlib.Path(__file__).parent / "aido.mplstyle")

    pareto = Pareto(results_dir)
    pareto.plot(os.path.join(results_dir,'plots','pareto.png'))
