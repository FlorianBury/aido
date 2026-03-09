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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import pandas as pd

import aido

matplotlib.use("agg")


class CaloOptPlotting:

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

    @staticmethod
    def mplstyle() -> None:
        plt.style.use(pathlib.Path(__file__).parent / "aido.mplstyle")

    @classmethod
    def add_plot_header(cls, ax: plt.Axes) -> plt.Axes:
        plt.text(
            0.0, 1.08,
            "AIDO",
            transform=ax.transAxes, fontsize=30, fontweight='bold', va='top', ha='left'
        )
        plt.text(
            0.13, 1.06,
            "Detector Optimization",
            transform=ax.transAxes, fontsize=20, style='italic', va='top', ha='left'
        )
        return ax

    @staticmethod
    def mask_parameters(df,param_dict):
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


    def plot(self,savefig):

        fig = plt.figure(figsize=(15,25))
        gs = GridSpec(
            nrows = 5, ncols = 1,
            figure = fig,
            left = 0.1, bottom = 0.1, right = 0.9, top = 0.9,
            hspace = 0.2,
            wspace = 0.3,
            height_ratios=[3, 2, 1, 1, 1],
        )
        ax0 = fig.add_subplot(gs[0])
        gs_bottom = gs[1:].subgridspec(
            4, 2,
            width_ratios=[1.0, 0.2],  # second column = empty margin
            height_ratios=[2, 1, 1, 1],
            wspace=0,
            hspace = 0.1,
        )
        ax1 = fig.add_subplot(gs_bottom[0,0])
        ax2 = fig.add_subplot(gs_bottom[1,0])
        ax3 = fig.add_subplot(gs_bottom[2,0])
        ax4 = fig.add_subplot(gs_bottom[3,0])

        # Calo plot #
        df_list = []
        df_materials_list = []
        parameter_dir = os.path.join(self.results_dir, "parameters/")

        for file_name in os.listdir(parameter_dir):
            param_dict = aido.SimulationParameterDictionary.from_json(parameter_dir + file_name)
            df_list.append(pd.DataFrame(
                param_dict.get_current_values(format="dict", types="continuous"),
                index=[param_dict.iteration],
            ))
            df_materials = pd.DataFrame(param_dict.get_probabilities()).drop(index=0)
            df_materials.index = [param_dict.iteration]
            df_materials_list.append(df_materials)

        df: pd.DataFrame = pd.concat(df_list, axis=0).sort_index()
        df_materials: pd.DataFrame = pd.concat(df_materials_list, axis=0).sort_index()
        df_materials.columns = df.columns

        ax0 = self.add_plot_header(ax0)
        absorber_cmap = mcolors.LinearSegmentedColormap.from_list("blue_grey", ["blue", "grey"])
        scintillator_cmap = plt.get_cmap("spring")

        def get_color(label: str, prob: Iterable):
            if "absorber" in label:
                return absorber_cmap(prob)
            if "scintillator" in label:
                return scintillator_cmap(prob)
            else:
                return "white"

        for i in df.index:
            bottom = 0
            ax0.set_prop_cycle(None)

            for column in df.columns:
                ax0.bar(
                    i,
                    df[column][i],
                    bottom=bottom,
                    color=get_color(column, df_materials[column][i]),
                    width=1,
                    align="edge",
                    label=column.replace("_", " ").removeprefix("thickness ").capitalize(),
                )
                bottom += df[column][i]

        ax0.set_ylabel("Longitudinal Composition [cm]",fontsize=20)
        ax0.set_xlim(0, len(df)-1)
        ax0.set_ylim(0, 140)
        cbar_absorber = plt.cm.ScalarMappable(cmap=absorber_cmap)
        cbar_absorber.set_array([])
        cbar1 = plt.colorbar(
            cbar_absorber,
            ax=ax0,
            fraction=0.04,
            location="right",
        )
        cbar1.ax.invert_yaxis()  # Invert so Fe is high and Pb is low
        cbar1.ax.set_yticks([0, 1], labels=['Pb', 'Fe'], rotation=90, va='center',fontsize=20)  # Custom ticks

        cbar_scintillator = plt.cm.ScalarMappable(cmap=scintillator_cmap)
        cbar_scintillator.set_array([])
        cbar2 = plt.colorbar(
            cbar_scintillator,
            ax=ax0,
            fraction=0.04,
            location="right",
        )
        cbar2.ax.invert_yaxis()
        cbar2.ax.set_yticks([0, 1], labels=['PbWO4', 'Polystyrene'], rotation=90, va='center',fontsize=20)  # Custom ticks


        # Loss plot #
        reco_losses = []
        class_losses = []
        reco_losses_best = []
        class_losses_best = []
        iterations = sorted(list(self.reco_output_paths.keys()))

        for iteration in iterations:
            print ('loss',iteration)
            df = pd.read_parquet(self.reco_output_paths[iteration])
            reco_losses.append([math.inf,-math.inf])
            class_losses.append([math.inf,-math.inf])
            for param_dict_json in self.simulation_parameter_paths[iteration]:
                param_dict = aido.SimulationParameterDictionary.from_json(param_dict_json)
                idx = self.mask_parameters(df['Parameters'],param_dict)
                reco_loss = np.mean(df["Loss"]["Reco_loss"][idx])
                class_loss = np.mean(df["Loss"]["Class_loss"][idx])
                if reco_loss < reco_losses[-1][0]:
                    reco_losses[-1][0] = reco_loss
                if reco_loss > reco_losses[-1][1]:
                    reco_losses[-1][1] = reco_loss
                if class_loss < class_losses[-1][0]:
                    class_losses[-1][0] = class_loss
                if class_loss > class_losses[-1][1]:
                    class_losses[-1][1] = class_loss
            param_dict = aido.SimulationParameterDictionary.from_json(self.optimizer_parameter_paths[iteration])
            idx = self.mask_parameters(df['Parameters'],param_dict)
            reco_losses_best.append(np.mean(df["Loss"]["Reco_loss"][idx]))
            class_losses_best.append(np.mean(df["Loss"]["Class_loss"][idx]))

        iterations = np.array(iterations)
        order = iterations.argsort()
        iterations = iterations[order]
        reco_losses = np.array(reco_losses)[order]
        class_losses = np.array(class_losses)[order]
        reco_losses_best = np.array(reco_losses_best)[order]
        class_losses_best = np.array(class_losses_best)[order]

        df_loss: pd.DataFrame = aido.Plotting.optimizer_loss(results_dir=self.results_dir)
        df_loss = df_loss[["Scaled Epoch", "Loss"]]

        ax1.fill_between(
            iterations,
            reco_losses[:,0],
            reco_losses[:,1],
            interpolate = True,
            color = 'royalblue',
            alpha = 0.3,
            label="Regression\n"+r"($\mathcal{L}_\text{reg}$)",
        )
        ax1.plot(
            iterations,
            reco_losses_best,
            color = 'royalblue',
            linestyle = 'dashed',
        )
        ax1.fill_between(
            iterations,
            class_losses[:,0],
            class_losses[:,1],
            interpolate = True,
            color = 'darkred',
            alpha = 0.3,
            label="Classification\n"+r"($\mathcal{L}_\text{class}$)",
        )
        ax1.plot(
            iterations,
            class_losses_best,
            color = 'red',
            linestyle = 'dashed',
        )
        ax1.plot(
            df_loss["Scaled Epoch"],
            df_loss["Loss"],
            color = 'forestgreen',
            label="Optimizer\n"+r"($\mathcal{L}_{\text{tot}}$)"
        )
        ax1.fill_between(
            [],
            [],[],
            color = 'grey',
            alpha = 0.3,
            label = 'Sampled parameters\n(current iteration)',
        )
        ax1.plot(
            [],[],
            color = 'black',
            linestyle = 'dashed',
            label = 'Best parameters\n(previous iteration)',
        )
        ax1.legend(
            loc="center left",
            bbox_to_anchor=(1., 0.5),
            fontsize=18,
        )
        ax1.set_ylabel("Loss",fontsize=20)
        ax1.set_xlim(
            min(iterations.min(),df_loss["Scaled Epoch"].min()),
            max(iterations.max(),df_loss["Scaled Epoch"].max()),
        )
        ax1.set_ylim(
            min(
                [
                    reco_losses.min(),
                    class_losses.min(),
                    reco_losses_best.min(),
                    class_losses_best.min(),
                    df_loss["Loss"].min(),
                ]
            ) / 2,
            max(
                [
                    reco_losses.max(),
                    class_losses.max(),
                    reco_losses_best.max(),
                    class_losses_best.max(),
                    df_loss["Loss"].max(),
                ]
            ) * 2,
        )
        ax1.set_yscale("log")

        # trade-off parameter #
        def sigmoid_turn_on(x,yi,yf,k,T):
            return yi + (yf-yi) * 1 / (1+np.exp(-k*(x-T)))

        turn_on_reco = (1.,1.,1.,30)
        turn_on_class = (0.,1,0.5,30)

        lam = []
        for iteration in iterations:
            lam.append(sigmoid_turn_on(iteration,*turn_on_class))
        ax2.plot(
            iterations,
            lam,
            linewidth = 2,
            color = 'forestgreen',
        )
        ax2.set_ylabel(r'$\lambda$',fontsize=20)
        ax2.text(
            x = 0.50,
            y = 0.35,
            s = r'$\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{reg}} + \lambda  \ \mathcal{L}_{\text{class}}$',
            fontsize = 30,
            transform=ax2.transAxes
        )
        ax2.set_xlim(iterations.min(),iterations.max())


        # Plot regression MSE + classification #
        mses = []
        aucs = []
        for iteration in iterations:
            print ('mse',iteration)
            df = pd.read_parquet(self.reco_output_paths[iteration])
            param_dict = aido.SimulationParameterDictionary.from_json(self.optimizer_parameter_paths[iteration])
            idx = self.mask_parameters(df['Parameters'],param_dict)
            mses.append(
                (abs(df["Targets"]["true_energy"] - df["Reconstructed"]["true_energy"])/np.sqrt(df["Targets"]["true_energy"]))[idx].mean()
            )
            #mses.append(
            #    ((df["Targets"]["true_energy"] - df["Reconstructed"]["true_energy"])**2)[idx].mean()
            #)
            class_name = df["Classes"].columns[0]
            fpr,tpr,_ = roc_curve(
                df["Classes"][class_name][idx],
                df["Reconstructed"]["true_logits_0"][idx],
            )
            aucs.append(auc(fpr,tpr))
        iterations = np.array(iterations)
        order = iterations.argsort()
        mses = np.array(mses)[order]
        aucs = np.array(aucs)[order]

        ax3.plot(
            iterations,
            mses,
            linewidth = 2,
            color = 'darkred',
        )
        ax3.set_ylabel(r"$\frac{\vert E_\text{rec} - E_\text{true} \vert}{\sqrt{E_\text{true}}}$",fontsize=20)
        ax3.set_xlim(iterations.min(),iterations.max())
        #ax3.set_yscale('log')
        ax3.set_ylim(mses.min(),mses.max())

        ax4.plot(
            iterations,
            aucs,
            linewidth = 2,
            color = 'royalblue',
        )
        ax4.set_ylabel("ROC AUC",fontsize=20)
        ax4.set_xlim(iterations.min(),iterations.max())

        ax4.set_xlabel("Iteration",fontsize=20)

        # Save fig #
        fig.savefig(savefig)



if __name__ == "__main__":
    results_dir: str = sys.argv[1]

    plotter = CaloOptPlotting(results_dir)
    plotter.mplstyle()
    plotter.plot(sys.argv[2])
