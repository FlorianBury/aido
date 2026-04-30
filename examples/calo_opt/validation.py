from typing import Union

import sys
from copy import deepcopy
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("agg")

def number_plot(ax,data_list):
    N_true = [data['particles']['pos'].shape[0] for data in data_list]
    N_reco = [data['vertices']['idx'].shape[0] for data in data_list]
    labels = np.arange(max(max(N_true),max(N_reco)))
    cm = confusion_matrix(N_true, N_reco)
    cm = confusion_matrix(N_true, N_reco, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot(
        ax = ax,
        cmap = plt.cm.Blues,
        colorbar = True,
        values_format = 'd',
        text_kw = {'fontsize': 6},
    )
    ax.invert_yaxis()
    ax.set_ylabel('True number of particles')
    ax.set_xlabel('Reco number of particles')

def energy_plot(ax,data_list):
    E_true = np.array([data['particles']['E'].sum() for data in data_list])
    E_reco = np.array([data['vertices']['E'].sum() for data in data_list])

    bins = np.linspace(
        min(E_true.min(),E_reco.min()),
        max(E_true.max(),E_reco.max()),
        51,
    )
    H = ax.hist2d(
        E_true,
        E_reco,
        bins = bins,
        norm = matplotlib.colors.LogNorm(),
    )
    plt.colorbar(H[3],ax=ax)
    ax.set_xlabel(r'Reco $\sum E_{\text{particles}}$')
    ax.set_ylabel(r'True $\sum E_{\text{particles}}$')

def id_plot(ax,data_list):
    id_true = np.array([
        val
        for data in data_list
        for val in data['particles']['id'].argmax(dim=-1)
    ])
    id_reco = np.array([
        val
        for data in data_list
        for val in data['vertices']['id'].argmax(dim=-1)
    ])

    bins = np.arange(max(id_true.max(),id_reco.max())+2)
    ax.hist(
        id_true,
        bins = bins,
        color = 'green',
        histtype = 'step',
        label = 'True',
    )
    ax.hist(
        id_reco,
        bins = bins,
        color = 'darkred',
        histtype = 'step',
        label = 'Reco',
    )
    ax.legend()
    ax.set_ylim(0,None)
    ax.set_xlabel(r'Ids')


def validation_plot(
    dataset,
    fig_savepath,
):
    fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(3*6,4))
    plt.subplots_adjust(wspace=0.4)

    data_list = [dataset[i] for i in range(len(dataset))]

    number_plot(axs[0],data_list)
    if 'E' in data_list[0]["vertices"].keys():
        energy_plot(axs[1],data_list)
    if 'id' in data_list[0]["vertices"].keys():
        id_plot(axs[2],data_list)


    if fig_savepath is not None:
        plt.savefig(fig_savepath,dpi=600)
        plt.close()
        print(f"Validation Plots Saved to '{fig_savepath}'")
        return None
    else:
        return fig, axs

if __name__ == "__main__":
    from dataset import CaloGraphDataset
    dataset = CaloGraphDataset.load(sys.argv[1])
    validation_plot(dataset,sys.argv[2])



#class ReconstructionValidation():
#    def __init__(
#        self,
#        model,
#        t_beta,
#        t_dist,
#    ):
#        self.model = model
#        self.t_beta = t_beta
#        self.t_dist = t_dist
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#    def validate(
#        self,
#        dataset,
#    ):
#        return
#        dataset =
#        val_result, val_loss, _ = self.reco_model.apply_model_in_batches(val_dataset, batch_size=batch_size)
#        output_mf = deepcopy(val_dataset.mf)
#        del output_mf._data['Inputs']
#
#        reco_mf = MiniFrame({"true_energy": val_result})
#        output_mf.add('Reconstructed',reco_mf)
#        output_mf.add('Loss',MiniFrame({'Reco_loss':val_loss}))
#        return output_mf
#
#    @classmethod
#    def plot(cls, validation_df: pd.DataFrame, fig_savepath: Union[str, None]) -> None:
#
#        reco = validation_df["Reconstructed"]["true_energy"]
#        true = validation_df["Targets"]["true_energy"]
#
#        fig, axs = plt.subplots(ncols=2,figsize=(9,4))
#        bins = np.linspace(
#            min(reco.min(),true.min()),
#            max(reco.max(),true.max()),
#            40 + 1,
#        )
#
#        axs[0].hist(true, bins=bins, label=r"$E_\text{true}$" + " (Simulation)", histtype="step", color="green")
#        axs[0].hist(reco, bins=bins, label=r"$E_\text{reco}$" + " (Reconstruction)", histtype="step", color="blue")
#        axs[0].set_xlabel("Energy [GeV]")
#        axs[0].set_ylabel(f"Counts / ({(bins[1] - bins[0]):.2f} GeV)")
#        axs[0].set_yscale('log')
#        y_min,y_max = axs[0].get_ylim()
#        axs[0].set_ylim(1e-1,y_max*20)
#        axs[0].legend()
#
#        h = axs[1].hist2d(
#            true,
#            reco,
#            bins = bins,
#            norm = matplotlib.colors.LogNorm(vmin=1),
#        )
#        axs[1].set_xlabel(r"$E_\text{true}$" + " (Simulation) GeV")
#        axs[1].set_ylabel(r"$E_\text{reco}$" + " (Reconstruction) GeV")
#        fig.colorbar(h[3], ax=axs[1])
#        plt.tight_layout()
#
#        if fig_savepath is not None:
#            plt.savefig(fig_savepath,dpi=600)
#            plt.close()
#
#            print(f"Validation Plots Saved to '{fig_savepath}'")
#            return None
#        else:
#            return fig, axs, bins
