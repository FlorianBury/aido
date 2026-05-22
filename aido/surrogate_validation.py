"""
Generate plots to validate the surrogate model for the example "full_calorimeter"
"""
import os
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import torch

class SurrogateValidation():
    def __init__(
            self,
            surrogate_model,
            ):
        self.surrogate_model = surrogate_model

    def validate(
            self,
            dataset,
            batch_size: int = 512,
            ) -> pd.DataFrame:
        validation_df = dataset.df
        surrogate_reconstructed_array = self.surrogate_model.apply_model_in_batches(dataset,batch_size)
        if dataset.reconstruction:
            validation_df[("Surrogate","true_energy")] = surrogate_reconstructed_array[:,0]
            idx = 1
        else:
            idx = 0
        if dataset.classification:
            for column in validation_df["Reconstructed"].columns:
                if 'true_logits' in column:
                    validation_df[("Surrogate",column)] = surrogate_reconstructed_array[:,idx]
                    idx += 1
        return validation_df

    @classmethod
    def plot(
        cls,
        validation_df: pd.DataFrame,
        fig_savepath: Union[os.PathLike, str],
        reconstruction_loss_function = None,
        classification_loss_function = None,
    ) -> None:
        """ Plot the reconstructed 'true_energy'
        """
        if fig_savepath is not None:
            os.makedirs(os.path.dirname(fig_savepath), exist_ok=True)

        ncols = len(validation_df['Surrogate'].columns)
        nrows = 3
        if reconstruction_loss_function is not None or classification_loss_function is not None:
            nrows = 4
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(ncols*5,nrows*4))
        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        if not isinstance(axs,np.ndarray):
            axs = np.array([[axs]])
        if ncols == 1:
            axs = axs.reshape(-1,1)

        if 'true_energy' in validation_df['Surrogate'].columns:
            idx_first_plot = 1
            # E comparison #
            true_energy = validation_df["Targets"]["true_energy"].values
            validation_energy = validation_df["Reconstructed"]["true_energy"].values
            surrogate_energy = validation_df["Surrogate"]["true_energy"].values

            bins = np.linspace(0, max([true_energy.max(),validation_energy.max(),surrogate_energy.max()]), 50 + 1)
            axs[0,0].hist(
                [validation_energy, surrogate_energy, true_energy],
                bins = bins,
                label = [
                    r"$E_\text{reco}$" + " (Validation)",
                    r"$E'$" + " (Surrogate)",
                    r"$E_\text{true}$" + " (Simulation)",
                ],
                color = ["orange", "royalblue", "forestgreen"],
                histtype = "step",
            )
            axs[0,0].legend()
            axs[0,0].set_xlabel("Initial Energy [GeV]")
            axs[0,0].set_ylabel(f"Counts / ({(bins[1] - bins[0]):.2f} GeV)")
            axs[0,0].set_yscale('log')
            ylim_min,ylim_max = axs[0,0].get_ylim()
            axs[0,0].set_ylim(1e-1,ylim_max*20)

            # E/E_true comparison #
            validation_ratio = validation_energy / true_energy
            surrogate_ratio = surrogate_energy / true_energy
            ratio_bins = np.linspace(0, max([validation_ratio.max(),surrogate_ratio.max()]), 40 + 1)
            axs[1,0].hist(
                [validation_ratio, surrogate_ratio],
                bins = ratio_bins,
                label = [
                    r"$E_\text{reco}$" + " (Validation)",
                    r"$E'$" + " (Surrogate)",
                ],
                color = ["orange", "royalblue"],
                histtype = "step",
            )
            axs[1,0].legend()
            axs[1,0].set_xlabel(r"$\frac{E_{reco}}{E_{true}}$")
            axs[1,0].set_ylabel(f"Counts / ({(bins[1] - bins[0]):.2f})")
            axs[1,0].set_yscale('log')
            ylim_min,ylim_max = axs[1,0].get_ylim()
            axs[1,0].set_ylim(1e-1,ylim_max*20)

            # 2D comparison
            H = axs[2,0].hist2d(
                validation_energy,
                surrogate_energy,
                bins = bins,
                norm = matplotlib.colors.LogNorm(vmin=1e-1),
            )
            axs[2,0].plot(
                [bins[0],bins[-1]],
                [bins[0],bins[-1]],
                linestyle = '--',
                color = 'r',
            )
            axs[2,0].set_xlabel('Validation energy')
            axs[2,0].set_ylabel('Surrogate energy')
            plt.colorbar(H[3],ax=axs[2,0])

            if reconstruction_loss_function is not None:
                validation_loss = reconstruction_loss_function(
                    torch.tensor(true_energy),
                    torch.tensor(validation_energy),
                )
                surrogate_loss = reconstruction_loss_function(
                    torch.tensor(true_energy),
                    torch.tensor(surrogate_energy),
                )
                bins = np.logspace(
                    np.log10(min(validation_loss.min(),surrogate_loss.min())),
                    np.log10(max(validation_loss.max(),surrogate_loss.max())),
                    50,
                )
                H = axs[3,0].hist2d(
                    validation_loss,
                    surrogate_loss,
                    bins = bins,
                    norm = matplotlib.colors.LogNorm(vmin=1e-1),
                )
                axs[3,0].plot(
                    [bins[0],bins[-1]],
                    [bins[0],bins[-1]],
                    linestyle = '--',
                    color = 'r',
                )
                axs[3,0].set_xscale('log')
                axs[3,0].set_yscale('log')
                axs[3,0].set_xlabel('Validation reco loss')
                axs[3,0].set_ylabel('Surrogate reco loss')
                plt.colorbar(H[3],ax=axs[3,0])
        else:
            idx_first_plot = 0


        if 'true_logits_0' in validation_df['Surrogate'].columns:
            columns = list(validation_df["Classes"].columns)
            for i in range(ncols-idx_first_plot):
                j = i + idx_first_plot
                name = columns[i].replace('contains:','')
                true_class = validation_df["Classes"][f"{columns[i]}"].values * 1
                validation_logits = validation_df["Reconstructed"][f"true_logits_{i}"].values
                surrogate_logits = validation_df["Surrogate"][f"true_logits_{i}"].values
                bins = np.linspace(
                    min([validation_logits.min(),surrogate_logits.min()]),
                    max([validation_logits.max(),surrogate_logits.max()]),
                    50 + 1,
                )
                axs[0,j].hist(
                    [validation_logits[true_class==0], surrogate_logits[true_class==0]],
                    bins=bins,
                    color=["orange", "royalblue"],
                        histtype="step",
                        linestyle = 'dashed',
                    )
                axs[0,j].hist(
                    [validation_logits[true_class==1], surrogate_logits[true_class==1]],
                    bins=bins,
                    color=["orange", "royalblue"],
                    histtype="step",
                    linestyle = 'dotted',
                )
                axs[0,j].plot(
                    [],[],
                    color="orange",
                    label=f"Class {name} (Validation)",
                    linestyle = 'solid',
                )
                axs[0,j].plot(
                    [],[],
                    color="royalblue",
                    label=f"Class {name} (Surrogate)",
                    linestyle = 'solid',
                )
                axs[0,j].plot(
                    [],[],
                    color="black",
                    label=f"Class = 0",
                    linestyle = 'dashed',
                )
                axs[0,j].plot(
                    [],[],
                    color="black",
                    label=f"Class = 1",
                    linestyle = 'dotted',
                )
                axs[0,j].legend()
                axs[0,j].set_xlabel(f"Classification logits (class {name})")
                axs[0,j].set_ylabel(f"Counts / ({(bins[1] - bins[0]):.2f})")
                axs[0,j].set_yscale('log')
                ylim_min,ylim_max = axs[0,j].get_ylim()
                axs[0,j].set_ylim(1e-1,ylim_max*20)

                val_fpr, val_tpr, _= roc_curve(true_class,validation_logits)
                sur_fpr, sur_tpr, _= roc_curve(true_class,surrogate_logits)

                axs[1,j].plot(
                    val_tpr,
                    val_fpr,
                    color = 'orange',
                    label = f'Validation (AUC = {auc(val_fpr,val_tpr):.5f})',
                )
                axs[1,j].plot(
                    sur_tpr,
                    sur_fpr,
                    color = 'royalblue',
                    label = f'Surrogate (AUC = {auc(sur_fpr,sur_tpr):.5f})',
                )
                axs[1,j].plot(
                    np.linspace(0,1,100),
                    np.linspace(0,1,100),
                    linestyle = 'dashed',
                    color = 'grey',
                    label = 'Random classifier',
                )
                axs[1,j].set_xlabel('TPR')
                axs[1,j].set_ylabel('FPR')
                axs[1,j].legend()
                axs[1,j].set_xlim(0,1)
                axs[1,j].set_ylim(0,1)
                axs[1,j].set_yscale('symlog',linthresh=1e-4)

                # 2D comparison
                H = axs[2,j].hist2d(
                    validation_logits,
                    surrogate_logits,
                    bins = bins,
                    norm = matplotlib.colors.LogNorm(vmin=1e-1),
                )
                axs[2,j].plot(
                    [bins[0],bins[-1]],
                    [bins[0],bins[-1]],
                    linestyle = '--',
                    color = 'r',
                )
                axs[2,j].set_xlabel(f'Validation logits (class {name})')
                axs[2,j].set_ylabel(f'Surrogate logits (class {name})')
                plt.colorbar(H[3],ax=axs[2,j])

            if classification_loss_function is not None:
                true_classes = torch.tensor(validation_df["Classes"][columns].values * 1.)
                validation_logits = torch.tensor(validation_df["Reconstructed"][
                    [
                        f"true_logits_{i}"
                        for i in range(len(columns))
                    ]
                ].values)
                surrogate_logits = torch.tensor(
                    validation_df["Surrogate"][
                    [
                        f"true_logits_{i}"
                        for i in range(len(columns))
                    ]
                ].values)
                validation_loss = classification_loss_function(
                    true_classes,
                    validation_logits,
                    multiclass = False,
                )
                surrogate_loss = classification_loss_function(
                    true_classes,
                    surrogate_logits,
                    multiclass = False,
                )
                bins = np.logspace(
                    np.log10(
                        min(
                            validation_loss[validation_loss>0].min(),
                            surrogate_loss[surrogate_loss>0].min(),
                        )),
                    np.log10(max(validation_loss.max(),surrogate_loss.max())),
                    50,
                )
                H = axs[3,1].hist2d(
                    validation_loss.ravel(),
                    surrogate_loss.ravel(),
                    bins = bins,
                    norm = matplotlib.colors.LogNorm(vmin=1e-1),
                )
                axs[3,1].plot(
                    [bins[0],bins[-1]],
                    [bins[0],bins[-1]],
                    linestyle = '--',
                    color = 'r',
                )
                axs[3,1].set_xscale('log')
                axs[3,1].set_yscale('log')
                axs[3,1].set_xlabel('Validation class loss')
                axs[3,1].set_ylabel('Surrogate class loss')
                plt.colorbar(H[3],ax=axs[3,1])




        plt.tight_layout()
        if fig_savepath is not None:
            plt.savefig(fig_savepath,dpi=600)
            print (f'Surrogate validation plot saved in {fig_savepath}')


    if __name__ == "__main__":
        logger.setLevel("DEBUG")
        results_dir: str = ...

        #for iteration in (5, 200):
        #    dataset_path = f"{results_dir}/task_outputs/iteration={iteration}/validation=True/validation_output_df"
        #    surrogate_model: Surrogate = torch.load(f"{results_dir}/models/surrogate_{iteration}.pt")
        #    validate_surrogate_func(
    #        surrogate=surrogate_model,
    #        validation_df_path=dataset_path,
    #        results_dir=f"{results_dir}",
    #    )
