from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .model import Classification, ClassificationDataset

matplotlib.use("agg")


class ClassificationValidation():
    """ Validate a given instance of the Classification model
    """
    def __init__(
            self,
            class_model: Classification,
            ):
        """
        Initializes the Validation object for a given Classification model.
        Args:
            class_model (Classification): The reconstruction model to be used.
        Attributes:
            device (torch.device): The device to be used for computation, either 'cuda' if a GPU is available or 'cpu'.
            """
        self.class_model = class_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def validate(
            self,
            val_dataset: ClassificationDataset,
            batch_size: int = 512,
            ) -> pd.DataFrame:
        """ Apply the Classification model on the validation dataset `val_dataset` and concatenate the
        results with that dataset, adding the columns ("Loss", "Reco_loss") and ("Reconstructed", "true_energy")
        as a multi-column index.
        Args:
            val_dataset (ClassificationDataset): A valid dataset trained on simulation data distinct from the
            original Classification model's training dataset.
            batch_size (int): Batch size for validation
        """
        val_result, val_loss, _ = self.class_model.apply_model_in_batches(val_dataset, batch_size=batch_size)

        validation_df = pd.DataFrame({
            f"true_logits_{i}": val_result[:,i]
            for i in range(val_result.shape[-1])
        })
        validation_df = pd.concat({"Reconstructed": validation_df}, axis=1)
        loss_df_val = pd.DataFrame({"Class_loss": val_loss.tolist()})
        loss_df_val = pd.concat({"Loss": loss_df_val}, axis=1)
        output_df_val: pd.DataFrame = pd.concat([val_dataset.df, validation_df, loss_df_val], axis=1)
        return output_df_val

    @classmethod
    def plot(cls, validation_df: pd.DataFrame, fig_savepath: Union[str, None]) -> None:

        columns = list(validation_df["Classes"].columns)
        names = [col.replace('contains:','') for col in columns]
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        reco = np.concatenate(
            [
                sigmoid(validation_df["Reconstructed"][f"true_logits_{i}"].values).reshape(-1,1)
                for i in range(len(columns))
            ]
        )
        true = np.concatenate(
            [
                validation_df["Classes"][col].values.reshape(-1,1)
                for col in columns
            ]
        )

        fig, axs = plt.subplots(ncols=2,nrows=len(names),figsize=(9,len(names)*4))
        if axs.ndim == 1:
            axs = axs.reshape(1,-1)
        bins = np.linspace(0, 1, 40 + 1)
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(names)))
        for i, name in enumerate(names):
            axs[i,0].hist(true[:,i], bins=bins, label=f"Class {name} (Simulation)", histtype="step", color='green')
            axs[i,0].hist(reco[:,i], bins=bins, label=f"Class {name} (Reconstruction)", histtype="step", color='blue')
            axs[i,0].set_xlabel("Class score")
            axs[i,0].set_ylabel(f"Counts / ({(bins[1] - bins[0]):.2f})")
            axs[i,0].set_yscale('log')
            y_min,y_max = axs[i,0].get_ylim()
            axs[i,0].set_ylim(1e-1,y_max*20)
            axs[i,0].legend()

            h = axs[i,1].hist2d(
                true[:,i],
                reco[:,i],
                bins = bins,
                norm = matplotlib.colors.LogNorm(vmin=1),
            )
            axs[i,1].set_xlabel("Class score (Simulation)")
            axs[i,1].set_ylabel("Class score (Reconstruction)")
            fig.colorbar(h[3], ax=axs[i,1])
            plt.tight_layout()

        if fig_savepath is not None:
            plt.savefig(fig_savepath)
            plt.close()
            print(f"Validation Plots Saved to '{fig_savepath}'")
            return None
        else:
            return fig, axs, bins

