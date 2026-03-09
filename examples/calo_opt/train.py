# flake8: noqa: E402
import os
import re
import pathlib
import sys
from typing import Union
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(pathlib.Path(__file__).parent.parent))

from minipandas import MiniFrame
from merge import SuperMiniFrame, super_concat
from utils import LossPlotting, EarlyStopping
from config import CaloConfig, ReconstructionConfig, ClassificationConfig
from reconstruction.dataset import ReconstructionDataset
from reconstruction.model import Reconstruction
from reconstruction.validation_reconstruction import ReconstructionValidation
from classification.dataset import ClassificationDataset
from classification.dataset import FlipAugmentation, RotationAugmentation, ShiftAugmentation
from classification.model import Classification
from classification.validation_classification import ClassificationValidation


def pre_train(
    model: Reconstruction,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    n_epochs: int,
    batch_size: int,
    lr: float,
    plotter: LossPlotting,
    early_stopping: EarlyStopping = None,
):
    """ Pre-train the  a given model

    TODO Reconstruction results are normalized. In the future only expose the un-normalized ones,
    but also requires adjustments to the surrogate dataset
    """

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    print("Reconstruction: Pre-training")
    model.train_model(
        train_dataset,
        valid_dataset,
        batch_size = batch_size,
        n_epochs = n_epochs,
        lr = lr,
        plotter = plotter,
        early_stopping = early_stopping,
    )

    model.to('cpu')

def split(smf):
    train_smf = SuperMiniFrame()
    valid_smf = SuperMiniFrame()
    idx = np.arange(len(smf))
    train_idx, valid_idx = train_test_split(idx, test_size=0.2, shuffle=True)
    for superkey,mf in smf.items():
        train_mf = MiniFrame(
            {
                key: mf[key][train_idx]
                for key in mf.keys()
            }
        )
        valid_mf = MiniFrame(
            {
                key: mf[key][valid_idx]
                for key in mf.keys()
            }
        )
        train_smf.add(superkey,train_mf)
        valid_smf.add(superkey,valid_mf)
    return train_smf, valid_smf

def train(
    config_path: Union[str, os.PathLike],
    input_mf_path: Union[str, os.PathLike],
    output_df_path: Union[str, os.PathLike],
    isVal: bool,
    results_dir: Union[str, os.PathLike],
):
    config = CaloConfig.from_json(config_path)
    iteration = int(re.search(r"iteration=(\d+)", input_mf_path).group(1))
    print (f'Loading {input_mf_path}')
    simulation_mf: SuperMiniFrame = SuperMiniFrame.read_pickle(input_mf_path)
    print (f'Simulation dataframe : {len(simulation_mf)} events')
    train_mf, valid_mf = split(simulation_mf)
    print (f'Train dataset : {len(train_mf)} / Validation dataset : {len(valid_mf)}')

    if isVal:
        pass
        # Below has not been updated with latest changes
        #reco_model: Reconstruction = torch.load(os.path.join(results_dir, "reco_model"))
        #reco_dataset = ReconstructionDataset(simulation_mf, means=reco_model.means, stds=reco_model.stds)

        #validator = ReconstructionValidation(reco_model)
        #output_mf_val = validator.validate(reco_dataset)
        #output_mf_val.to_parquet(output_mf_path)
        #validator.plot(
        #    output_mf_val,
        #    os.path.join(results_dir, "plots", "validation", "reco_model", "on_validationData")
        #)
    else:
        # Reconstruction training:
        reco_model_previous_path = os.path.join(results_dir, "models","reco.pt")
        reco_loss_plotter = LossPlotting()
        reco_early_stopping = EarlyStopping(patience=config.reconstruction.early_stopping)

        print ('Creating reco dataset')
        reco_train_dataset = ReconstructionDataset(
            targets = config.reconstruction.targets,
            input_mf = train_mf,
        )
        reco_valid_dataset = ReconstructionDataset(
            targets = config.reconstruction.targets,
            input_mf = valid_mf,
            means = reco_train_dataset.means,
            stds = reco_train_dataset.stds,
        )

        if os.path.exists(reco_model_previous_path) and not config.reconstruction.retrain:
            print (f'Loading reco model from {reco_model_previous_path}')
            reco_model: Reconstruction = torch.load(reco_model_previous_path,weights_only=False)
            reco_train_dataset.update(
                initial_means = reco_model.means,
                initial_stds = reco_model.stds,
                momentum = config.reconstruction.momentum,
            )
            reco_valid_dataset.update(
                initial_means = reco_train_dataset.means,
                initial_stds = reco_train_dataset.stds,
                momentum = 0.,
            )
            reco_model.means = reco_train_dataset.means
            reco_model.stds = reco_train_dataset.stds
            reco_train_dataset.preprocess()
            reco_valid_dataset.preprocess()
            print (reco_model)
        else:
            print ('Creating reco model')
            reco_model = Reconstruction(
                *reco_train_dataset.shape,
                initial_means = reco_train_dataset.means,
                initial_stds = reco_train_dataset.stds,
            )
            reco_train_dataset.preprocess()
            reco_valid_dataset.preprocess()
            print (reco_model)
            print ('Pre-training reco model')
            pre_train(
                model = reco_model,
                train_dataset = reco_train_dataset,
                valid_dataset = reco_valid_dataset,
                n_epochs = config.reconstruction.n_epochs_pre,
                batch_size = config.reconstruction.batch_size,
                lr = config.reconstruction.lr_pre,
                plotter = reco_loss_plotter,
            )

        print ('Train reco model')
        reco_model.to("cuda" if torch.cuda.is_available() else "cpu")
        for lr in config.reconstruction.lr_main:
            reco_model.train_model(
                reco_train_dataset,
                reco_valid_dataset,
                batch_size = config.reconstruction.batch_size,
                n_epochs = config.reconstruction.n_epochs_main,
                lr = lr,
                plotter = reco_loss_plotter,
                early_stopping = reco_early_stopping,
            )
        reco_early_stopping.restore_best_weights(reco_model)

        # Validation #
        torch.save(reco_model, reco_model_previous_path)

        reco_validator = ReconstructionValidation(reco_model)
        reco_train_mf = reco_validator.validate(reco_train_dataset)
        reco_valid_mf = reco_validator.validate(reco_valid_dataset)

        reco_loss_plotter.plot(
            os.path.join(
                results_dir,
                "plots",
                "validation",
                "reco_model",
                "losses",
                f"loss_{iteration}.png",
            )
        )
        reco_validator.plot(
            reco_train_mf,
            os.path.join(
                results_dir,
                "plots",
                "validation",
                "reco_model",
                "on_trainingData",
                f"validation_{iteration}.png",
            )
        )
        reco_validator.plot(
            reco_valid_mf,
            os.path.join(
                results_dir,
                "plots",
                "validation",
                "reco_model",
                "on_validationData",
                f"validation_{iteration}.png",
            )
        )

        # Classification training:
        class_model_previous_path = os.path.join(results_dir, "models","class.pt")
        class_loss_plotter = LossPlotting()
        class_early_stopping = EarlyStopping(patience=config.classification.early_stopping)

        print ('Creating class dataset')
        print ('Training:')
        class_train_dataset = ClassificationDataset(
            classes = config.classification.classes,
            input_mf = train_mf,
            augmentations = [
                FlipAugmentation(),
                RotationAugmentation(),
                ShiftAugmentation(),
            ],
        )
        print ('Validation')
        class_valid_dataset = ClassificationDataset(
            classes = config.classification.classes,
            input_mf = valid_mf,
            means = class_train_dataset.means,
            stds = class_train_dataset.stds,
        )

        if os.path.exists(class_model_previous_path) and not config.classification.retrain:
            print (f'Loading class model from {class_model_previous_path}')
            class_model: Classification = torch.load(class_model_previous_path,weights_only=False)
            class_train_dataset.update(
                initial_means = class_model.means,
                initial_stds = class_model.stds,
                momentum = config.classification.momentum,
            )
            class_valid_dataset.update(
                initial_means = class_train_dataset.means,
                initial_stds = class_train_dataset.stds,
                momentum = 0.,
            )
            class_model.means = class_train_dataset.means
            class_model.stds = class_train_dataset.stds
            class_train_dataset.preprocess()
            class_valid_dataset.preprocess()
            print (class_model)
        else:
            print ('Creating class model')
            class_model = Classification(
                *class_train_dataset.shape,
                initial_means = class_train_dataset.means,
                initial_stds = class_train_dataset.stds,
                multiclass = config.classification.multiclass,
                weight = config.classification.weight,
            )
            class_train_dataset.preprocess()
            class_valid_dataset.preprocess()
            print (class_model)
            print ('Pre-training class model')
            pre_train(
                model = class_model,
                train_dataset = class_train_dataset,
                valid_dataset = class_valid_dataset,
                n_epochs = config.classification.n_epochs_pre,
                batch_size = config.classification.batch_size,
                lr = config.classification.lr_pre,
                plotter = class_loss_plotter,
            )

        print ('Train class model')
        class_model.to("cuda" if torch.cuda.is_available() else "cpu")
        for lr in config.classification.lr_main:
            class_model.train_model(
                class_train_dataset,
                class_valid_dataset,
                batch_size = config.classification.batch_size,
                n_epochs = config.classification.n_epochs_main,
                lr = lr,
                plotter = class_loss_plotter,
                early_stopping = class_early_stopping,
            )
        class_early_stopping.restore_best_weights(class_model)

        # Validation #
        torch.save(class_model, class_model_previous_path)

        class_validator = ClassificationValidation(class_model)
        class_train_mf = class_validator.validate(class_train_dataset)
        class_valid_mf = class_validator.validate(class_valid_dataset)


        class_loss_plotter.plot(
            os.path.join(
                results_dir,
                "plots",
                "validation",
                "class_model",
                "losses",
                f"loss_{iteration}.png",
            )
        )
        class_validator.plot(
            class_train_mf,
            os.path.join(
                results_dir,
                "plots",
                "validation",
                "class_model",
                "on_trainingData",
                f"validation_{iteration}.png",
            )
        )
        class_validator.plot(
            class_valid_mf,
            os.path.join(
                results_dir,
                "plots",
                "validation",
                "class_model",
                "on_validationData",
                f"validation_{iteration}.png",
            )
        )

        # Combined mf for surrogate task #
        reco_mf = super_concat([reco_train_mf,reco_valid_mf])
        class_mf = super_concat([class_train_mf,class_valid_mf])

        reco_df = reco_mf.to_pandas()
        class_df = class_mf.to_pandas()
        common_cols = [col for col in reco_df.columns if col in class_df.columns]
        output_df = pd.merge(reco_df,class_df,on=common_cols,how='outer')
        output_df.to_parquet(output_df_path)
        print (f'Saved output df to {output_df_path}')
        os.remove(input_mf_path)
        with open(input_mf_path, "w") as f:
            pass
        print (f'Removed {input_mf_path} and replaced by empty file to save space')



if __name__ == "__main__":
    config_path = sys.argv[1]
    input_mf_path = sys.argv[2]
    output_mf_path = sys.argv[3]
    isVal = sys.argv[4].strip().lower() == "true"
    results_dir = sys.argv[5]
    train(config_path, input_mf_path, output_mf_path, isVal, results_dir)
