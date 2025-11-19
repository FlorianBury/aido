# flake8: noqa: E402
import os
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

from utils import LossPlotting, EarlyStopping
from reconstruction.dataset import ReconstructionDataset
from reconstruction.model import Reconstruction
from reconstruction.validation_reconstruction import ReconstructionValidation
from classification.dataset import ClassificationDataset
from classification.model import Classification
from classification.validation_classification import ClassificationValidation


def pre_train(
    model: Reconstruction,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    n_epochs: int,
    batch_size: int,
    plotter: LossPlotting,
    early_stopping: EarlyStopping,
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
        lr = 0.001,
        plotter = plotter,
        early_stopping = early_stopping,
    )

    model.to('cpu')

def split_dataset(df):
    """ When many columns, faster to do pd −> np -> split -> pd """
    X = df.to_numpy()
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, shuffle=True)
    X_train = X[train_idx]
    X_test = X[test_idx]
    train_df = pd.DataFrame(X_train, columns=df.columns)
    test_df  = pd.DataFrame(X_test, columns=df.columns)
    return train_df,test_df

def train(
    input_df_path: Union[str, os.PathLike],
    output_df_path: Union[str, os.PathLike],
    isVal: bool,
    results_dir: Union[str, os.PathLike],
):
    print (f'Loading {input_df_path}')
    simulation_df: pd.DataFrame = pd.read_parquet(input_df_path)
    print (f'Simulation dataframe : {simulation_df.shape}')
    idx = np.random.permutation(len(simulation_df))
    split = int(0.7 * len(simulation_df))
    train_idx = idx[:split]
    test_idx  = idx[split:]
    train_df, valid_df = split_dataset(simulation_df)
    #train_df, valid_df = train_test_split(simulation_df, test_size=0.3,random_state=42)
    print (f'Train dataset : {train_df.shape} / Validation dataset : {valid_df.shape}')

    if isVal:
        reco_model: Reconstruction = torch.load(os.path.join(results_dir, "reco_model"))
        reco_dataset = ReconstructionDataset(simulation_df, means=reco_model.means, stds=reco_model.stds)

        validator = ReconstructionValidation(reco_model)
        output_df_val = validator.validate(reco_dataset)
        output_df_val.to_parquet(output_df_path)
        validator.plot(
            output_df_val,
            os.path.join(results_dir, "plots", "validation", "reco_model", "on_validationData")
        )
    else:
        n_epochs_pre = 50
        n_epochs_main = 100
        batch_size = 256
        reco_model_previous_path = os.path.join(results_dir, "models","reco.pt")
        class_model_previous_path = os.path.join(results_dir, "models","class.pt")

        reco_loss_plotter = LossPlotting()
        class_loss_plotter = LossPlotting()

        reco_early_stopping = EarlyStopping(patience=50)
        class_early_stopping = EarlyStopping(patience=50)

        # Reconstruction training:
        #if os.path.exists(reco_model_previous_path):
        #    print ('Loading reco model')
        #    reco_model: Reconstruction = torch.load(reco_model_previous_path,weights_only=False)
        #    reco_train_dataset = ReconstructionDataset(train_df, means=reco_model.means, stds=reco_model.stds)
        #    reco_valid_dataset = ReconstructionDataset(valid_df, means=reco_model.means, stds=reco_model.stds)
        #    print (reco_model)
        #else:
        #    print ('Creating reco model')
        #    reco_train_dataset = ReconstructionDataset(train_df)
        #    reco_valid_dataset = ReconstructionDataset(valid_df,means=reco_train_dataset.means,stds=reco_train_dataset.stds)
        #    reco_model = Reconstruction(*reco_train_dataset.shape, reco_train_dataset.means, reco_train_dataset.stds)
        #    print (reco_model)
        #    print ('Pre-training reco model')
        #    pre_train(
        #        reco_model,
        #        reco_train_dataset,
        #        reco_valid_dataset,
        #        n_epochs_pre,
        #        batch_size,
        #        reco_loss_plotter,
        #        reco_early_stopping,
        #    )

        #print ('Train reco model')
        #reco_model.to("cuda" if torch.cuda.is_available() else "cpu")
        #reco_model.train_model(
        #    reco_train_dataset,
        #    reco_valid_dataset,
        #    batch_size = batch_size,
        #    n_epochs = n_epochs_main,
        #    lr = 0.001,
        #    plotter = reco_loss_plotter,
        #    early_stopping = reco_early_stopping,
        #)
        #reco_model.train_model(
        #    reco_train_dataset,
        #    reco_valid_dataset,
        #    batch_size = batch_size,
        #    n_epochs = n_epochs_main,
        #    lr = 0.0005,
        #    plotter = reco_loss_plotter,
        #    early_stopping = reco_early_stopping,
        #)
        #reco_model.train_model(
        #    reco_train_dataset,
        #    reco_valid_dataset,
        #    batch_size = batch_size,
        #    n_epochs = n_epochs_main,
        #    lr = 0.0001,
        #    plotter = reco_loss_plotter,
        #    early_stopping = reco_early_stopping,
        #)
        #reco_early_stopping.restore_best_weights(reco_model)

        ## Validation #
        #torch.save(reco_model, reco_model_previous_path)

        #reco_validator = ReconstructionValidation(reco_model)
        #reco_train_df = reco_validator.validate(reco_train_dataset)
        #reco_valid_df = reco_validator.validate(reco_valid_dataset)

        #os.makedirs(os.path.join(results_dir, "plots", "validation", "reco_model"),exist_ok=True)
        #reco_validator.plot(
        #    reco_train_df,
        #    os.path.join(results_dir, "plots", "validation", "reco_model", "on_trainingData")
        #)
        #reco_validator.plot(
        #    reco_valid_df,
        #    os.path.join(results_dir, "plots", "validation", "reco_model", "on_validationData")
        #)
        #reco_loss_plotter.plot(os.path.join(results_dir, "plots", "validation", "reco_model", "losses.png"))

        # Classification training:
        if os.path.exists(class_model_previous_path):
            print ('Creating class dataset')
            class_model: Classification = torch.load(class_model_previous_path,weights_only=False)
            class_train_dataset = ClassificationDataset(train_df, means=class_model.means, stds=class_model.stds)
            class_valid_dataset = ClassificationDataset(valid_df, means=class_model.means, stds=class_model.stds)
            print ('Loading class model')
            print (class_model)
        else:
            print ('Creating class dataset')
            class_train_dataset = ClassificationDataset(train_df)
            class_valid_dataset = ClassificationDataset(valid_df,means=class_train_dataset.means,stds=class_train_dataset.stds)
            print ('Creating class model')
            class_model = Classification(*class_train_dataset.shape, class_train_dataset.means, class_train_dataset.stds)
            print (class_model)
            print ('Pre-training class model')
            pre_train(
                class_model,
                class_train_dataset,
                class_valid_dataset,
                n_epochs_pre,
                batch_size,
                class_loss_plotter,
                class_early_stopping,
            )


        print ('Train class model')
        class_model.to("cuda" if torch.cuda.is_available() else "cpu")
        class_model.train_model(
            class_train_dataset,
            class_valid_dataset,
            batch_size = batch_size,
            n_epochs = n_epochs_main,
            lr = 0.001,
            plotter = class_loss_plotter,
            early_stopping = class_early_stopping,
        )
        class_model.train_model(
            class_train_dataset,
            class_valid_dataset,
            batch_size = batch_size,
            n_epochs = n_epochs_main,
            lr = 0.0005,
            plotter = class_loss_plotter,
            early_stopping = class_early_stopping,
        )
        class_model.train_model(
            class_train_dataset,
            class_valid_dataset,
            batch_size = batch_size,
            n_epochs = n_epochs_main,
            lr = 0.0001,
            plotter = class_loss_plotter,
            early_stopping = class_early_stopping,
        )
        class_early_stopping.restore_best_weights(class_model)

        # Validation #
        torch.save(class_model, class_model_previous_path)

        class_validator = ClassificationValidation(class_model)
        class_train_df = class_validator.validate(class_train_dataset)
        class_valid_df = class_validator.validate(class_valid_dataset)

        os.makedirs(os.path.join(results_dir, "plots", "validation", "class_model"),exist_ok=True)
        class_loss_plotter.plot(os.path.join(results_dir, "plots", "validation", "class_model", "losses.png"))
        class_validator.plot(
            class_train_df,
            os.path.join(results_dir, "plots", "validation", "class_model", "on_trainingData")
        )
        class_validator.plot(
            class_valid_df,
            os.path.join(results_dir, "plots", "validation", "class_model", "on_validationData")
        )

        # Combined df for surrogate task #
        reco_df = pd.concat([reco_train_df,reco_valid_df])
        class_df = pd.concat([class_train_df,class_valid_df])
        common_cols = [col for col in reco_df.columns if col in class_df.columns]
        output_df = pd.merge(reco_df,class_df,on=common_cols,how='outer')
        output_df.to_parquet(output_df_path)


if __name__ == "__main__":
    input_df_path = sys.argv[1]
    output_df_path = sys.argv[2]
    isVal = sys.argv[3].strip().lower() == "true"
    results_dir = sys.argv[4]
    train(input_df_path, output_df_path, isVal, results_dir)
