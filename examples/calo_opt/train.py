# flake8: noqa: E402
import os
import pathlib
import sys
from typing import Union

import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(pathlib.Path(__file__).parent.parent))

from reconstruction.dataset import ReconstructionDataset
from reconstruction.model import Reconstruction
from reconstruction.validation_reconstruction import ReconstructionValidation
from classification.dataset import ClassificationDataset
from classification.model import Classification
from classification.validation_classification import ClassificationValidation


def pre_train(model: Reconstruction, dataset: Dataset, n_epochs: int):
    """ Pre-train the  a given model

    TODO Reconstruction results are normalized. In the future only expose the un-normalized ones,
    but also requires adjustments to the surrogate dataset
    """

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    print("Reconstruction: Pre-training 0")
    model.train_model(dataset, batch_size=512, n_epochs=n_epochs, lr=0.001)

    print("Reconstruction: Pre-training 1")
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.001)
    model.to('cpu')


def train(
    input_df_path: Union[str, os.PathLike],
    output_df_path: Union[str, os.PathLike],
    isVal: bool,
    results_dir: Union[str, os.PathLike],
):
    simulation_df: pd.DataFrame = pd.read_parquet(input_df_path)

    if isVal:
        sys.exit()
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
        n_epochs_main = 50
        reco_model_previous_path = os.path.join(results_dir, "reco_model")
        class_model_previous_path = os.path.join(results_dir, "class_model")

        if os.path.exists(reco_model_previous_path):
            print ('Loading model')
            reco_model: Reconstruction = torch.load(reco_model_previous_path,weights_only=False)
            reco_dataset = ReconstructionDataset(simulation_df, means=reco_model.means, stds=reco_model.stds)
            class_model: Classification = torch.load(class_model_previous_path,weights_only=False)
            class_dataset = ClassificationDataset(simulation_df, means=class_model.means, stds=class_model.stds)
            print (reco_model)
            print (class_model)
        else:
            print ('Creating model')
            reco_dataset = ReconstructionDataset(simulation_df)
            reco_model = Reconstruction(*reco_dataset.shape, reco_dataset.means, reco_dataset.stds)
            class_dataset = ClassificationDataset(simulation_df)
            class_model = Classification(*reco_dataset.shape, reco_dataset.means, reco_dataset.stds)
            print (reco_model)
            print (class_model)
            print ('Pre-training reco model')
            pre_train(reco_model, reco_dataset, n_epochs_pre)
            print ('Pre-training class model')
            pre_train(class_model, class_dataset, n_epochs_pre)

        # Reconstruction training:
        print ('Train reco model')
        reco_model.to("cuda" if torch.cuda.is_available() else "cpu")
        reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.001)
        reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.0005)
        reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.0001)

        print ('Train class model')
        class_model.to("cuda" if torch.cuda.is_available() else "cpu")
        class_model.train_model(class_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.001)
        class_model.train_model(class_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.0005)
        class_model.train_model(class_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.0001)

        reco_validator = ReconstructionValidation(reco_model)
        reco_df_val = reco_validator.validate(reco_dataset)

        class_validator = ClassificationValidation(class_model)
        class_df_val = class_validator.validate(class_dataset)

        common_cols = [col for col in reco_df_val.columns if col in class_df_val.columns]
        output_df_val = pd.merge(reco_df_val,class_df_val,on=common_cols,how='outer')

        output_df_val.to_parquet(output_df_path)
        torch.save(reco_model, reco_model_previous_path)
        torch.save(class_model, class_model_previous_path)

        print ('reco_validator')
        os.makedirs(os.path.join(results_dir, "plots", "validation", "reco_model"),exist_ok=True)
        os.makedirs(os.path.join(results_dir, "plots", "validation", "class_model"),exist_ok=True)
        reco_validator.plot(
            reco_df_val,
            os.path.join(results_dir, "plots", "validation", "reco_model", "on_trainingData")
        )
        print ('class_validator')
        class_validator.plot(
            class_df_val,
            os.path.join(results_dir, "plots", "validation", "class_model", "on_trainingData")
        )



if __name__ == "__main__":
    input_df_path = sys.argv[1]
    output_df_path = sys.argv[2]
    isVal = sys.argv[3].strip().lower() == "true"
    results_dir = sys.argv[4]
    train(input_df_path, output_df_path, isVal, results_dir)
