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
from torch.utils.data import random_split

sys.path.append(os.path.abspath(pathlib.Path(__file__).parent.parent))

from dataset import CaloGraphDataset, concat_dataset
from utils import LossPlotting, EarlyStopping
from config import CaloConfig
from model import GravNetModel
from validation import validation_plot

#def pre_train(
#    model: Reconstruction,
#    train_dataset: Dataset,
#    valid_dataset: Dataset,
#    n_epochs: int,
#    batch_size: int,
#    lr: float,
#    plotter: LossPlotting,
#    early_stopping: EarlyStopping = None,
#):
#    """ Pre-train the  a given model
#
#    TODO Reconstruction results are normalized. In the future only expose the un-normalized ones,
#    but also requires adjustments to the surrogate dataset
#    """
#
#    dev = "cuda" if torch.cuda.is_available() else "cpu"
#    model.to(dev)
#
#    print("Reconstruction: Pre-training")
#    model.train_model(
#        train_dataset,
#        valid_dataset,
#        batch_size = batch_size,
#        n_epochs = n_epochs,
#        lr = lr,
#        plotter = plotter,
#        early_stopping = early_stopping,
#    )
#
#    model.to('cpu')


def train(
    config_path: Union[str, os.PathLike],
    input_graph_path: Union[str, os.PathLike],
    output_graph_path: Union[str, os.PathLike],
    isVal: bool,
    results_dir: Union[str, os.PathLike],
):
    config = CaloConfig.from_json(config_path)
    iteration = int(re.search(r"iteration=(\d+)", input_graph_path).group(1))
    print (f'Loading {input_graph_path}')
    simulation_dataset = CaloGraphDataset.load(input_graph_path)
    print (f'Simulation dataframe : {len(simulation_dataset)} events')
    train_dataset, valid_dataset = random_split(
        dataset = simulation_dataset,
        lengths = [0.9,0.1],
    )
    print (f'Train dataset : {len(train_dataset)} / Validation dataset : {len(valid_dataset)}')
    model_previous_path = os.path.join(results_dir, "models", f"graph_{iteration-1}.pt")

    loss_plotter = LossPlotting()

    if os.path.exists(model_previous_path) and not config.graph.retrain:
        print ('Loading graph model')
        model = torch.load(model_previous_path,weights_only=False)
    else:
        print ('Creating graph model')
        model = GravNetModel(
            inputs = config.graph.inputs,
            regression = config.graph.regression,
            classification = config.graph.classification,
            loss_factors = config.graph.loss_factors,
            shapes = simulation_dataset.get_shapes(),
            means = simulation_dataset.get_means(),
            stds = simulation_dataset.get_stds(),
        )
        print (model)
        print ('Train reco model')
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.train_model(
            train_dataset,
            valid_dataset,
            batch_size = 64,
            n_epochs = 100,
            lr = 1e-3,
            plotter = loss_plotter,
            #early_stopping = reco_early_stopping,
        )
        #model.train_model(
        #    train_dataset,
        #    valid_dataset,
        #    batch_size = 64,
        #    n_epochs = 5,
        #    lr = 1e-4,
        #    plotter = loss_plotter,
        #    #early_stopping = reco_early_stopping,
        #)

        torch.save(model,model_previous_path)

        loss_plotter.plot(
            os.path.join(
                results_dir,
                "plots",
                "validation",
                "reco_model",
                "losses",
                f"loss_{iteration}.png",
            )
        )

    train_dataset = model.inference(
        dataset = train_dataset,
        batch_size = 200,
        t_beta = config.graph.t_beta,
        t_dist = config.graph.t_dist,
    )
    validation_plot(
        train_dataset,
        os.path.join(
            results_dir,
            "plots",
            "validation",
            "reco_model",
            "on_trainingData",
            f"validation_{iteration}.png",
        )
    )
    valid_dataset = model.inference(
        dataset = valid_dataset,
        batch_size = 200,
        t_beta = config.graph.t_beta,
        t_dist = config.graph.t_dist,
    )
    validation_plot(
        valid_dataset,
        os.path.join(
            results_dir,
            "plots",
            "validation",
            "reco_model",
            "on_validationData",
            f"validation_{iteration}.png",
        )
    )

    output_dataset = concat_dataset([train_dataset,valid_dataset])
    output_dataset.save(output_graph_path)

    print (f'Saved output dataset to {output_graph_path}')

#        for lr in config.reconstruction.lr_main:
#            reco_model.train_model(
#                reco_train_dataset,
#                reco_valid_dataset,
#                batch_size = config.reconstruction.batch_size,
#                n_epochs = config.reconstruction.n_epochs_main,
#                lr = lr,
#                plotter = reco_loss_plotter,
#                early_stopping = reco_early_stopping,
#            )
#        reco_early_stopping.restore_best_weights(reco_model)
#
#        # Validation #
#        torch.save(reco_model, model_previous_path)
#
#        reco_validator = ReconstructionValidation(reco_model)
#        reco_train_mf = reco_validator.validate(reco_train_dataset)
#        reco_valid_mf = reco_validator.validate(reco_valid_dataset)
#
#        reco_loss_plotter.plot(
#            os.path.join(
#                results_dir,
#                "plots",
#                "validation",
#                "reco_model",
#                "losses",
#                f"loss_{iteration}.png",
#            )
#        )
#        reco_validator.plot(
#            reco_train_mf,
#            os.path.join(
#                results_dir,
#                "plots",
#                "validation",
#                "reco_model",
#                "on_trainingData",
#                f"validation_{iteration}.png",
#            )
#        )
#        reco_validator.plot(
#            reco_valid_mf,
#            os.path.join(
#                results_dir,
#                "plots",
#                "validation",
#                "reco_model",
#                "on_validationData",
#                f"validation_{iteration}.png",
#            )
#        )
#        # Combined mf for surrogate task #
#        reco_mf = super_concat([reco_train_mf,reco_valid_mf])
#        class_mf = super_concat([class_train_mf,class_valid_mf])
#
#        reco_df = reco_mf.to_pandas()
#        class_df = class_mf.to_pandas()
#        common_cols = [col for col in reco_df.columns if col in class_df.columns]
#        output_df = pd.merge(reco_df,class_df,on=common_cols,how='outer')
#        output_df.to_parquet(output_graph_path)
#        print (f'Saved output df to {output_graph_path}')


if __name__ == "__main__":
    config_path = sys.argv[1]
    input_graph_path = sys.argv[2]
    output_mf_path = sys.argv[3]
    isVal = sys.argv[4].strip().lower() == "true"
    results_dir = sys.argv[5]
    train(config_path, input_graph_path, output_mf_path, isVal, results_dir)
