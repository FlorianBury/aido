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
from timing import Timer

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
            n_epochs = config.graph.n_epochs,
            batch_size = config.graph.batch_size,
            lr = config.graph.lr,
            annealing = config.graph.annealing,
            plotter = loss_plotter,
            #early_stopping = reco_early_stopping,
        )
        model.train_model(
            train_dataset,
            valid_dataset,
            n_epochs = 10,
            batch_size = config.graph.batch_size,
            lr = config.graph.lr / 10,
            plotter = loss_plotter,
            #early_stopping = reco_early_stopping,
        )
        model.train_model(
            train_dataset,
            valid_dataset,
            n_epochs = 10,
            batch_size = config.graph.batch_size,
            lr = config.graph.lr / 100,
            plotter = loss_plotter,
            #early_stopping = reco_early_stopping,
        )

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

    train_dataset,t_dist = model.inference(
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
    valid_dataset,_ = model.inference(
        dataset = valid_dataset,
        batch_size = 200,
        t_beta = config.graph.t_beta,
        t_dist = t_dist,
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

    if config.timing.enabled:
        timer = Timer(
            model,
            runs = config.timing.runs,
        )
        output_dataset = timer(output_dataset)


    output_dataset.save(output_graph_path)

    print (f'Saved output dataset to {output_graph_path}')

    for i in range(20):
        output_dataset.plot(
            idx = i,
            path = os.path.join(
                os.path.dirname(output_graph_path),
                f'event_{i}.png'
            ),
        )

if __name__ == "__main__":
    config_path = sys.argv[1]
    input_graph_path = sys.argv[2]
    output_mf_path = sys.argv[3]
    isVal = sys.argv[4].strip().lower() == "true"
    results_dir = sys.argv[5]
    train(config_path, input_graph_path, output_mf_path, isVal, results_dir)
