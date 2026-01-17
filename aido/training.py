import json
import os
import random
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from aido.config import AIDOConfig
from aido.logger import logger
from aido.optimizer import Optimizer
from aido.simulation_helpers import SimulationParameterDictionary
from aido.surrogate import Surrogate, SurrogateDataset, validation_plot
from aido.losses import HungarianMatching
from aido.utils import LossPlotting


#def pre_train(model: Surrogate, dataset: SurrogateDataset, n_epochs: int):
#    """Pre-train the Surrogate Model using a three-stage process.
#
#    This function performs pre-training in three stages with different
#    batch sizes and learning rates to ensure stable convergence.
#
#    Parameters
#    ----------
#    model : Surrogate
#        The surrogate model to pre-train.
#    dataset : SurrogateDataset
#        The dataset to use for training.
#    n_epochs : int
#        Number of epochs to train in each stage.
#    """
#    model.to("cuda" if torch.cuda.is_available() else "cpu")
#
#    logger.info('Surrogate: Pre-Training 0')
#    model.train_model(dataset, batch_size=512, n_epochs=n_epochs, lr=0.001)
#
#    logger.info('Surrogate: Pre-Training 1')
#    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.001)
#
#    logger.info('Surrogate: Pre-Training 2')
#    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.0003)


def training_loop(
        reco_file_paths_dict: dict | str | os.PathLike,
        iteration: int,
        constraints: None | Callable[[SimulationParameterDictionary], float | torch.Tensor] = None,
        ):
    """Internal training of the Surrogate and Optimizer models

    Args:
        reco_file_paths_dict (dict | str | os.PathLike): Either the dict with all the file paths
            or a single filepath (str or os.PathLike) that we first have to read from JSON.
        reconstruction_loss_function (Callable): The user-defined loss function that provides the
            goodness of a given design. Has to take two Tensors (truth and predicted) and return a scalar
            Tensor used as the Optimizer loss.
        constraints (Callable, optional). Additional loss function to be applied on top of the regular
            loss function, for example to account for cost penalties. Default is None

    Returns:
        SimulationParameterDictionary: The updated values as proposed by the Optimizer model.

    Note:
        This function is integral to the correct training of the surrogate and optimizer models. The
        training itself consists of these steps:
         1. Track all the file paths needed
         2. Instantiate the Surrogate model if not done so, load it from .pt file if available from
            current iteration (if training was stopped), then train it.
         3. Run the Optimizer
         4. Save results
    """

    if isinstance(reco_file_paths_dict, (str, os.PathLike)):
        with open(reco_file_paths_dict, "r") as file:
            reco_file_paths_dict = json.load(file)

    config = AIDOConfig.from_json(reco_file_paths_dict["config_path"])

    results_dir = reco_file_paths_dict["results_dir"]
    output_df_path = reco_file_paths_dict["reco_output_df"]
    parameter_dict_input_path = reco_file_paths_dict["current_parameter_dict"]
    surrogate_previous_path = reco_file_paths_dict["surrogate_model_previous_path"]
    optimizer_previous_path = reco_file_paths_dict["optimizer_model_previous_path"]
    surrogate_save_path = reco_file_paths_dict["surrogate_model_save_path"]
    optimizer_save_path = reco_file_paths_dict["optimizer_model_save_path"]
    optimizer_loss_save_path = reco_file_paths_dict["optimizer_loss_save_path"]
    surrogate_loss_save_path = reco_file_paths_dict["surrogate_loss_save_path"]
    constraints_loss_save_path = reco_file_paths_dict["constraints_loss_save_path"]
    parameter_optimizer_savepath = os.path.join(results_dir, "models", "parameter_optimizer_df")

    # Surrogate
    parameter_dict = SimulationParameterDictionary.from_json(parameter_dict_input_path)
    surrogate_dataset = SurrogateDataset.load_sparse(config,output_df_path)

    indices = list(range(len(surrogate_dataset)))
    random.shuffle(indices)
    split = int(len(surrogate_dataset)*0.9)
    train_indices = indices[:split]
    valid_indices = indices[split:]

    train_dataset = Subset(surrogate_dataset, train_indices)
    valid_dataset = Subset(surrogate_dataset, valid_indices)

    loss_matching = HungarianMatching(
        feature_dict = surrogate_dataset.feature_dict,
        loss_factors = config.loss.loss_factors,
        fake_penalty = config.loss.fake_penalty,
        missing_penalty = config.loss.missing_penalty,
        classification = config.surrogate.classification,
        regression = config.surrogate.regression,
    )


    if os.path.isfile(surrogate_save_path):
        print ('Loading surrogate')
        surrogate: Surrogate = torch.load(surrogate_save_path,weights_only=False)
        print (surrogate)
    else:
        if os.path.isfile(surrogate_previous_path):
            print ('Loading previous surrogate')
            surrogate: Surrogate = torch.load(surrogate_previous_path)
        else:
            print ('Creating surrogate')
            surrogate = Surrogate(
                shapes = surrogate_dataset.get_shapes(),
                means = surrogate_dataset.get_means(),
                stds = surrogate_dataset.get_stds(),
                max_seq_len = surrogate_dataset.max_seq_len,
                feature_dict = surrogate_dataset.feature_dict,
                classification = config.surrogate.classification,
                regression = config.surrogate.regression,
                multiplicity = config.surrogate.multiplicity,
                loss_factors = config.surrogate.loss_factors,
            )

            print (surrogate)

            plotter = LossPlotting()


            surrogate.train_model(
                train_dataset,
                valid_dataset,
                n_epochs = 100,
                batch_size = 256,
                lr = 1e-3,
                teacher_forcing = 1.0,
                plotter = plotter,
            )
            surrogate.train_model(
                train_dataset,
                valid_dataset,
                n_epochs = 100,
                batch_size = 256,
                lr = 1e-4,
                teacher_forcing = 1.0,
                plotter = plotter,
            )
            surrogate.train_model(
                train_dataset,
                valid_dataset,
                n_epochs = 100,
                batch_size = 256,
                lr = 1e-5,
                teacher_forcing = 1.0,
                plotter = plotter,
            )
            #surrogate.train_model(
            #    train_dataset,
            #    valid_dataset,
            #    n_epochs = 10,
            #    batch_size = 1024,
            #    lr = 1e-4,
            #    teacher_forcing = 0.5,
            #    plotter = plotter,
            #)


#            surrogate.train_model(
#                train_dataset,
#                valid_dataset,
#                n_epochs = 25,
#                batch_size = 64,
#                lr = 1e-4,
#                teacher_forcing = 0.75,
#                plotter = plotter,
#            )

            plotter.plot(
                os.path.join(
                    results_dir,
                    "plots",
                    "validation",
                    "surrogate",
                    "losses",
                    f"loss_{iteration}.png",
                )
            )

            torch.save(surrogate, surrogate_save_path)

        train_samples, train_samples_mask, train_times = surrogate.inference(
            dataset = train_dataset,
            batch_size = 1024,
            oversampling = config.loss.oversampling,
        )
        valid_samples, valid_samples_mask, valid_times = surrogate.inference(
            dataset = valid_dataset,
            batch_size = 1024,
            oversampling = config.loss.oversampling,
        )

        validation_plot(
            loss_matching = loss_matching,
            savepath = os.path.join(
                results_dir,
                "plots",
                "validation",
                "surrogate",
                "on_trainingData",
                f"plot_{iteration}.png",
            ),
            true_part = surrogate_dataset.true_part[train_indices],
            true_mask = surrogate_dataset.true_mask[train_indices],
            reco_part = surrogate_dataset.reco_part[train_indices],
            reco_mask = surrogate_dataset.reco_mask[train_indices],
            samp_part = train_samples,
            samp_mask = train_samples_mask,
            reco_time = surrogate_dataset.reco_time[train_indices],
            samp_time = train_times,
        )
        validation_plot(
            loss_matching = loss_matching,
            savepath = os.path.join(
                results_dir,
                "plots",
                "validation",
                "surrogate",
                "on_validationData",
                f"plot_{iteration}.png",
            ),
            true_part = surrogate_dataset.true_part[valid_indices],
            true_mask = surrogate_dataset.true_mask[valid_indices],
            reco_part = surrogate_dataset.reco_part[valid_indices],
            reco_mask = surrogate_dataset.reco_mask[valid_indices],
            samp_part = valid_samples,
            samp_mask = valid_samples_mask,
            reco_time = surrogate_dataset.reco_time[valid_indices],
            samp_time = valid_times,
        )


        surrogate_dataset.samp_part, surrogate_dataset.samp_mask, surrogate_dataset.samp_time = surrogate.inference(
            dataset = surrogate_dataset,
            batch_size = 1024,
            oversampling = config.loss.oversampling,
        )

        for i in range(20):
            surrogate_dataset.plot(
                config = config,
                idx = train_indices[i],
                savepath = os.path.join(
                    os.path.dirname(output_df_path),
                    f'event_train_{i}.png',
                )
            )
            surrogate_dataset.plot(
                config = config,
                idx = valid_indices[i],
                savepath = os.path.join(
                    os.path.dirname(output_df_path),
                    f'event_valid_{i}.png',
                )
            )

        output_dense_path = os.path.join(os.path.dirname(output_df_path),'reco_output_dense')
        surrogate_dataset.save_dense(output_dense_path)

    # Optimization
    optimizer = Optimizer(parameter_dict=parameter_dict)
    if os.path.isfile(optimizer_previous_path):
        checkpoint = torch.load(optimizer_previous_path)
        optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    updated_parameter_dict, is_optimal = optimizer.optimize(
        surrogate_model=surrogate,
        dataset=surrogate_dataset,
        batch_size=config.optimizer.batch_size,
        n_epochs=config.optimizer.n_epochs,
        loss_matching = loss_matching,
        additional_constraints=constraints,
        parameter_optimizer_savepath=parameter_optimizer_savepath,
        lr=config.optimizer.lr
    )
    if not is_optimal:
        raise RuntimeError
    else:
        torch.save({"optimizer_state_dict": optimizer.optimizer.state_dict()}, optimizer_save_path)

    pd.DataFrame(
        np.array(surrogate.surrogate_loss),
        columns=["Surrogate Loss"]
    ).to_csv(surrogate_loss_save_path)
    pd.DataFrame(
        np.array(optimizer.optimizer_loss),
        columns=["Optimizer Loss"]
    ).to_csv(optimizer_loss_save_path)
    pd.DataFrame(
        np.array(optimizer.constraints_loss),
        columns=["Constraints Loss"]
    ).to_csv(constraints_loss_save_path)

    return updated_parameter_dict
