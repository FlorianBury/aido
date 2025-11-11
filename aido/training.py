import json
import os
from typing import Callable

import numpy as np
import pandas as pd
import torch

from aido.config import AIDOConfig
from aido.logger import logger
from aido.optimizer import Optimizer
from aido.simulation_helpers import SimulationParameterDictionary
from aido.surrogate import Surrogate, SurrogateDataset
from aido.surrogate_validation import SurrogateValidation


def pre_train(model: Surrogate, dataset: SurrogateDataset, n_epochs: int):
    """Pre-train the Surrogate Model using a three-stage process.

    This function performs pre-training in three stages with different
    batch sizes and learning rates to ensure stable convergence.

    Parameters
    ----------
    model : Surrogate
        The surrogate model to pre-train.
    dataset : SurrogateDataset
        The dataset to use for training.
    n_epochs : int
        Number of epochs to train in each stage.
    """
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    logger.info('Surrogate: Pre-Training 0')
    model.train_model(dataset, batch_size=512, n_epochs=n_epochs, lr=0.001)

    logger.info('Surrogate: Pre-Training 1')
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.001)

    logger.info('Surrogate: Pre-Training 2')
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.0003)


def training_loop(
        reco_file_paths_dict: dict | str | os.PathLike,
        reconstruction_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        classification_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        constraints: None | Callable[[SimulationParameterDictionary], float | torch.Tensor] = None,
        ):

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
    surrogate_df = pd.read_parquet(output_df_path)

    if os.path.isfile(surrogate_save_path):
        surrogate: Surrogate = torch.load(surrogate_save_path,weights_only=False)
        surrogate_dataset = SurrogateDataset(
            input_df = surrogate_df,
            means = surrogate.means,
            stds = surrogate.stds,
            reconstruction = config.surrogate.reconstruction,
            classification = config.surrogate.classification,
            normalize_parameters = False,
        )
    else:
        if os.path.isfile(surrogate_previous_path):
            print ('Loading surrogate')
            surrogate: Surrogate = torch.load(surrogate_previous_path,weights_only=False)
            surrogate_dataset = SurrogateDataset(
                input_df = surrogate_df,
                means = surrogate.means,
                stds = surrogate.stds,
                reconstruction = config.surrogate.reconstruction,
                classification = config.surrogate.classification,
                normalize_parameters = False,
            )
            print (surrogate)
        else:
            print ('Creating surrogate')
            surrogate_dataset = SurrogateDataset(
                input_df = surrogate_df,
                reconstruction = config.surrogate.reconstruction,
                classification = config.surrogate.classification,
                normalize_parameters = False,
            )
            surrogate = Surrogate(
                *surrogate_dataset.shape,
                initial_means = surrogate_dataset.means,
                initial_stds = surrogate_dataset.stds,
            )
            print (surrogate)
            pre_train(surrogate, surrogate_dataset, config.surrogate.n_epoch_pre)

        logger.info("Surrogate Training")
        n_epochs_main = config.surrogate.n_epochs_main
        surrogate.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.005)
        surrogate_loss = surrogate.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.0003)

        surrogate_lr = 0.001 * (1 if parameter_dict.iteration <= 50 else 0.5)

        while not surrogate.update_best_surrogate_loss(surrogate_loss):
            logger.info("Surrogate retraining")
            pre_train(surrogate, surrogate_dataset, config.surrogate.n_epoch_pre)
            surrogate.train_model(
                surrogate_dataset,
                batch_size=1024,
                n_epochs=n_epochs_main // 5,
                lr=3 * surrogate_lr
            )
            surrogate.train_model(
                surrogate_dataset,
                batch_size=1024,
                n_epochs=n_epochs_main // 2,
                lr=1 * surrogate_lr
            )
            surrogate.train_model(
                surrogate_dataset,
                batch_size=1024,
                n_epochs=n_epochs_main // 2,
                lr=0.3 * surrogate_lr)
            surrogate_loss = surrogate.train_model(
                surrogate_dataset,
                batch_size=1024,
                n_epochs=n_epochs_main // 2,
                lr=0.1 * surrogate_lr,
            )

    torch.save(surrogate, surrogate_save_path)

    # Validation
    surrogate_validator = SurrogateValidation(surrogate)
    validation_df = surrogate_validator.validate(surrogate_dataset)
    surrogate_validator.plot(
        validation_df,
        fig_savepath=os.path.join(results_dir, "plots", "validation", "surrogate", "on_trainingData.png"),
    )


    # Optimization
    optimizer = Optimizer(parameter_dict=parameter_dict)
    if os.path.isfile(optimizer_previous_path):
        checkpoint = torch.load(optimizer_previous_path,weights_only=False)
        optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    updated_parameter_dict, is_optimal = optimizer.optimize(
        surrogate_model=surrogate,
        dataset=surrogate_dataset,
        batch_size=config.optimizer.batch_size,
        n_epochs=config.optimizer.n_epochs,
        reconstruction_loss=reconstruction_loss_function,
        classification_loss=classification_loss_function,
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
    ).to_csv(surrogate_loss_save_path+'.csv')
    pd.DataFrame(
        np.array(optimizer.optimizer_loss),
        columns=["Optimizer Loss"]
    ).to_csv(optimizer_loss_save_path+'.csv')
    pd.DataFrame(
        np.array(optimizer.constraints_loss),
        columns=["Constraints Loss"]
    ).to_csv(constraints_loss_save_path+'.csv')

    return updated_parameter_dict
