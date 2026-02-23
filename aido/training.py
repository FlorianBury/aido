import json
import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from aido.config import AIDOConfig
from aido.logger import logger
from aido.optimizer import Optimizer
from aido.simulation_helpers import SimulationParameterDictionary
from aido.surrogate import Surrogate, SurrogateDataset
from aido.surrogate_validation import SurrogateValidation


def pre_train(model: Surrogate, train_dataset: SurrogateDataset, valid_dataset: SurrogateDataset, n_epochs: int, batch_size: int):
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

    model.train_model(train_dataset, valid_dataset, batch_size=batch_size, n_epochs=n_epochs, lr=0.005)


def training_loop(
        reco_file_paths_dict: dict | str | os.PathLike,
        reconstruction_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        classification_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        iteration: int,
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
    train_df, valid_df = train_test_split(surrogate_df, test_size=0.2,random_state=42)

    print ('Creating surrogate datasets')
    surrogate_train_dataset = SurrogateDataset(
        input_df = train_df,
        reconstruction = config.surrogate.reconstruction,
        classification = config.surrogate.classification,
        normalize_parameters = True,
    )
    surrogate_valid_dataset = SurrogateDataset(
        input_df = valid_df,
        reconstruction = config.surrogate.reconstruction,
        classification = config.surrogate.classification,
        normalize_parameters = True,
        means = surrogate_train_dataset.means,
        stds = surrogate_train_dataset.stds,
    )


    if os.path.isfile(surrogate_save_path):
        print ('Surrogate already trained')
        surrogate: Surrogate = torch.load(surrogate_save_path,weights_only=False)
    else:
        if os.path.isfile(surrogate_previous_path) and not config.surrogate.retrain:
            print ('Loading surrogate')
            surrogate: Surrogate = torch.load(surrogate_previous_path,weights_only=False)
            print (surrogate)
        else:
            print ('Creating surrogate')
            surrogate = Surrogate(
                *surrogate_train_dataset.shape,
                n_time_steps = config.surrogate.n_time_steps,
                betas = config.surrogate.betas,
                initial_means = surrogate_train_dataset.means,
                initial_stds = surrogate_train_dataset.stds,
            )
            print (surrogate)
            pre_train(
                surrogate,
                surrogate_train_dataset,
                surrogate_valid_dataset,
                config.surrogate.n_epoch_pre,
                config.surrogate.batch_size,
            )

        logger.info("Surrogate Training")
        n_epochs_main = config.surrogate.n_epochs_main
        batch_size = config.surrogate.batch_size
        surrogate.train_model(
            surrogate_train_dataset,
            surrogate_valid_dataset,
            batch_size = batch_size,
            n_epochs = n_epochs_main,
            lr = 0.001,
        )
        surrogate.train_model(
            surrogate_train_dataset,
            surrogate_valid_dataset,
            batch_size = batch_size,
            n_epochs = n_epochs_main,
            lr = 0.0005,
        )
        surrogate.train_model(
            surrogate_train_dataset,
            surrogate_valid_dataset,
            batch_size = batch_size,
            n_epochs = n_epochs_main,
            lr = 0.0001,
        )

        torch.save(surrogate, surrogate_save_path)

        # Validation
        surrogate_validator = SurrogateValidation(surrogate)
        train_df = surrogate_validator.validate(surrogate_train_dataset)
        surrogate_validator.plot(
            train_df,
            fig_savepath = os.path.join(
                results_dir,
                "plots",
                "validation",
                "surrogate",
                "on_trainingData",
                f"validation_{iteration}.png",
            ),
        )
        valid_df = surrogate_validator.validate(surrogate_valid_dataset)
        surrogate_validator.plot(
            valid_df,
            fig_savepath = os.path.join(
                results_dir,
                "plots",
                "validation",
                "surrogate",
                "on_validationData",
                f"validation_{iteration}.png",
            ),
        )



    # Optimization
    optimizer = Optimizer(parameter_dict=parameter_dict)
    if os.path.isfile(optimizer_previous_path):
        checkpoint = torch.load(optimizer_previous_path,weights_only=False)
        optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    surrogate_dataset = SurrogateDataset(
        input_df = surrogate_df,
        means = surrogate.means,
        stds = surrogate.stds,
        reconstruction = config.surrogate.reconstruction,
        classification = config.surrogate.classification,
        normalize_parameters = True,
    )

    optimizer_lr = config.optimizer.lr * config.optimizer.gamma ** iteration

    def sigmoid_turn_on(x,yi,yf,k,T):
        return yi + (yf-yi) * 1 / (1+np.exp(-k*(x-T)))

    alpha_reco = sigmoid_turn_on(iteration,*config.optimizer.turn_on_reco)
    alpha_class = sigmoid_turn_on(iteration,*config.optimizer.turn_on_class)

    updated_parameter_dict, is_optimal = optimizer.optimize(
        surrogate_model = surrogate,
        dataset = surrogate_dataset,
        scale = config.optimizer.scale,
        batch_size = config.optimizer.batch_size,
        n_epochs = config.optimizer.n_epochs,
        lr = optimizer_lr,
        end_factor = config.optimizer.end_factor,
        alpha_reco = alpha_reco,
        alpha_class = alpha_class,
        reconstruction_loss = reconstruction_loss_function,
        classification_loss = classification_loss_function,
        additional_constraints = constraints,
        parameter_optimizer_savepath = parameter_optimizer_savepath,
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
