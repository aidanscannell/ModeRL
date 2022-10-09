#!/usr/bin/env python3
from typing import Callable, List, Optional

import gpflow as gpf
import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import tensorflow as tf
import wandb
from gpflow import default_float
from gpflow.inducing_variables import InducingPoints, MultioutputInducingVariables
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Constant
from mosvgpe.custom_types import Dataset, InputData
from mosvgpe.experts import SVGPExpert
from mosvgpe.gating_networks import SVGPGatingNetwork
from mosvgpe.mixture_of_experts import MixtureOfSVGPExperts
from wandb.keras import WandbCallback


# import examples

# from examples.mcycle.plot import build_plotting_callbacks

# from examples.mcycle.plot import *

# from .plot import build_plotting_callbacks

# from mosvgpe.utils import sample_mosvgpe_inducing_inputs_from_data


@hydra.main(config_path="configs", config_name="main")
def train(cfg: omegaconf.DictConfig):
    # print(examples.mcycle.plot.build_plotting_callbacks)
    tf.random.set_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    ###### Initialise WandB run and save experiment config ######
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(
        entity=cfg.wandb.entity, project=cfg.wandb.project, config=cfg_dict
    )
    log_dir = run.dir

    ###### Load the dataset ######
    dataset = hydra.utils.instantiate(cfg.dataset)

    ###### Instantiate the model ######
    model = hydra.utils.instantiate(cfg.model)
    gpf.utilities.print_summary(model)
    model(dataset[0])  # Need to build model before training
    # sample_mosvgpe_inducing_inputs_from_data(dataset[0], model)

    # Define WandbCallback for experiment tracking and plotting callbacks
    callbacks = [
        # hydra.utils.instantiate(cfg.build_plotting_callbacks),
        # build_plotting_callbacks(
        #     model=model, logging_epoch_freq=cfg.logging_epoch_freq
        # ),
        WandbCallback(
            monitor="val_loss",
            log_weights=False,
            # save=False,
            log_evaluation=True,
            validation_steps=5,
        ),
    ]
    # Compile the Keras model and train it
    optimiser = tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate)
    model.compile(optimizer=optimiser)

    ###### Train the model ######
    model.fit(
        dataset[0],
        dataset[1],
        callbacks=callbacks,
        batch_size=cfg.train.batch_size,
        epochs=cfg.train.epochs,
        verbose=cfg.train.verbose,
        validation_split=cfg.train.validation_split,
    )


if __name__ == "__main__":
    train()  # pyright: ignore
