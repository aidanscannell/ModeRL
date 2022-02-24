#!/usr/bin/env python3
import hydra
import numpy as np
import tensorflow as tf
from modeopt.mode_opt import ModeOpt
from omegaconf import DictConfig

from .mode_opt_train_dynamics import build_contour_plotter_callbacks

meaning_of_life = 42
tf.random.set_seed(meaning_of_life)
np.random.seed(meaning_of_life)


# @hydra.main(config_path="keras_configs/scenario_7", config_name="train_dynamics")
@hydra.main(
    config_path="keras_configs/scenario_7/train_dynamics",
    config_name="mode_opt_continue_training_dynamics",
)
def continue_training_dynamics_from_cfg(cfg: DictConfig):
    mode_optimiser = tf.keras.models.load_model(
        cfg.dynamics.ckpt_dir, custom_objects={"ModeOpt": ModeOpt}
    )
    # gpf.utilities.set_trainable(
    #     mode_optimiser.dynamics.mosvgpe.gating_network.gp.inducing_variable, False
    # )
    # gpf.utilities.print_summary(
    #     mode_optimiser.dynamics.mosvgpe.gating_network.gp.inducing_variable
    # )

    plotting_callbacks = build_contour_plotter_callbacks(
        mode_optimiser.dynamics.mosvgpe,
        dataset=mode_optimiser.dataset,
        logging_epoch_freq=cfg.logging_epoch_freq,
    )
    mode_optimiser.add_dynamics_callbacks(plotting_callbacks)
    mode_optimiser.optimise_dynamics()


if __name__ == "__main__":

    continue_training_dynamics_from_cfg()
