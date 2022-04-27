#!/usr/bin/env python3
from typing import List, Optional

import hydra
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from modeopt.dynamics import ModeOptDynamics
from modeopt.controllers.utils import build_riemannian_energy_controller
from modeopt.controllers.non_feedback.explorative_controller import (
    ExplorativeController,
)
from modeopt.mode_opt import ModeOpt
from modeopt.monitor.callbacks import PlotFn, TensorboardImageCallbackScipy
from modeopt.plotting import ModeOptContourPlotter
from omegaconf import DictConfig

tfd = tfp.distributions

meaning_of_life = 42
tf.random.set_seed(meaning_of_life)
np.random.seed(meaning_of_life)


@hydra.main(
    config_path="keras_configs/scenario_10/trajectory_optimisation",
    config_name="explore",
)
def mode_opt_explore_from_cfg(cfg: DictConfig):
    dynamics = tf.keras.models.load_model(
        cfg.dynamics.ckpt_dir, custom_objects={"ModeOptDynamics": ModeOptDynamics}
    )
    dynamics.desired_mode = cfg.dynamics.desired_mode  # update desired mode
    start_state = tf.reshape(
        tf.constant(cfg.start_state, dtype=default_float()), shape=(1, -1)
    )
    target_state = tf.reshape(
        tf.constant(cfg.target_state, dtype=default_float()), shape=(1, -1)
    )
    terminal_state_cost_matrix = tf.constant(
        cfg.cost_fn.terminal_state_cost_matrix, dtype=default_float()
    )
    control_cost_matrix = tf.constant(
        cfg.cost_fn.control_cost_matrix, dtype=default_float()
    )

    explorative_controller = ExplorativeController(
        start_state,
        target_state,
        dynamics,
        cfg.dynamics.desired_mode,
        horizon=cfg.horizon,
        control_dim=cfg.dynamics.control_dim,
        max_iterations=cfg.max_iterations,
        mode_satisfaction_prob=cfg.mode_satisfaction_prob,
        terminal_state_cost_weight=cfg.terminal_state_cost_weight,
        state_diff_cost_weight=cfg.state_diff_cost_weight,
        control_cost_weight=cfg.control_cost_weight,
        # keep_last_solution
        method=cfg.method,
    )

    mode_optimiser = ModeOpt(
        start_state,
        target_state,
        env_name=cfg.env_name,
        dynamics=dynamics,
        explorative_controller=explorative_controller,
    )

    plotting_callbacks = build_contour_plotter_callbacks(
        mode_optimiser, logging_freq=cfg.logging_freq
    )
    mode_optimiser.mode_controller_callback = plotting_callbacks
    mode_optimiser.optimise_mode_controller()


def build_contour_plotter_callbacks(
    mode_optimiser: ModeOpt,
    logging_freq: Optional[int] = 3,
    log_dir: Optional[str] = "./logs",
) -> List[PlotFn]:
    test_inputs = create_test_inputs(x_min=[-3, -3], x_max=[3, 3], input_dim=4)

    mode_optimiser_plotter = ModeOptContourPlotter(
        mode_optimiser=mode_optimiser, test_inputs=test_inputs
    )
    gating_gps_plotting_cb = TensorboardImageCallbackScipy(
        plot_fn=mode_optimiser_plotter.plot_trajectories_over_gating_network_gps,
        logging_epoch_freq=logging_freq,
        log_dir=log_dir,
        name="Trajectories over gating function GPs",
    )
    mixing_probs_plotting_cb = TensorboardImageCallbackScipy(
        plot_fn=mode_optimiser_plotter.plot_trajectories_over_mixing_probs,
        logging_epoch_freq=logging_freq,
        log_dir=log_dir,
        name="Trajectories over mixing probabilities",
    )

    def plotting_callback(step, variables, value):
        mixing_probs_plotting_cb(step, variables, value)
        gating_gps_plotting_cb(step, variables, value)

    return plotting_callback


if __name__ == "__main__":

    riemannian_energy_trajectory_optimisation_from_cfg()
