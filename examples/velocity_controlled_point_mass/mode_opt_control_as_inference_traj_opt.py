#!/usr/bin/env python3
from typing import List, Optional

import hydra
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from modeopt.controllers.utils import build_control_as_inference_controller
from modeopt.dynamics import ModeOptDynamics
from modeopt.mode_opt import ModeOpt
from modeopt.monitor.callbacks import PlotFn, TensorboardImageCallbackScipy
from modeopt.plotting import ModeOptContourPlotter
from omegaconf import DictConfig
from .mode_opt_riemannian_energy_traj_opt import create_test_inputs

tfd = tfp.distributions

meaning_of_life = 42
tf.random.set_seed(meaning_of_life)
np.random.seed(meaning_of_life)


@hydra.main(
    # config_path="keras_configs/scenario_7/trajectory_optimisation",
    config_path="keras_configs/scenario_5/trajectory_optimisation",
    config_name="control_as_inference",
    # config_name="control_as_inference_deterministic",
)
def control_as_inference_trajectory_optimisation_from_cfg(cfg: DictConfig):
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

    controller = build_control_as_inference_controller(
        start_state,
        target_state,
        dynamics,
        cfg.dynamics.desired_mode,
        cfg.horizon,
        cfg.dynamics.control_dim,
        control_cost_matrix,
        terminal_state_cost_matrix,
        max_iterations=cfg.max_iterations,
        # constraints_lower_bound=list(cfg.control_constraints_lower),
        # constraints_upper_bound=list(cfg.control_constraints_upper),
        # keep_last_solution
        method=cfg.method,
        gaussian_controls=cfg.gaussian_controls,
    )

    mode_optimiser = ModeOpt(
        start_state,
        target_state,
        env_name=cfg.env_name,
        dynamics=dynamics,
        mode_controller=controller,
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

    control_as_inference_trajectory_optimisation_from_cfg()
