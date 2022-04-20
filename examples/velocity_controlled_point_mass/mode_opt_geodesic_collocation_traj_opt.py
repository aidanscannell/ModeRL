#!/usr/bin/env python3
from typing import List, Optional

import hydra
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from modeopt.controllers import GeodesicController
from modeopt.cost_functions import (
    ControlQuadraticCostFunction,
    StateQuadraticCostFunction,
    quadratic_cost_fn,
)
from modeopt.dynamics import ModeOptDynamics
from modeopt.mode_opt import ModeOpt
from modeopt.monitor.callbacks import PlotFn, TensorboardImageCallbackScipy
from modeopt.plotting import ModeOptContourPlotter
from mogpe.keras.utils import load_from_json_config, save_json_config
from omegaconf import DictConfig

from .mode_opt_riemannian_energy_traj_opt import create_test_inputs

tfd = tfp.distributions

meaning_of_life = 42
tf.random.set_seed(meaning_of_life)
np.random.seed(meaning_of_life)


@hydra.main(
    config_path="keras_configs/scenario_5/trajectory_optimisation",
    # config_path="keras_configs/scenario_9/trajectory_optimisation",
    # config_path="keras_configs/scenario_7/trajectory_optimisation",
    # config_name="geodesic_collocation_mid_point",
    # config_name="geodesic_collocation",
    # config_name="geodesic_collocation_low",
    # config_name="geodesic_collocation_high",
    config_name="geodesic_collocation_high_mid_point",
)
def collocation_trajectory_optimisation_via_constraints_from_cfg(
    cfg: DictConfig,
):
    dynamics = tf.keras.models.load_model(
        cfg.dynamics.ckpt_dir, custom_objects={"ModeOptDynamics": ModeOptDynamics}
    )
    dynamics.desired_mode = cfg.dynamics.desired_mode  # update desired mode
    start_state = tf.reshape(
        tf.constant(cfg.controller.start_state, dtype=default_float()), shape=(1, -1)
    )
    target_state = tf.reshape(
        tf.constant(cfg.controller.target_state, dtype=default_float()), shape=(1, -1)
    )
    dummy_cost_weight = tf.constant(
        cfg.controller.dummy_cost_matrix, dtype=default_float()
    )
    try:
        mid_state = tf.reshape(
            tf.constant(cfg.controller.mid_state, dtype=default_float()), shape=(1, -1)
        )
    except:
        mid_state = None

    # controller= tf.keras.models.load_model(
    #     cfg.dynamics.ckpt_dir, custom_objects={"GeodesicController": GeodesicController}
    # )
    controller = GeodesicController(
        start_state=start_state,
        target_state=target_state,
        dynamics=dynamics,
        horizon=cfg.controller.horizon,
        t_init=cfg.controller.t_init,
        t_end=cfg.controller.t_end,
        riemannian_metric_covariance_weight=cfg.controller.riemannian_metric_covariance_weight,
        max_collocation_iterations=cfg.controller.max_collocation_iterations,
        collocation_constraints_lower=cfg.controller.collocation_constraints_lower,
        collocation_constraints_upper=cfg.controller.collocation_constraints_upper,
        dummy_cost_weight=dummy_cost_weight,
        # keep_last_solution=keep_last_solution,
        num_inference_iterations=cfg.controller.num_inference_iterations,
        num_control_samples=cfg.controller.num_control_samples,
        method=cfg.controller.method,
        mid_state=mid_state,
    )

    mode_optimiser = ModeOpt(
        start_state,
        target_state,
        env_name=cfg.env_name,
        dynamics=dynamics,
        mode_controller=controller,
    )

    plotting_callbacks = build_contour_plotter_callbacks(
        mode_optimiser, logging_epoch_freq=10
    )

    mode_optimiser.mode_controller_callback = plotting_callbacks
    mode_optimiser.optimise_mode_controller()

    # mode_optimiser_plotter = ModeOptContourPlotter(
    #     mode_optimiser=mode_optimiser,
    #     test_inputs=create_test_inputs(x_min=[-3, -3], x_max=[3, 3], input_dim=4),
    # )
    # mode_optimiser_plotter.plot_model()
    # plt.show()


def build_contour_plotter_callbacks(
    mode_optimiser: ModeOpt,
    logging_epoch_freq: Optional[int] = 3,
    log_dir: Optional[str] = "./logs",
) -> List[PlotFn]:
    test_inputs = create_test_inputs(x_min=[-3, -3], x_max=[3, 3], input_dim=4)

    mode_optimiser_plotter = ModeOptContourPlotter(
        mode_optimiser=mode_optimiser, test_inputs=test_inputs
    )
    gating_gps_plotting_cb = TensorboardImageCallbackScipy(
        plot_fn=mode_optimiser_plotter.plot_trajectories_over_gating_network_gps,
        logging_epoch_freq=logging_epoch_freq,
        log_dir=log_dir,
        name="Trajectories over gating function GPs",
    )
    mixing_probs_plotting_cb = TensorboardImageCallbackScipy(
        plot_fn=mode_optimiser_plotter.plot_trajectories_over_mixing_probs,
        logging_epoch_freq=logging_epoch_freq,
        log_dir=log_dir,
        name="Trajectories over mixing probabilities",
    )
    metric_trace_plotting_cb = TensorboardImageCallbackScipy(
        plot_fn=mode_optimiser_plotter.plot_trajectories_over_metric_trace,
        logging_epoch_freq=logging_epoch_freq,
        log_dir=log_dir,
        name="Trajectories over metric trace",
    )

    def plotting_callback(step, variables, value):
        mixing_probs_plotting_cb(step, variables, value)
        gating_gps_plotting_cb(step, variables, value)
        metric_trace_plotting_cb(step, variables, value)

    return plotting_callback


if __name__ == "__main__":

    collocation_trajectory_optimisation_via_constraints_from_cfg()
