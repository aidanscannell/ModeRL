#!/usr/bin/env python3
from typing import List, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from modeopt.controllers.non_feedback.trajectory_optimisation import (
    TrajectoryOptimisationController,
)
from modeopt.controllers.utils import build_riemannian_energy_controller
from modeopt.cost_functions import ControlQuadraticCostFunction, TargetStateCostFunction
from modeopt.dynamics import ModeOptDynamics
from modeopt.mode_opt import ModeOpt
from modeopt.monitor.callbacks import PlotFn, TensorboardImageCallbackScipy
from modeopt.objectives import build_variational_objective
from modeopt.plotting import ModeOptContourPlotter
from modeopt.trajectories import initialise_deterministic_trajectory
from omegaconf import DictConfig

tfd = tfp.distributions

meaning_of_life = 42
tf.random.set_seed(meaning_of_life)
np.random.seed(meaning_of_life)


@hydra.main(
    config_path="keras_configs/scenario_7/trajectory_optimisation",
    config_name="svgp_baseline_desired_mode",
    # config_name="svgp_baseline_both_modes",
)
def svgp_baseline_trajectory_optimisation_from_cfg(cfg: DictConfig):
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
    terminal_cost_fn = TargetStateCostFunction(
        weight_matrix=tf.constant(
            cfg.cost_fn.terminal_state_cost_matrix, dtype=default_float()
        ),
        target_state=target_state,
    )
    control_cost_fn = ControlQuadraticCostFunction(
        weight_matrix=tf.constant(
            cfg.cost_fn.control_cost_matrix, dtype=default_float()
        )
    )
    cost_fn = terminal_cost_fn + control_cost_fn
    objective_fn = build_variational_objective(dynamics, cost_fn, start_state)

    initial_solution = initialise_deterministic_trajectory(
        cfg.horizon, cfg.dynamics.control_dim
    )

    controller = TrajectoryOptimisationController(
        max_iterations=cfg.max_iterations,
        initial_solution=initial_solution,
        objective_fn=objective_fn,
        # constraints_lower_bound=list(cfg.control_constraints_lower),
        # constraints_upper_bound=list(cfg.control_constraints_upper),
        keep_last_solution=True,
        constraints=[],
        method=cfg.method,
    )

    mode_optimiser = ModeOpt(
        start_state,
        target_state,
        env_name=cfg.env_name,
        dynamics=dynamics,
        desired_mode=cfg.dynamics.desired_mode,
        mode_controller=controller,
    )

    plotting_callbacks = build_contour_plotter_callbacks(
        mode_optimiser, logging_freq=cfg.logging_freq
    )
    mode_optimiser.mode_controller_callback = plotting_callbacks
    mode_optimiser.optimise_mode_controller()
    mode_optimiser.save()

    # test_inputs = create_test_inputs(x_min=[-3, -3], x_max=[3, 3], input_dim=4)
    # mode_optimiser_plotter = ModeOptContourPlotter(
    #     mode_optimiser=mode_optimiser, test_inputs=test_inputs
    # )
    # # mode_optimiser_plotter.plot_model()
    # mode_optimiser_plotter.plot_trajectories_over_gating_network_gps()
    # mode_optimiser_plotter.plot_trajectories_over_desired_gating_network_gp()
    # mode_optimiser_plotter.plot_trajectories_over_mixing_probs()
    # plt.show()


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


def create_test_inputs(
    x_min=[-3, -3], x_max=[3, 3], input_dim=4, num_test: int = 1600, factor: float = 1.2
):
    sqrtN = int(np.sqrt(num_test))
    # xx = np.linspace(x_min[0] * factor, x_max[0] * factor, sqrtN)
    xx = np.linspace(x_min[0] * factor, x_max[0], sqrtN)
    yy = np.linspace(x_min[1] * factor, x_max[1] * factor, sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    if input_dim > 2:
        zeros = np.zeros((num_test, input_dim - 2))
        test_inputs = np.concatenate([test_inputs, zeros], -1)
    return test_inputs


if __name__ == "__main__":

    svgp_baseline_trajectory_optimisation_from_cfg()
