#!/usr/bin/env python3
from typing import List, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from modeopt.controllers import TrajectoryOptimisationController
from modeopt.cost_functions import (
    ControlQuadraticCostFunction,
    RiemannianEnergyCostFunction,
    TargetStateCostFunction,
)
from modeopt.custom_types import StateDim
from modeopt.dynamics import ModeOptDynamics
from modeopt.mode_opt import ModeOpt
from modeopt.monitor.callbacks import PlotFn, TensorboardImageCallbackScipy
from modeopt.objectives import build_variational_objective
from modeopt.plotting import ModeOptContourPlotter
from modeopt.trajectories import (
    initialise_deterministic_trajectory,
    initialise_gaussian_trajectory,
)
from omegaconf import DictConfig

tfd = tfp.distributions

meaning_of_life = 42
tf.random.set_seed(meaning_of_life)
np.random.seed(meaning_of_life)


@hydra.main(
    config_path="keras_configs/scenario_7/trajectory_optimisation",
    # config_path="keras_configs/scenario_5/trajectory_optimisation",
    config_name="riemannian_energy",
)
def riemannian_energy_trajectory_optimisation_from_cfg(cfg: DictConfig):
    dynamics = tf.keras.models.load_model(
        cfg.dynamics.ckpt_dir, custom_objects={"ModeOptDynamics": ModeOptDynamics}
    )
    dynamics.desired_mode = cfg.dynamics.desired_mode  # update desired mode
    # env_name = "velocity-controlled-point-mass/scenario-" + str(cfg.scenario)
    riemannian_energy_trajectory_optimisation(
        tf.reshape(tf.constant(cfg.start_state, dtype=default_float()), shape=(1, -1)),
        tf.reshape(tf.constant(cfg.target_state, dtype=default_float()), shape=(1, -1)),
        cfg.env_name,
        cfg.num_iterations,
        cfg.horizon,
        cfg.dynamics.control_dim,
        dynamics,
        tf.constant(cfg.cost_fn.riemannian_metric_cost_matrix, dtype=default_float()),
        cfg.cost_fn.riemannian_metric_covariance_weight,
        tf.constant(cfg.cost_fn.terminal_state_cost_matrix, dtype=default_float()),
        tf.constant(cfg.cost_fn.control_cost_matrix, dtype=default_float()),
        cfg.control_constraints_lower,
        cfg.control_constraints_upper,
        cfg.gaussian_controls,
    )


def riemannian_energy_trajectory_optimisation(
    start_state,
    target_state,
    env_name: str,
    num_iterations: int,
    horizon: int,
    control_dim: int,
    dynamics: ModeOptDynamics,
    riemannian_metric_cost_matrix: ttf.Tensor2[StateDim, StateDim],
    riemannian_metric_covariance_weight: float,
    terminal_state_cost_matrix: ttf.Tensor2[StateDim, StateDim],
    control_cost_weight,
    control_constraints_lower: List[float],
    control_constraints_upper: List[float],
    gaussian_controls: bool = False,
):
    energy_cost_fn = RiemannianEnergyCostFunction(
        gp=dynamics.gating_gp,
        riemannian_metric_weight_matrix=riemannian_metric_cost_matrix,
        covariance_weight=riemannian_metric_covariance_weight,
    )
    terminal_cost_fn = TargetStateCostFunction(
        weight_matrix=terminal_state_cost_matrix, target_state=target_state
    )
    control_cost_fn = ControlQuadraticCostFunction(weight_matrix=control_cost_weight)
    cost_fn = energy_cost_fn + terminal_cost_fn + control_cost_fn
    # cost_fn = terminal_cost_fn + control_cost_fn
    objective_fn = build_variational_objective(dynamics, cost_fn, start_state)

    if gaussian_controls:
        initial_solution = initialise_gaussian_trajectory(horizon, control_dim)
    else:
        initial_solution = initialise_deterministic_trajectory(horizon, control_dim)

    controller = TrajectoryOptimisationController(
        num_iterations=num_iterations,
        initial_solution=initial_solution,
        objective_fn=objective_fn,
        constraints_lower_bound=control_constraints_lower,
        constraints_upper_bound=control_constraints_upper,
        # keep_last_solution=keep_last_solution,
        # constraints=constraints,
        # method=method,
    )

    mode_optimiser = ModeOpt(
        start_state,
        target_state,
        env_name=env_name,
        dynamics=dynamics,
        mode_controller=controller,
    )

    plotting_callbacks = build_contour_plotter_callbacks(mode_optimiser)
    # controller.optimise(callback=plotting_callbacks[0])
    controller.optimise(callback=plotting_callbacks)
    mode_optimiser_plotter = ModeOptContourPlotter(
        mode_optimiser=mode_optimiser,
        test_inputs=create_test_inputs(x_min=[-3, -3], x_max=[3, 3], input_dim=4),
    )
    mode_optimiser_plotter.plot_model()
    plt.show()


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

    def plotting_callback(step, variables, value):
        mixing_probs_plotting_cb(step, variables, value)
        gating_gps_plotting_cb(step, variables, value)

    return plotting_callback


def create_test_inputs(
    x_min=[-3, -3], x_max=[3, 3], input_dim=4, num_test: int = 400, factor: float = 1.2
):
    sqrtN = int(np.sqrt(num_test))
    xx = np.linspace(x_min[0] * factor, x_max[0] * factor, sqrtN)
    yy = np.linspace(x_min[1] * factor, x_max[1] * factor, sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    if input_dim > 2:
        zeros = np.zeros((num_test, input_dim - 2))
        test_inputs = np.concatenate([test_inputs, zeros], -1)
    return test_inputs


if __name__ == "__main__":

    riemannian_energy_trajectory_optimisation_from_cfg()
