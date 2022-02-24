#!/usr/bin/env python3
from typing import List, Optional

import hydra
import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from modeopt.controllers import TrajectoryOptimisationController
from modeopt.cost_functions import (
    ControlQuadraticCostFunction,
    StateQuadraticCostFunction,
    quadratic_cost_fn,
)
from modeopt.custom_types import StateDim
from modeopt.dynamics import ModeOptDynamics
from modeopt.mode_opt import ModeOpt
from modeopt.monitor.callbacks import PlotFn, TensorboardImageCallbackScipy
from modeopt.plotting import ModeOptContourPlotter

# from modeopt.constraints import build_geodesic_collocation_constraints_closure
from modeopt.objectives import build_geodesic_collocation_lagrange_objective
from modeopt.trajectories import (
    # FlatOutputTrajectory,
    GeodesicTrajectory,
    # VelocityControlledFlatOutputTrajectory,
)
from omegaconf import DictConfig

from .mode_opt_riemannian_energy_traj_opt import create_test_inputs

tfd = tfp.distributions

meaning_of_life = 42
tf.random.set_seed(meaning_of_life)
np.random.seed(meaning_of_life)


@hydra.main(
    config_path="keras_configs/scenario_7/trajectory_optimisation",
    # config_name="geodesic_collocation",
    config_name="geodesic_collocation_high_cov_weight",
)
def geodesic_collocation_trajectory_optimisation_via_constraints_from_cfg(
    cfg: DictConfig,
):
    dynamics = tf.keras.models.load_model(
        cfg.dynamics.ckpt_dir, custom_objects={"ModeOptDynamics": ModeOptDynamics}
    )
    dynamics.desired_mode = cfg.dynamics.desired_mode  # update desired mode
    # env_name = "velocity-controlled-point-mass/scenario-" + str(cfg.scenario)
    geodesic_collocation_trajectory_optimisation_via_constraints(
        tf.reshape(tf.constant(cfg.start_state, dtype=default_float()), shape=(1, -1)),
        tf.reshape(tf.constant(cfg.target_state, dtype=default_float()), shape=(1, -1)),
        cfg.env_name,
        cfg.num_iterations,
        cfg.horizon,
        cfg.dynamics.control_dim,
        dynamics,
        cfg.collocation_constraints.riemannian_metric_covariance_weight,
        tf.constant(cfg.cost_fn.state_cost_matrix, dtype=default_float()),
        cfg.collocation_constraints.collocation_constraints_lower,
        cfg.collocation_constraints.collocation_constraints_upper,
        cfg.t_init,
        cfg.t_end,
    )


@hydra.main(
    config_path="keras_configs/scenario_7/trajectory_optimisation",
    config_name="geodesic_collocation_lagrange",
)
def geodesic_collocation_trajectory_optimisation_via_lagrange_from_cfg(cfg: DictConfig):
    dynamics = tf.keras.models.load_model(
        cfg.dynamics.ckpt_dir, custom_objects={"ModeOptDynamics": ModeOptDynamics}
    )
    dynamics.desired_mode = cfg.dynamics.desired_mode  # update desired mode
    # env_name = "velocity-controlled-point-mass/scenario-" + str(cfg.scenario)
    geodesic_collocation_trajectory_optimisation_via_lagrange(
        tf.reshape(tf.constant(cfg.start_state, dtype=default_float()), shape=(1, -1)),
        tf.reshape(tf.constant(cfg.target_state, dtype=default_float()), shape=(1, -1)),
        cfg.env_name,
        cfg.num_iterations,
        cfg.horizon,
        cfg.dynamics.control_dim,
        dynamics,
        cfg.cost_fn.riemannian_metric_covariance_weight,
        tf.constant(cfg.cost_fn.state_cost_matrix, dtype=default_float()),
        cfg.t_init,
        cfg.t_end,
    )


def geodesic_collocation_trajectory_optimisation_via_constraints(
    start_state,
    target_state,
    env_name: str,
    num_iterations: int,
    horizon: int,
    control_dim: int,
    dynamics: ModeOptDynamics,
    riemannian_metric_covariance_weight: float,
    state_cost_weight: ttf.Tensor2[StateDim, StateDim],
    collocation_constraints_lower: List[float] = -0.1,
    collocation_constraints_upper: List[float] = 0.1,
    t_init: float = -1.0,
    t_end: float = 1.0,
):
    initial_solution = GeodesicTrajectory(
        start_state=start_state,
        target_state=target_state,
        gp=dynamics.desired_mode_gating_gp,
        riemannian_metric_covariance_weight=riemannian_metric_covariance_weight,
        horizon=horizon,
        t_init=t_init,
        t_end=t_end,
    )

    def objective_fn(initial_solution: GeodesicTrajectory):
        """Dummy cost function that regularises the trajectory"""
        costs = quadratic_cost_fn(
            # vector=initial_solution.states,
            vector=initial_solution.state_derivatives,
            weight_matrix=state_cost_weight,
            vector_var=None,
        )
        return tf.reduce_sum(costs)

    controller = TrajectoryOptimisationController(
        num_iterations=num_iterations,
        initial_solution=initial_solution,
        objective_fn=objective_fn,
        constraints_lower_bound=collocation_constraints_lower,
        constraints_upper_bound=collocation_constraints_upper,
        # keep_last_solution=keep_last_solution,
        # constraints=collocation_constraints,
        nonlinear_constraint_closure=initial_solution.geodesic_collocation_constraints,
        nonlinear_constraint_kwargs={
            "lb": collocation_constraints_lower,
            "ub": collocation_constraints_upper,
        },
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
    controller.optimise(callback=plotting_callbacks)

    # mode_optimiser_plotter = ModeOptContourPlotter(
    #     mode_optimiser=mode_optimiser,
    #     test_inputs=create_test_inputs(x_min=[-3, -3], x_max=[3, 3], input_dim=4),
    # )
    # mode_optimiser_plotter.plot_model()
    # plt.show()


def geodesic_collocation_trajectory_optimisation_via_lagrange(
    start_state,
    target_state,
    env_name: str,
    num_iterations: int,
    horizon: int,
    control_dim: int,
    dynamics: ModeOptDynamics,
    riemannian_metric_covariance_weight: float,
    state_cost_weight: ttf.Tensor2[StateDim, StateDim],
    t_init: float = -1.0,
    t_end: float = 1.0,
):
    # control_cost_fn = ControlQuadraticCostFunction(weight_matrix=control_cost_weight)
    # cost_fn = control_cost_fn
    # objective_fn = build_variational_objective(dynamics, cost_fn, start_state)

    # initial_solution = initialise_flat_trajectory(horizon, control_dim)
    #
    cost_fn = StateQuadraticCostFunction(weight_matrix=state_cost_weight / 100)
    # control_cost_fn = ControlQuadraticCostFunction(weight_matrix=control_cost_weight)
    # cost_fn = ControlQuadraticCostFunction(weight_matrix=state_cost_weight * 100)

    initial_solution = VelocityControlledFlatOutputTrajectory(
        start_state=start_state,
        target_state=target_state,
        horizon=horizon,
        t_init=t_init,
        t_end=t_end,
        lagrange_multipliers=True,
    )

    objective_fn = build_geodesic_collocation_lagrange_objective(
        gp=dynamics.desired_mode_gating_gp,
        covariance_weight=riemannian_metric_covariance_weight,
        cost_fn=cost_fn,
    )

    controller = TrajectoryOptimisationController(
        num_iterations=num_iterations,
        initial_solution=initial_solution,
        objective_fn=objective_fn,
        # constraints_lower_bound=collocation_constraints_lower,
        # constraints_upper_bound=collocation_constraints_upper,
        # keep_last_solution=keep_last_solution,
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
    controller.optimise(callback=plotting_callbacks)

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

    geodesic_collocation_trajectory_optimisation_via_constraints_from_cfg()
    # geodesic_collocation_trajectory_optimisation_via_lagrange_from_cfg()
