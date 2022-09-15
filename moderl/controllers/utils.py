#!/usr/bin/env python3
from typing import List, Optional, Union

import tensor_annotations.tensorflow as ttf
import tensorflow_probability as tfp
from moderl.cost_functions import (
    ControlQuadraticCostFunction,
    RiemannianEnergyCostFunction,
    TargetStateCostFunction,
)
from moderl.custom_types import ControlDim, StateDim
from moderl.dynamics import ModeRLDynamics
from moderl.objectives import (
    build_variational_objective,
    build_mode_variational_objective,
)
from moderl.trajectories import (
    initialise_deterministic_trajectory,
    initialise_gaussian_trajectory,
)
from scipy.optimize import LinearConstraint, NonlinearConstraint

from .non_feedback.trajectory_optimisation import TrajectoryOptimisationController

tfd = tfp.distributions


def build_riemannian_energy_controller(
    start_state,
    target_state,
    dynamics: ModeRLDynamics,
    desired_mode: int,
    horizon: int,
    control_dim: int,
    control_cost_matrix: Optional[ttf.Tensor2[ControlDim, ControlDim]],
    terminal_state_cost_matrix: Optional[ttf.Tensor2[StateDim, StateDim]],
    riemannian_metric_cost_matrix,
    riemannian_metric_covariance_weight: float,
    max_iterations: int,
    constraints_lower_bound: float,
    constraints_upper_bound: float,
    constraints: Optional[List[Union[LinearConstraint, NonlinearConstraint]]] = [],
    keep_last_solution: bool = True,
    method: str = "SLSQP",
):
    dynamics.desired_mode = desired_mode  # update desired mode

    energy_cost_fn = RiemannianEnergyCostFunction(
        gp=dynamics.gating_gp,
        riemannian_metric_weight_matrix=riemannian_metric_cost_matrix,
        covariance_weight=riemannian_metric_covariance_weight,
    )
    terminal_cost_fn = TargetStateCostFunction(
        weight_matrix=terminal_state_cost_matrix, target_state=target_state
    )
    control_cost_fn = ControlQuadraticCostFunction(weight_matrix=control_cost_matrix)
    cost_fn = energy_cost_fn + terminal_cost_fn + control_cost_fn
    objective_fn = build_variational_objective(dynamics, cost_fn, start_state)

    initial_solution = initialise_deterministic_trajectory(horizon, control_dim)

    controller = TrajectoryOptimisationController(
        max_iterations=max_iterations,
        initial_solution=initial_solution,
        objective_fn=objective_fn,
        constraints_lower_bound=constraints_lower_bound,
        constraints_upper_bound=constraints_upper_bound,
        keep_last_solution=keep_last_solution,
        constraints=constraints,
        method=method,
    )
    return controller


def build_control_as_inference_controller(
    start_state,
    target_state,
    dynamics: ModeRLDynamics,
    desired_mode: int,
    horizon: int,
    control_dim: int,
    control_cost_matrix: Optional[ttf.Tensor2[ControlDim, ControlDim]],
    terminal_state_cost_matrix: Optional[ttf.Tensor2[StateDim, StateDim]],
    max_iterations: int,
    # constraints_lower_bound: float,
    # constraints_upper_bound: float,
    constraints: Optional[List[Union[LinearConstraint, NonlinearConstraint]]] = [],
    keep_last_solution: bool = True,
    method: str = "SLSQP",
    gaussian_controls: bool = True,
):
    dynamics.desired_mode = desired_mode  # update desired mode
    terminal_cost_fn = TargetStateCostFunction(
        weight_matrix=terminal_state_cost_matrix, target_state=target_state
    )
    control_cost_fn = ControlQuadraticCostFunction(weight_matrix=control_cost_matrix)
    cost_fn = terminal_cost_fn + control_cost_fn

    objective_fn = build_mode_variational_objective(dynamics, cost_fn, start_state)

    if gaussian_controls:
        initial_solution = initialise_gaussian_trajectory(horizon, control_dim)
    else:
        initial_solution = initialise_deterministic_trajectory(horizon, control_dim)

    controller = TrajectoryOptimisationController(
        max_iterations=max_iterations,
        initial_solution=initial_solution,
        objective_fn=objective_fn,
        # constraints_lower_bound=constraints_lower_bound,
        # constraints_upper_bound=constraints_upper_bound,
        keep_last_solution=keep_last_solution,
        constraints=constraints,
        method=method,
    )
    return controller


# class RiemannianEnergyController(NonFeedbackController):
#     def __init__(
#         self,
#         # start_state: ttf.Tensor1[StateDim],
#         target_state: ttf.Tensor1[StateDim],
#         dynamics: ModeRLDynamics,
#         # Collocation args
#         horizon: int = 10,
#         t_init: float = -1.0,
#         t_end: float = 1.0,
#         riemannian_metric_covariance_weight: float = 1.0,
#         max_iterations: int = 100,
#         # collocation_constraints_lower: float = -0.1,
#         # collocation_constraints_upper: float = 0.1,
#         control_cost_weight: Optional[ttf.Tensor2[ControlDim, ControlDim]] = None,
#         terminal_state_cost_weight: Optional[ttf.Tensor2[StateDim, StateDim]] = None,
#         keep_last_solution: bool = True,
#         method: Optional[str] = "SLSQP",
#     ):

#         initial_solution = initialise_deterministic_trajectory(horizon, control_dim)

#         self.trajectory_optimiser = TrajectoryOptimisationController(
#             max_iterations,
#             initial_solution,
#             objective_fn,
#             keep_last_solution=keep_last_solution,
#             method=method,
#         )

#         gpf.utilities.print_summary(self)

#     def __call__(
#         self, timestep: Optional[int] = None, variance: bool = False
#     ) -> ControlMeanAndVariance:
#         # return self.previous_solution(timestep=timestep, variance=variance)
#         if timestep is not None:
#             idxs = [timestep, ...]
#         else:
#             idxs = [...]
#         if variance:
#             return self.controls[idxs], None
#             # return self.controls[idxs], self.control_vars[idxs]
#         else:
#             return self.controls[idxs]

#     @property
#     def controls(self):
#         return self.controls_posterior.mean()

#     def optimise(
#         self, callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = []
#     ):
#         # Optimise state trajectory
#         optimisation_result = self.trajectory_optimiser.optimise(callback)
#         self.infer_controls_from_states(callback, num_steps=optimisation_result.nit)

#     @property
#     def initial_solution(self):
#         return self.trajectory_optimiser.initial_solution

#     @property
#     def previous_solution(self):
#         return self.trajectory_optimiser.previous_solution

#     def get_config(self):
#         return {
#             "start_state": self.initial_solution.start_state.numpy(),
#             "target_state": self.initial_solution.target_state.numpy(),
#             "dynamics": tf.keras.utils.serialize_keras_object(self.dynamics),
#             "horizon": self.initial_solution.horizon,
#             "t_init": self.initial_solution.times[0].numpy(),
#             "t_end": self.initial_solution.times[-1].numpy(),
#             "riemannian_metric_covariance_weight": self.initial_solution.manifold.covariance_weight,
#             "max_collocation_iterations": self.trajectory_optimiser.max_iterations,
#             "collocation_constraints_lower": self.collocation_constraints_lower,
#             "collocation_constraints_upper": self.collocation_constraints_upper,
#             "dummy_cost_weight": self.dummy_cost_weight.numpy(),
#             "keep_last_solution": self.trajectory_optimiser.keep_last_solution,
#             "num_inference_iterations": self.num_inference_iterations,
#             "num_control_samples": self.num_control_samples,
#             "method": self.trajectory_optimiser.method,
#             # "initial_solution": tf.keras.utils.serialize_keras_object(
#             #     self.initial_solution
#             # ),
#             # "previous_solution": tf.keras.utils.serialize_keras_object(
#             #     self.previous_solution
#             # ),
#         }

#     @classmethod
#     def from_config(cls, cfg: dict):
#         # initial_solution = tf.keras.layers.deserialize(
#         #     cfg["initial_solution"], custom_objects=TRAJECTORY_OBJECTS
#         # )
#         # previous_solution = tf.keras.layers.deserialize(
#         #     cfg["previous_solution"], custom_objects=TRAJECTORY_OBJECTS
#         # )
#         dynamics = tf.keras.layers.deserialize(
#             cfg["dynamics"], custom_objects={"ModeRLDynamics": ModeRLDynamics}
#         )
#         return cls(
#             start_state=tf.constant(cfg["start_state"]),
#             target_state=tf.constant(cfg["target_state"]),
#             dynamics=dynamics,
#             horizon=cfg["horizon"],
#             t_init=cfg["t_init"],
#             t_end=cfg["t_end"],
#             riemannian_metric_covariance_weight=cfg[
#                 "riemannian_metric_covariance_weight"
#             ],
#             max_collocation_iterations=cfg["max_collocation_iterations"],
#             collocation_constraints_lower=cfg["collocation_constraints_lower"],
#             collocation_constraints_upper=cfg["collocation_constraints_upper"],
#             dummy_cost_weight=cfg["dummy_cost_weight"],
#             keep_last_solution=cfg["keep_last_solution"],
#             num_inference_iterations=cfg["num_inference_iterations"],
#             num_control_samples=cfg["num_control_samples"],
#             method=cfg["method"],
#         )
#         # controller.previous_solution = previous_solution
#         # return controller
