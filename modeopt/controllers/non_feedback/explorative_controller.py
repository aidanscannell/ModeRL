#!/usr/bin/env python3
from typing import Callable, Optional

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from modeopt.constraints import build_mode_chance_constraints_scipy
from modeopt.controllers.utils import (
    build_mode_variational_objective,
    initialise_deterministic_trajectory,
)
from modeopt.cost_functions import (
    ControlQuadraticCostFunction,
    StateDiffCostFunction,
    TargetStateCostFunction,
)
from modeopt.custom_types import StateDim
from modeopt.dynamics import ModeOptDynamics
from modeopt.objectives import build_explorative_objective
from modeopt.trajectories import ControlTrajectoryDist

from ..base import NonFeedbackController
from .trajectory_optimisation import TrajectoryOptimisationController

tfd = tfp.distributions


def find_initial_solution_in_desired_mode(
    dynamics: ModeOptDynamics,
    start_state,
    horizon: int,
    control_dim: int,
    fake_target_state,
    max_iterations: int = 1000,
    method: str = "SLSQP",
) -> ControlTrajectoryDist:
    state_dim = start_state.shape[-1]
    terminal_state_cost_matrix = tf.eye(state_dim, dtype=default_float())
    control_cost_matrix = tf.eye(control_dim, dtype=default_float())
    terminal_cost_fn = TargetStateCostFunction(
        weight_matrix=terminal_state_cost_matrix, target_state=fake_target_state
    )
    control_cost_fn = ControlQuadraticCostFunction(weight_matrix=control_cost_matrix)
    initial_cost_fn = terminal_cost_fn + control_cost_fn

    initial_solution = initialise_deterministic_trajectory(horizon, control_dim)
    objective_fn = build_mode_variational_objective(
        dynamics, initial_cost_fn, start_state
    )
    explorative_controller = TrajectoryOptimisationController(
        max_iterations=max_iterations,
        initial_solution=initial_solution,
        objective_fn=objective_fn,
        method=method,
    )
    explorative_controller.optimise()
    return explorative_controller.previous_solution


class ExplorativeController(NonFeedbackController):
    def __init__(
        self,
        start_state: ttf.Tensor1[StateDim],
        target_state: ttf.Tensor1[StateDim],
        dynamics: ModeOptDynamics,
        horizon: int = 10,
        max_iterations: int = 100,
        mode_satisfaction_prob: float = 0.8,
        terminal_state_cost_weight: float = 100.0,
        state_diff_cost_weight: float = 1.0,
        control_cost_weight: float = 1.0,
        keep_last_solution: bool = True,
        method: Optional[str] = "SLSQP",
        name: str = "ExplorativeController",
    ):
        super().__init__(name=name)
        self.dynamics = dynamics

        self.initial_solution = find_initial_solution_in_desired_mode()

        self.mode_chance_constraints = build_mode_chance_constraints_scipy(
            dynamics,
            start_state,
            horizon,
            control_dim=self.initial_solution.control_dim,
            lower_bound=mode_satisfaction_prob,
        )

        terminal_cost_fn = TargetStateCostFunction(
            weight_matrix=tf.eye(dynamics.state_dim, dtype=default_float())
            * terminal_state_cost_weight,
            target_state=start_state,
        )
        control_cost_fn = ControlQuadraticCostFunction(
            weight_matrix=tf.eye(
                self.initial_solution.control_dim, dtype=default_float()
            )
            * control_cost_weight
        )
        state_diff_cost_fn = StateDiffCostFunction(
            weight_matrix=tf.eye(dynamics.state_dim, dtype=default_float())
            * state_diff_cost_weight
        )

        cost_fn = control_cost_fn + state_diff_cost_fn + terminal_cost_fn
        explorative_objective = build_explorative_objective(
            dynamics, cost_fn, start_state
        )

        self.trajectory_optimiser = TrajectoryOptimisationController(
            max_iterations=max_iterations,
            initial_solution=self.initial_solution,
            objective_fn=explorative_objective,
            keep_last_solution=keep_last_solution,
            constraints=[self.mode_chance_constraints],
            method=method,
        )

    # def __call__(
    #     self, timestep: Optional[int] = None, variance: bool = False
    # ) -> ControlMeanAndVariance:
    #     # return self.previous_solution(timestep=timestep, variance=variance)
    #     if timestep is not None:
    #         idxs = [timestep, ...]
    #     else:
    #         idxs = [...]
    #     if variance:
    #         return self.controls[idxs], None
    #         # return self.controls[idxs], self.control_vars[idxs]
    #     else:
    #         return self.controls[idxs]

    # @property
    # def controls(self):
    #     return self.controls_posterior.mean()

    def optimise(
        self, callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = []
    ):
        # Optimise state trajectory
        optimisation_result = self.trajectory_optimiser.optimise(callback)

    @property
    def initial_solution(self):
        return self.trajectory_optimiser.initial_solution

    @property
    def previous_solution(self):
        return self.trajectory_optimiser.previous_solution

    def get_config(self):
        return {
            "start_state": self.initial_solution.start_state.numpy(),
            "target_state": self.initial_solution.target_state.numpy(),
            "dynamics": tf.keras.utils.serialize_keras_object(self.dynamics),
            "horizon": self.initial_solution.horizon,
            "max_iterations": self.trajectory_optimiser.max_iterations,
            "keep_last_solution": self.trajectory_optimiser.keep_last_solution,
            "method": self.trajectory_optimiser.method,
        }

    @classmethod
    def from_config(cls, cfg: dict):
        dynamics = tf.keras.layers.deserialize(
            cfg["dynamics"], custom_objects={"ModeOptDynamics": ModeOptDynamics}
        )
        controller = cls(
            start_state=tf.constant(cfg["start_state"], dtype=default_float()),
            target_state=tf.constant(cfg["target_state"], dtype=default_float()),
            dynamics=dynamics,
            horizon=cfg["horizon"],
            max_iterations=cfg["max_iterations"],
            keep_last_solution=cfg["keep_last_solution"],
            method=cfg["method"],
        )
        return controller
