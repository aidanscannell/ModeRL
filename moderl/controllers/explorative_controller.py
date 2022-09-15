#!/usr/bin/env python3
from typing import Callable, Optional

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from moderl.controllers.utils import (
    build_mode_variational_objective,
    initialise_deterministic_trajectory,
)
from moderl.cost_functions import (
    ControlQuadraticCostFunction,
    CostFunction,
    TargetStateCostFunction,
)
from moderl.custom_types import StateDim
from moderl.dynamics import ModeRLDynamics
from moderl.objectives import build_explorative_objective
from moderl.trajectories import ControlTrajectoryDist

from ..base import NonFeedbackController
from .constraints import build_mode_chance_constraints_scipy
from .trajectory_optimiser import TrajectoryOptimiser

tfd = tfp.distributions


class ExplorativeController(NonFeedbackController):
    def __init__(
        self,
        start_state: ttf.Tensor1[StateDim],
        dynamics: ModeRLDynamics,
        cost_fn: CostFunction,
        horizon: int = 10,
        max_iterations: int = 100,
        mode_satisfaction_prob: float = 0.8,
        keep_last_solution: bool = True,
        callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = None,
        method: Optional[str] = "SLSQP",
        name: str = "ExplorativeController",
    ):
        super().__init__(name=name)
        self.dynamics = dynamics

        self.initial_solution = self.find_initial_solution_in_desired_mode()

        self.mode_chance_constraints = build_mode_chance_constraints_scipy(
            dynamics,
            start_state,
            horizon,
            control_dim=self.initial_solution.control_dim,
            lower_bound=mode_satisfaction_prob,
        )

        explorative_objective = build_explorative_objective(
            dynamics, cost_fn, start_state
        )

        self.trajectory_optimiser = TrajectoryOptimiser(
            max_iterations=max_iterations,
            initial_solution=self.initial_solution,
            objective_fn=explorative_objective,
            keep_last_solution=keep_last_solution,
            constraints=[self.mode_chance_constraints],
            method=method,
        )

    def __call__(
        self, timestep: Optional[int] = None, variance: Optional[bool] = False
    ) -> ControlMeanAndVariance:
        # return self.previous_solution(timestep=timestep, variance=variance)
        if timestep is not None:
            idxs = [timestep, ...]
        else:
            idxs = [...]
        if variance:
            return self.controls[idxs], None
            # return self.controls[idxs], self.control_vars[idxs]
        else:
            return self.controls[idxs]

    # @property
    # def controls(self):
    #     return self.controls_posterior.mean()

    def optimise(self):
        optimisation_result = self.trajectory_optimiser.optimise(self.callback)

    def find_initial_solution_in_desired_mode(
        self,
        control_dim: int,
        fake_target_state,
        max_iterations: int = 1000,
        method: str = "SLSQP",
    ) -> ControlTrajectoryDist:
        state_dim = self.start_state.shape[-1]
        terminal_state_cost_matrix = tf.eye(state_dim, dtype=default_float())
        control_cost_matrix = tf.eye(control_dim, dtype=default_float())
        terminal_cost_fn = TargetStateCostFunction(
            weight_matrix=terminal_state_cost_matrix, target_state=fake_target_state
        )
        control_cost_fn = ControlQuadraticCostFunction(
            weight_matrix=control_cost_matrix
        )
        initial_cost_fn = terminal_cost_fn + control_cost_fn

        initial_solution = initialise_deterministic_trajectory(
            self.horizon, control_dim
        )
        objective_fn = build_mode_variational_objective(
            self.dynamics, initial_cost_fn, self.start_state
        )
        explorative_controller = TrajectoryOptimiser(
            max_iterations=max_iterations,
            initial_solution=initial_solution,
            objective_fn=objective_fn,
            method=method,
        )
        explorative_controller.optimise()
        return explorative_controller.previous_solution

    @property
    def initial_solution(self):
        return self.trajectory_optimiser.initial_solution

    @property
    def previous_solution(self):
        return self.trajectory_optimiser.previous_solution
