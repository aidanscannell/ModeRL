#!/usr/bin/env python3
import logging

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from moderl.custom_types import ControlTrajectory, State
from moderl.dynamics import ModeRLDynamics
from moderl.optimisers import TrajectoryOptimiser
from moderl.reward_functions import (
    ControlQuadraticRewardFunction,
    TargetStateRewardFunction,
)
from moderl.rollouts import rollout_ControlTrajectory_in_ModeRLDynamics

tfd = tfp.distributions


def find_solution_in_desired_mode(
    dynamics: ModeRLDynamics,
    horizon: int,
    control_dim: int,
    start_state: State,
    target_state_weight: float = 100,
) -> ControlTrajectory:
    initial_solution = ControlTrajectory(
        dist=tfd.Deterministic(
            tf.Variable(np.random.random((horizon, control_dim)) * 0.01)
        )
    )
    terminal_reward_fn = TargetStateRewardFunction(
        weight_matrix=target_state_weight
        * tf.eye(dynamics.state_dim, dtype=default_float()),
        target_state=start_state
        + tf.ones(start_state.shape, dtype=default_float()) * 0.2,
    )
    control_reward_fn = ControlQuadraticRewardFunction(
        weight_matrix=tf.eye(control_dim, dtype=default_float())
    )
    reward_fn = terminal_reward_fn + control_reward_fn

    def objective_fn(initial_solution: ControlTrajectory) -> ttf.Tensor0:
        state_dists = rollout_ControlTrajectory_in_ModeRLDynamics(
            dynamics=dynamics,
            control_trajectory=initial_solution,
            start_state=start_state,
        )
        control_dists = initial_solution()
        return reward_fn(state=state_dists, control=control_dists)

    trajectory_optimiser = TrajectoryOptimiser(
        max_iterations=1000,
        initial_solution=initial_solution,
        objective_fn=objective_fn,
        keep_last_solution=True,
        method="SLSQP",
    )
    trajectory_optimiser.optimise()
    logging.info("Found initial solution in desired dynamics mode")
    return trajectory_optimiser.previous_solution
