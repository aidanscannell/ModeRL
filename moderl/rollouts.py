#!/usr/bin/env python3
import typing
from typing import Tuple, Union

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from tensor_annotations import axes
from tensor_annotations.axes import Batch
from moderl.controllers import ControllerInterface
from moderl.controllers.base import TrajectoryOptimisationController

from tf_agents.environments import py_environment

# py_environment = None

from moderl.custom_types import (
    ControlDim,
    ControlTrajectory,
    Dataset,
    Horizon,
    One,
    StateDim,
    HorizonPlusOne,
)
from moderl.dynamics.dynamics import ModeRLDynamics

tfd = tfp.distributions


def rollout_trajectory_optimisation_controller_in_env(
    env: py_environment.PyEnvironment,
    start_state: ttf.Tensor2[Batch, StateDim],
    controller: TrajectoryOptimisationController,
) -> ttf.Tensor2[HorizonPlusOne, StateDim]:
    """Rollout ExplorativeController in environment"""
    states = tf.identity(start_state)
    # controls = controller()
    env.state_init = states
    env.reset()

    for t in range(controller.horizon):
        # next_time_step = env.step(controls[t])
        next_time_step = env.step(controller(timestep=t))
        states = tf.concat([states, next_time_step.observation], axis=0)
    return tf.stack(states)


# def rollout_ExplorativeController_in_ModeRLDynamics(
#     dynamics: ModeRLDynamics,
#     controller: ExplorativeController,
#     start_state: ttf.Tensor2[One, StateDim],  # [1, StateDim]
# ) -> tfd.Distribution:  # [Horizon, StateDim]
#     """Rollout a ControlTrajectory in dynamics"""
#     # if not isinstance(control_trajectory, ControlTrajectory):
#     if not isinstance(controller, ExplorativeController):
#         raise NotImplementedError
#     control_trajectory = controller()
#     return rollout_ControlTrajectory_in_ModeRLDynamics(
#         dynamics=dynamics,
#         control_trajectory=control_trajectory,
#         start_state=start_state,
#     )


def rollout_ControlTrajectory_in_ModeRLDynamics(
    dynamics: ModeRLDynamics,
    control_trajectory: ControlTrajectory,
    start_state: ttf.Tensor2[One, StateDim],  # [1, StateDim]
) -> tfd.Distribution:  # [Horizon, StateDim]
    """Rollout a ControlTrajectory in dynamics"""
    state_dist = tfd.Normal(loc=start_state, scale=0.0)
    state_means = state_dist.mean()
    state_vars = state_dist.variance()
    for t in range(control_trajectory.horizon):
        state_dist = dynamics.forward(
            state=state_dist,
            control=control_trajectory(timestep=t),
            predict_state_difference=False,
        )
        state_means = tf.concat([state_means, state_dist.mean()], 0)
        state_vars = tf.concat([state_vars, state_dist.variance()], 0)
    return tfd.Normal(loc=state_means, scale=tf.math.sqrt(state_vars))


# def rollout_controls_in_env(
#     env: py_environment.PyEnvironment,
#     start_state: ttf.Tensor2[Batch, StateDim],
#     controls: ttf.Tensor2[Horizon, ControlDim],
# ) -> ttf.Tensor2[HorizonPlusOne, StateDim]:
#     """Rollout a controls in environment

#     :returns: states
#     """
#     states = tf.identity(start_state)
#     env.state_init = states
#     env.reset()
#     horizon = controls.shape[0]

#     for t in range(horizon):
#         next_time_step = env.step(controls[t])
#         states = tf.concat([states, next_time_step.observation], axis=0)
#     return tf.stack(states)


def collect_data_from_env(
    env: py_environment.PyEnvironment,
    start_state: ttf.Tensor2[Batch, StateDim],
    controls: ttf.Tensor2[Horizon, ControlDim],
) -> Dataset:
    """Rollout a control trajectory in environment

    :returns: (states, delta_states)
    """
    if isinstance(start_state, tf.Tensor):
        state = start_state.numpy()
    else:
        state = start_state.copy()
    if len(state.shape) == 2:
        state = state.reshape(-1)

    if isinstance(controls, tf.Tensor):
        controls = controls.numpy()

    env.state_init = state
    env.reset()

    state_control_inputs, delta_state_outputs = [], []
    for t in range(controls.shape[0]):
        state_control_inputs.append(np.concatenate([state, controls[t, :]], -1))
        delta_state_outputs.append(env.transition_dynamics(state, controls[t, :]))
        state = state + delta_state_outputs[-1]
    return np.stack(state_control_inputs, 0), np.stack(delta_state_outputs, 0)


# def rollout_controller_in_env(
#     env: py_environment.PyEnvironment,
#     controller: ControllerInterface,
#     horizon: int,
#     start_state: ttf.Tensor2[One, StateDim],
# ) -> Tuple[
#     ttf.Tensor2[HorizonPlusOne, StateDim], ttf.Tensor2[HorizonPlusOne, StateDim]
# ]:
#     """Rollout a given set of control means and vars

#     :returns: (states_means, state_vars)
#     """
#     # TODO fix this to not use NonFeedbackController
#     if isinstance(controls, tf.Tensor):
#         controls = controls.numpy()
#     return rollout_controls_in_env(env=env, start_state=start_state, controls=controls)

# states = tf.identity(start_state)
# env.state_init = states
# env.reset()

# for t in range(controller.horizon):
#     next_time_step = env.step(controller(states[-1, :], t))
#     states = tf.concat([states, next_time_step.observation], axis=0)
# return tf.stack(states)
