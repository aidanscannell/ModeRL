#!/usr/bin/env python3
import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from moderl.controllers.base import TrajectoryOptimisationController
from moderl.custom_types import (
    ControlDim,
    ControlTrajectory,
    Dataset,
    Horizon,
    HorizonPlusOne,
    One,
    StateDim,
)
from moderl.dynamics.dynamics import ModeRLDynamics
from tensor_annotations.axes import Batch
from tf_agents.environments import py_environment


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


def rollout_ControlTrajectory_in_ModeRLDynamics(
    dynamics: ModeRLDynamics,
    control_trajectory: ControlTrajectory,
    start_state: ttf.Tensor2[One, StateDim],  # [1, StateDim]
) -> tfd.Distribution:  # [Horizon, StateDim]
    """Rollout a ControlTrajectory in dynamics"""
    # state_dist = tfd.Normal(loc=start_state, scale=0.0)
    state_dist = tfd.Deterministic(loc=start_state)
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
