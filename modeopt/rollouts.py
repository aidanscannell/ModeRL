#!/usr/bin/env python3
import typing
import numpy as np
from typing import Callable

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import default_float
from tensor_annotations import axes
from tensor_annotations.axes import Batch
from modeopt.policies import GaussianPolicy

StateDim = typing.NewType("StateDim", axes.Axis)
ControlDim = typing.NewType("ControlDim", axes.Axis)


def rollout_controller_in_dynamics(
    controller,
    start_state: ttf.Tensor2[Batch, StateDim],
    start_state_var: ttf.Tensor2[Batch, StateDim] = None,
):
    rollout_policy_in_dynamics(
        policy=controller.policy,
        dynamics=controller.dynamics,
        start_state=start_state,
        start_state_var=start_state_var,
    )


def rollout_policy_in_dynamics(
    policy: GaussianPolicy,
    dynamics: Callable,
    start_state: ttf.Tensor2[Batch, StateDim],
    start_state_var: ttf.Tensor2[Batch, StateDim] = None,
):
    control_means, control_vars = policy()
    return rollout_controls_in_dynamics(
        dynamics=dynamics,
        start_state=start_state,
        control_means=control_means,
        start_state_var=start_state_var,
        control_vars=control_vars,
    )


def rollout_controls_in_dynamics(
    dynamics: Callable,
    start_state: ttf.Tensor2[Batch, StateDim],
    control_means: ttf.Tensor2[Batch, ControlDim],
    start_state_var: ttf.Tensor2[Batch, StateDim] = None,
    control_vars: ttf.Tensor2[Batch, ControlDim] = None,
):
    """Rollout a given set of control means and vars

    :returns: (states_means, state_vars)
    """
    print("here")
    print(start_state.shape)
    num_time_steps = control_means.shape[0]
    if start_state_var is None:
        start_state_var = tf.zeros((1, start_state.shape[1]), dtype=default_float())

    state_means = start_state
    state_vars = start_state_var
    control_var = None
    for t in range(num_time_steps):
        control_mean = control_means[t : t + 1, :]
        if control_vars is not None:
            control_var = control_vars[t : t + 1, :]
        next_state_mean, next_state_var = dynamics(
            state_means[-1:, :], control_mean, state_vars[-1:, :], control_var
        )
        print("next_state_mean")
        print(next_state_mean.shape)
        print(next_state_var.shape)
        state_means = tf.concat([state_means, next_state_mean], 0)
        state_vars = tf.concat([state_vars, next_state_var], 0)
    return state_means, state_vars


def rollout_controller_in_env(
    env, controller, start_state: ttf.Tensor2[Batch, StateDim] = None
):
    """Rollout a controller in an environment"""
    return rollout_policy_in_env(env, controller.policy, start_state)


def rollout_policy_in_env(
    env, policy, start_state: ttf.Tensor2[Batch, StateDim] = None
):
    """Rollout a given policy on an environment

    :param policy: Callable representing policy to rollout
    :param timesteps: number of timesteps to rollout
    :returns: (states, delta_states)
    """
    env.state_init = start_state.numpy()
    time_step = env.reset()

    states = time_step.observation
    for t in range(policy.num_time_steps):
        control, _ = policy(t)
        next_time_step = env.step(control.numpy())
        states = np.concatenate([states, next_time_step.observation])
    return np.stack(states)
