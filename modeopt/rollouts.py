#!/usr/bin/env python3
import typing
from typing import Tuple, Union

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import default_float
from tensor_annotations import axes
from tensor_annotations.axes import Batch
from tf_agents.environments import py_environment

from modeopt.dynamics import SVGPDynamicsWrapper, ModeOptDynamics
from modeopt.controllers import NonFeedbackController, FeedbackController

# from modeopt.policies import VariationalPolicy
from modeopt.custom_types import StateDim, ControlDim, Horizon, One

HorizonPlusOne = typing.NewType("HorizonPlusOne", axes.Axis)
Controller = Union[FeedbackController, NonFeedbackController]


def rollout_controller_in_dynamics(
    dynamics: Union[SVGPDynamicsWrapper, ModeOptDynamics],
    controller: Controller,
    start_state: ttf.Tensor2[One, StateDim],
) -> Tuple[
    ttf.Tensor2[HorizonPlusOne, StateDim], ttf.Tensor2[HorizonPlusOne, StateDim]
]:
    """Rollout a given set of control means and vars

    :returns: (states_means, state_vars)
    """
    if isinstance(controller, NonFeedbackController):
        control_means, control_vars = controller(variance=True)
        print("control_means rollout")
        print(control_means)
        print(control_vars)
    else:
        raise NotImplementedError
    means, var = rollout_controls_in_dynamics(
        dynamics=dynamics,
        start_state=start_state,
        control_means=control_means,
        control_vars=control_vars,
    )
    print("state means ROLLOUT")
    print(means)
    print(vars)
    return rollout_controls_in_dynamics(
        dynamics=dynamics,
        start_state=start_state,
        control_means=control_means,
        control_vars=control_vars,
    )


def rollout_controller_in_env(
    env: py_environment.PyEnvironment,
    controller: Controller,
    start_state: ttf.Tensor2[One, StateDim],
) -> Tuple[
    ttf.Tensor2[HorizonPlusOne, StateDim], ttf.Tensor2[HorizonPlusOne, StateDim]
]:
    """Rollout a given set of control means and vars

    :returns: (states_means, state_vars)
    """
    if isinstance(controller, NonFeedbackController):
        return rollout_controls_in_env(
            env=env, start_state=start_state, controls=controller()
        )
    else:
        raise NotImplementedError


def rollout_controls_in_dynamics(
    dynamics: Union[SVGPDynamicsWrapper, ModeOptDynamics],
    start_state: ttf.Tensor2[One, StateDim],
    control_means: ttf.Tensor2[Horizon, ControlDim],
    start_state_var: ttf.Tensor2[One, StateDim] = None,
    control_vars: ttf.Tensor2[Horizon, ControlDim] = None,
) -> Tuple[
    ttf.Tensor2[HorizonPlusOne, StateDim], ttf.Tensor2[HorizonPlusOne, StateDim]
]:
    """Rollout a given set of control means and vars

    :returns: (states_means, state_vars)
    """
    horizon = control_means.shape[0]
    if start_state_var is None:
        start_state_var = tf.zeros((1, start_state.shape[1]), dtype=default_float())

    state_means = start_state
    state_vars = start_state_var
    for t in range(horizon):
        control_mean = control_means[t : t + 1, :]
        if control_vars is not None:
            control_var = control_vars[t : t + 1, :]
        else:
            control_var = None
        next_state_mean, next_state_var = dynamics.forward(
            state_mean=state_means[-1:, :],
            control_mean=control_mean,
            state_var=state_vars[-1:, :],
            control_var=control_var,
            predict_state_difference=False,
        )
        state_means = tf.concat([state_means, next_state_mean], 0)
        state_vars = tf.concat([state_vars, next_state_var], 0)
    return state_means, state_vars


# def rollout_policy_in_dynamics(
#     policy: VariationalPolicy,
#     dynamics: SVGPDynamicsWrapper,
#     start_state: ttf.Tensor2[One, StateDim],
#     start_state_var: ttf.Tensor2[One, StateDim] = None,
# ) -> Tuple[
#     ttf.Tensor2[HorizonPlusOne, StateDim], ttf.Tensor2[HorizonPlusOne, StateDim]
# ]:
#     """Rollout a policy in gp dynamics model

#     :returns: (states_means, state_vars)
#     """
#     raise DeprecationWarning()
#     # if start_state_var is None:
#     #     start_state_var = tf.zeros((1, start_state.shape[1]), dtype=default_float())

#     # state_means = start_state
#     # state_vars = start_state_var
#     # for t in range(policy.num_time_steps):
#     #     control_mean, control_var = policy(t)
#     #     next_state_mean, next_state_var = dynamics(
#     #         state_means[-1:, :], control_mean, state_vars[-1:, :], control_var
#     #     )
#     #     state_means = tf.concat([state_means, next_state_mean], 0)
#     #     state_vars = tf.concat([state_vars, next_state_var], 0)
#     # return state_means, state_vars


def rollout_controls_in_env(
    env: py_environment.PyEnvironment,
    start_state: ttf.Tensor2[Batch, StateDim],
    controls: ttf.Tensor2[Horizon, ControlDim],
) -> ttf.Tensor2[HorizonPlusOne, StateDim]:
    """Rollout a controls in environment

    :returns: states
    """
    env.state_init = start_state
    env.reset()
    states = start_state
    horizon = controls.shape[0]

    for t in range(horizon):
        next_time_step = env.step(controls[t])
        states = tf.concat([states, next_time_step.observation], axis=0)
    return tf.stack(states)


# def rollout_policy_in_env(
#     env: py_environment.PyEnvironment,
#     policy: VariationalPolicy,
#     start_state: ttf.Tensor2[Batch, StateDim] = None,
# ) -> ttf.Tensor2[HorizonPlusOne, StateDim]:
#     """Rollout a policy in environment

#     :param policy: Callable representing policy to rollout
#     :returns: (states, delta_states)
#     """
#     env.state_init = start_state.numpy()
#     env.reset()
#     states = start_state.numpy()

#     for t in range(policy.num_time_steps):
#         control, _ = policy(t)
#         next_time_step = env.step(control.numpy())
#         states = np.concatenate([states, next_time_step.observation])
#     return np.stack(states)


def collect_data_from_env(
    env: py_environment.PyEnvironment,
    trajectory,
    # trajectory: ControlTrajectoryDist,
    start_state: ttf.Tensor2[Batch, StateDim] = None,
) -> ttf.Tensor2[HorizonPlusOne, StateDim]:
    """Rollout a control trajectory in environment

    :returns: (states, delta_states)
    """
    env.state_init = start_state.numpy()
    env.reset()
    state = start_state.numpy()

    delta_state_outputs = []
    for t in range(trajectory.horizon):
        control, _ = trajectory(t)
        state_control_inputs = np.concatenate([state, control], -1)
        next_time_step = env.step(control.numpy())
        delta_state_outputs.append(next_time_step.observation - state)
        state = next_time_step.observation
    return state_control_inputs, np.stack(delta_state_outputs)
