#!/usr/bin/env python3
import typing
from typing import Callable

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import default_float
from scipy.optimize import NonlinearConstraint
from tensor_annotations import axes
from tensor_annotations.axes import Batch

from modeopt.dynamics import ModeOptDynamics
from modeopt.rollouts import rollout_controls_in_dynamics

StateDim = typing.NewType("StateDim", axes.Axis)
ControlDim = typing.NewType("ControlDim", axes.Axis)


def build_mode_chance_constraints_fn(
    mode_opt_dynamics: ModeOptDynamics,
    start_state: ttf.Tensor2[Batch, StateDim],
    horizon: int = 10,
    compile: bool = True,
) -> Callable:
    control_dim = mode_opt_dynamics.control_dim

    def mode_chance_constraints(controls_flat: tf.Tensor) -> tf.Tensor:
        if controls_flat.shape[0] == horizon * control_dim:
            controls = tf.reshape(controls_flat, [-1, control_dim])
            control_means = controls[:, 0:2]
            control_vars = tf.zeros(control_means.shape, dtype=default_float())
        elif controls_flat.shape[0] == horizon * control_dim * 2:
            controls = tf.reshape(controls_flat, [-1, control_dim * 2])
            control_means = controls[:, 0:control_dim]
            control_vars = controls[:, control_dim : control_dim * 2]
        else:
            raise NotImplementedError("Wrong shape for controls")
        state_means, state_vars = rollout_controls_in_dynamics(
            dynamics=mode_opt_dynamics,
            start_state=start_state,
            control_means=control_means,
            control_vars=control_vars,
        )
        mode_prob = mode_opt_dynamics.predict_mode_probability(
            state_means[:-1, :],
            control_means,
            state_vars[:-1, :],
            control_vars
            # state_means[1:, :], control_means, state_vars[1:, :], control_vars
        )
        mode_prob_flat = tf.reshape(mode_prob, [-1])
        tf.print("mode_prob_flat")
        tf.print(mode_prob_flat)
        if controls_flat.shape[0] == horizon * control_dim * 2:
            var_probs = tf.ones(mode_prob_flat.shape, dtype=default_float())
            return tf.concat([mode_prob_flat, var_probs], axis=0)
        # return tf.reshape(mode_prob_flat, [-1])
        return mode_prob_flat

    if compile:
        return tf.function(mode_chance_constraints)
    else:
        return mode_chance_constraints


def build_mode_chance_constraints_scipy(
    mode_opt_dynamics: ModeOptDynamics,
    start_state: ttf.Tensor2[Batch, StateDim],
    horizon: int = 10,
    lower_bound: float = 0.5,
    upper_bound: float = 1.0,
    compile: bool = True,
) -> NonlinearConstraint:
    constraints_fn = build_mode_chance_constraints_fn(
        mode_opt_dynamics, start_state=start_state, horizon=horizon, compile=compile
    )
    return NonlinearConstraint(constraints_fn, lower_bound, upper_bound)
