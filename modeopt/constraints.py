#!/usr/bin/env python3
import tensorflow as tf
from typing import Callable
from gpflow import default_float
from vimpc.rollouts import rollout_controls_in_dynamics
from scipy.optimize import NonlinearConstraint


def build_mode_chance_constraints_fn(
    controller, start_state, compile: bool = True
) -> Callable:
    num_timesteps = controller.num_time_steps
    control_dim = controller.control_dim

    def mode_chance_constraints(controls_flat: tf.Tensor) -> tf.Tensor:
        if controls_flat.shape[0] == num_timesteps * control_dim:
            controls = tf.reshape(controls_flat, [-1, control_dim])
            control_means = controls[:, 0:2]
            control_vars = tf.zeros(control_means.shape, dtype=default_float())
        elif controls_flat.shape[0] == num_timesteps * control_dim * 2:
            controls = tf.reshape(controls_flat, [-1, control_dim * 2])
            control_means = controls[:, 0:control_dim]
            control_vars = controls[:, control_dim : control_dim * 2]
        else:
            raise NotImplementedError("Wrong shape for controls")
        state_means, state_vars = rollout_controls_in_dynamics(
            dynamics=controller.dynamics,
            start_state=start_state,
            control_means=control_means,
            control_vars=control_vars,
        )
        mode_prob = controller.mode_probability(
            state_means[1:, :], control_means, state_vars[1:, :], control_vars
        )
        mode_prob_flat = tf.reshape(mode_prob, [-1])
        if controls_flat.shape[0] == num_timesteps * control_dim * 2:
            var_probs = tf.ones(mode_prob_flat.shape, dtype=default_float())
            return tf.concat([mode_prob_flat, var_probs], axis=0)
        return tf.reshape(mode_prob_flat, [-1])

    if compile:
        return tf.function(mode_chance_constraints)
    else:
        return mode_chance_constraints


def build_mode_chance_constraints_scipy(
    controller, start_state, lower_bound=0.5, upper_bound=1.0, compile: bool = True
) -> Callable:
    constraints_fn = build_mode_chance_constraints_fn(
        controller, start_state=start_state, compile=compile
    )
    return NonlinearConstraint(constraints_fn, lower_bound, upper_bound)
