#!/usr/bin/env python3
from typing import Callable

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from moderl.custom_types import ControlTrajectory, StateDim
from moderl.dynamics import ModeRLDynamics
from moderl.rollouts import rollout_ControlTrajectory_in_ModeRLDynamics
from scipy.optimize import NonlinearConstraint
from tensor_annotations.axes import Batch


tfd = tfp.distributions


def build_mode_chance_constraints_scipy(
    dynamics: ModeRLDynamics,
    control_trajectory: ControlTrajectory,
    start_state: ttf.Tensor2[Batch, StateDim],
    lower_bound: float = 0.5,
    upper_bound: float = 1.0,
    compile: bool = True,
) -> NonlinearConstraint:
    constraints_fn = build_mode_chance_constraints_fn(
        dynamics,
        control_trajectory=control_trajectory,
        start_state=start_state,
        compile=compile,
    )
    return NonlinearConstraint(constraints_fn, lower_bound, upper_bound)


def build_mode_chance_constraints_fn(
    dynamics: ModeRLDynamics,
    control_trajectory: ControlTrajectory,
    start_state: ttf.Tensor2[Batch, StateDim],
    compile: bool = True,
) -> Callable:
    if isinstance(control_trajectory.dist, tfd.Deterministic):

        def controls_flat_to_ControlTrajectory(
            controls_flat: tf.Tensor,
        ) -> ControlTrajectory:
            controls = tf.reshape(controls_flat, [-1, control_trajectory.control_dim])
            return ControlTrajectory(tfd.Deterministic(controls))

    elif isinstance(control_trajectory.dist, tfd.Normal):

        def controls_flat_to_ControlTrajectory(
            controls_flat: tf.Tensor,
        ) -> ControlTrajectory:
            controls = tf.reshape(
                controls_flat, [-1, control_trajectory.control_dim * 2]
            )
            control_vars = controls[
                :, control_trajectory.control_dim : control_trajectory.control_dim * 2
            ]
            return ControlTrajectory(
                tfd.Normal(loc=controls, scale=tf.math.sqrt(control_vars))
            )

    else:
        raise NotImplementedError(
            "ControlTrajectory.dist should be Normal or Deterministic"
        )

    def mode_chance_constraints(controls_flat: tf.Tensor) -> tf.Tensor:
        control_trajectory = controls_flat_to_ControlTrajectory(controls_flat)
        state_dists = rollout_ControlTrajectory_in_ModeRLDynamics(
            dynamics=dynamics,
            control_trajectory=control_trajectory,
            start_state=start_state,
        )
        mode_prob = dynamics.predict_mode_probability(
            state=state_dists[1:, :], control=control_trajectory.dist
        )
        mode_prob_flat = tf.reshape(mode_prob, [-1])
        if isinstance(control_trajectory.dist, tfd.Normal):
            var_probs = tf.ones(mode_prob_flat.shape, dtype=default_float())
            return tf.concat([mode_prob_flat, var_probs], axis=0)
        return mode_prob_flat

    if compile:
        return tf.function(mode_chance_constraints)
    else:
        return mode_chance_constraints
