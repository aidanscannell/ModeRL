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


# def build_geodesic_collocation_constraints_scipy(
#     mode_opt_dynamics: ModeOptDynamics,
#     start_state: ttf.Tensor2[Batch, StateDim],
#     horizon: int = 10,
#     lower_bound: float = 0.5,
#     upper_bound: float = 1.0,
#     compile: bool = True,
# ) -> NonlinearConstraint:

#     manifold = GPManifold(
#         gp=mode_opt_dynamics.gating_gp, covariance_weight=covariance_weight
#     )

#     def constraints_fn(state_at_knots):
#         hermite_simpson_collocation_constraints_fn(
#             state_at_knots=state_at_knots, times=times, ode_fn=manifold.geodesic_ode
#         )

#     constraints_fn = build_mode_chance_constraints_fn(
#         mode_opt_dynamics, start_state=start_state, horizon=horizon, compile=compile
#     )
#     return NonlinearConstraint(constraints_fn, lower_bound, upper_bound)


# def hermite_simpson_collocation_constraints_fn(state_at_knots, times, ode_fn):
#     """Return Hermite-Simpson collocation constraints

#     for state_prime = ode_fn(state_guesses)

#     The continuos dynamics are transcribed to a set of collocation equations when

#     Simpson quadrature approximates the integrand of the integral as a piecewise quadratic function

#     The state trajectory is a cubic Hermite spline, which has a continuous first derivative.

#     The collocation constraints returned are in compressed form.

#     The Simpson collocation defect is given by,
#         0 = x_{i+1} - \hat{x}_{i}
#           = x_{i+1} - (x_{i} + h_{i} f_{i})
#           = x_{i+1} - x_{i} - h_{i+1}/6 (f_{i} + 4*\hat{f}_{i} + f_{i+1})
#     where h_{i} f_{i} is the interpolant.
#     The Hermite interpolant is given by,
#         0.5 (x_{i} + x_{i+1}) + h_{i+1}/8 (f_{j} - f_{j+1})

#     At each knot point the state and its derivative should equal those from the system dynamics (ode_fn)

#     :param state_at_knots: states at knot points [num_states, state_dim]
#     :param times: [num_states,]
#     :param ode_fn:
#     :returns: collocation defects [num_states-1, state_dim]
#     """
#     times_before = times[0:-1]
#     times_after = times[1:]
#     delta_times = times_after - times_before
#     delta_times = delta_times.reshape(-1, 1)

#     delta_times = times[-1] - times[0]
#     print("delta_times")
#     print(delta_times)

#     def ode_fn_combined(state):
#         pos = state[:, :2]
#         vel = state[:, 2:]
#         return ode_fn(pos=pos, vel=vel)

#     # def geodesic_ode(
#     #     self,
#     #     pos: Union[ttf.Tensor1[InputDim], ttf.Tensor2[NumData, InputDim]],
#     #     vel: Union[ttf.Tensor1[InputDim], ttf.Tensor2[NumData, InputDim]],
#     # ) -> Union[ttf.Tensor1[TwoInputDim], ttf.Tensor2[NumData, TwoInputDim]]:

#     # State derivative according to the true continuous dynamics
#     # state_prime_at_knots = jax.vmap(ode_fn_single_state)(state_at_knots)
#     state_prime_at_knots = ode_fn_combined(state_at_knots)
#     # state_prime_at_knots = ode_fn(times, state_at_knots)

#     # Approx states at mid points using Hermite interpolation
#     state_at_knots_before = state_at_knots[0:-1, :]
#     state_at_knots_after = state_at_knots[1:, :]
#     state_prime_at_knots_before = state_prime_at_knots[0:-1, :]
#     state_prime_at_knots_after = state_prime_at_knots[1:, :]
#     state_at_mid_points = states_at_mid_points_hermite_interpolation(
#         state_at_knots,
#         state_prime_at_knots,
#         delta_times=delta_times,
#     )

#     # Calculate state derivatives at mid points
#     # state_prime_at_mid_points = jax.vmap(ode_fn_single_state)(state_at_mid_points)
#     state_prime_at_mid_points = ode_fn_combined(state_at_mid_points)
#     # state_prime_at_mid_points = ode_fn(times_at_mid_points, state_at_mid_points)

#     # Calculate collocation defects
#     defects = (state_at_knots_after - state_at_knots_before) - delta_times / 6 * (
#         state_prime_at_knots_before
#         + 4 * state_prime_at_mid_points
#         + state_prime_at_knots_after
#     )
#     # defects = (-state_at_knots_after + state_at_knots_before) + delta_times / 6 * (
#     #     state_prime_at_knots_before
#     #     + 4 * state_prime_at_mid_points
#     #     + state_prime_at_knots_after
#     # )
#     print("Collcation Defects")
#     print(defects)
#     return defects.flatten()


# def states_at_mid_points_hermite_interpolation(
#     state_at_knots, state_prime_at_knots, delta_times
# ):
#     """Approximate states at mid points using Hermite interpolation

#     The Hermite interpolant is given by,
#         x_{k+1/2} = 0.5 (x_{k} + x_{k+1}) + h_{k}/8 (f_{k} - f_{k+1})

#     :param state_at_knots: states at knot points [num_states, state_dim]
#     :param state_prime_at_knots: states derivatives at knot points [num_states, state_dim]
#     :param delta_times: delta time between knot points [num_states-1, 1] e.g. t_{k+1} - t_{k}
#     :returns: interpolated states at mid points [num_states-1, state_dim]
#     """
#     # print("delta_times")
#     # print(delta_times.shape)
#     # print(delta_times)
#     state_at_knots_before = state_at_knots[0:-1, :]
#     state_at_knots_after = state_at_knots[1:, :]
#     state_prime_at_knots_before = state_prime_at_knots[0:-1, :]
#     state_prime_at_knots_after = state_prime_at_knots[1:, :]
#     state_at_mid_points = 0.5 * (
#         state_at_knots_before + state_at_knots_after
#     ) + delta_times / 8 * (state_prime_at_knots_before - state_prime_at_knots_after)
#     return state_at_mid_points
