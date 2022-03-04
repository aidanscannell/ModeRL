#!/usr/bin/env python3
from typing import Callable

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from modeopt.custom_types import Horizon, TwoStateDim, Times, StateDim

# def build_geodesic_collocation_constraints_closure(
#     gp: GPModel,
#     initial_solution: GeodesicTrajectory,
#     covariance_weight: float = 1.0,
# ) -> Callable[[], ttf.Tensor0]:
#     manifold = GPManifold(gp=gp, covariance_weight=covariance_weight)

#     def constraints_fn():
#         return hermite_simpson_collocation_constraints_fn(
#             initial_solution=initial_solution, ode_fn=manifold.geodesic_ode
#         )

#     return constraints_fn


def hermite_simpson_collocation_constraints_fn(
    state_at_knots: ttf.Tensor2[Horizon, TwoStateDim],
    times: Times,
    ode_fn: Callable[
        [ttf.Tensor2[Horizon, StateDim], ttf.Tensor2[Horizon, StateDim]],
        ttf.Tensor2[Horizon, TwoStateDim],
    ],
):
    """Return Hermite-Simpson collocation constraints

    for state_prime = ode_fn(state_guesses)

    The continuos dynamics are transcribed to a set of collocation equations when

    Simpson quadrature approximates the integrand of the integral as a piecewise quadratic function

    The state trajectory is a cubic Hermite spline, which has a continuous first derivative.

    The collocation constraints returned are in compressed form.

    The Simpson collocation defect is given by,
        0 = x_{i+1} - \hat{x}_{i}
          = x_{i+1} - (x_{i} + h_{i} f_{i})
          = x_{i+1} - x_{i} - h_{i+1}/6 (f_{i} + 4*\hat{f}_{i} + f_{i+1})
    where h_{i} f_{i} is the interpolant.
    The Hermite interpolant is given by,
        0.5 (x_{i} + x_{i+1}) + h_{i+1}/8 (f_{j} - f_{j+1})

    At each knot point the state and its derivative should equal those from the system dynamics (ode_fn)

    :param state_at_knots: states at knot points [num_states, state_dim]
    :param times: [num_states,]
    :param ode_fn:
    :returns: collocation defects [num_states-1, state_dim]
    """

    times_before = times[0:-1]
    times_after = times[1:]
    delta_times = times_after - times_before
    delta_times = tf.reshape(delta_times, [-1, 1])

    delta_times = times[-1] - times[0]
    half_state_dim = int(state_at_knots.shape[-1] / 2)

    def ode_fn_combined(state):
        pos = state[:, :half_state_dim]
        vel = state[:, half_state_dim:]
        return ode_fn(pos=pos, vel=vel)

    # State derivative according to the true continuous dynamics
    state_prime_at_knots = ode_fn_combined(state_at_knots)

    # Approx states at mid points using Hermite interpolation
    state_at_knots_before = state_at_knots[0:-1, :]
    state_at_knots_after = state_at_knots[1:, :]
    state_prime_at_knots_before = state_prime_at_knots[0:-1, :]
    state_prime_at_knots_after = state_prime_at_knots[1:, :]
    state_at_mid_points = states_at_mid_points_hermite_interpolation(
        state_at_knots,
        state_prime_at_knots,
        delta_times=delta_times,
    )

    # Calculate state derivatives at mid points
    state_prime_at_mid_points = ode_fn_combined(state_at_mid_points)
    # state_prime_at_mid_points = ode_fn(times_at_mid_points, state_at_mid_points)

    # Calculate collocation defects
    defects = (state_at_knots_after - state_at_knots_before) - delta_times / 6 * (
        state_prime_at_knots_before
        + 4 * state_prime_at_mid_points
        + state_prime_at_knots_after
    )
    # defects = (-state_at_knots_after + state_at_knots_before) + delta_times / 6 * (
    #     state_prime_at_knots_before
    #     + 4 * state_prime_at_mid_points
    #     + state_prime_at_knots_after
    # )
    return tf.reshape(defects, [-1])
    # return defects.flatten()


# def hermite_simpson_collocation_constraints_fn_initial_solution(
#     initial_solution: FlatOutputTrajectory, ode_fn
# ):
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
#     state_at_knots = tf.concat(
#         [initial_solution.flat_output, initial_solution.flat_output_first_derivative],
#         -1,
#     )

#     times_before = initial_solution.times[0:-1]
#     times_after = initial_solution.times[1:]
#     delta_times = times_after - times_before
#     delta_times = tf.reshape(delta_times, [-1, 1])

#     delta_times = initial_solution.times[-1] - initial_solution.times[0]
#     print("delta_times")
#     print(delta_times)

#     def ode_fn_combined(state):
#         flat_dim = initial_solution.flat_dim
#         pos = state[:, :flat_dim]
#         vel = state[:, flat_dim:]
#         return ode_fn(pos=pos, vel=vel)

#     # State derivative according to the true continuous dynamics
#     state_prime_at_knots = ode_fn_combined(state_at_knots)

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
#     tf.print("Collcation Defects")
#     tf.print(defects)
#     return tf.reshape(defects, [-1])
#     # return defects.flatten()


def states_at_mid_points_hermite_interpolation(
    state_at_knots, state_prime_at_knots, delta_times
):
    """Approximate states at mid points using Hermite interpolation

    The Hermite interpolant is given by,
        x_{k+1/2} = 0.5 (x_{k} + x_{k+1}) + h_{k}/8 (f_{k} - f_{k+1})

    :param state_at_knots: states at knot points [num_states, state_dim]
    :param state_prime_at_knots: states derivatives at knot points [num_states, state_dim]
    :param delta_times: delta time between knot points [num_states-1, 1] e.g. t_{k+1} - t_{k}
    :returns: interpolated states at mid points [num_states-1, state_dim]
    """
    # print("delta_times")
    # print(delta_times.shape)
    # print(delta_times)
    state_at_knots_before = state_at_knots[0:-1, :]
    state_at_knots_after = state_at_knots[1:, :]
    state_prime_at_knots_before = state_prime_at_knots[0:-1, :]
    state_prime_at_knots_after = state_prime_at_knots[1:, :]
    state_at_mid_points = 0.5 * (
        state_at_knots_before + state_at_knots_after
    ) + delta_times / 8 * (state_prime_at_knots_before - state_prime_at_knots_after)
    return state_at_mid_points
