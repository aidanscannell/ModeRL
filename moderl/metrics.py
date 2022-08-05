#!/usr/bin/env python3
import tensorflow as tf
from geoflow.manifolds import GPManifold

from modeopt.cost_functions import RiemannianEnergyCostFunction
from modeopt.mode_opt import ModeOpt
from modeopt.rollouts import rollout_controller_in_dynamics
from modeopt.utils import (
    append_zero_control,
    weight_to_matrix,
    combine_state_controls_to_input,
)


def mode_probability(
    mode_optimiser: ModeOpt,
    marginalise_state: bool = True,
    marginalise_gating_func: bool = True,
    sum: bool = False,
):
    state_means, state_vars = rollout_controller_in_dynamics(
        dynamics=mode_optimiser.dynamics,
        controller=mode_optimiser.mode_controller,
        start_state=mode_optimiser.start_state,
        variance=False,
    )
    control_means, control_vars = append_zero_control(
        *mode_optimiser.mode_controller(variance=True)
    )
    input_mean, input_var = combine_state_controls_to_input(
        state_mean=state_means,
        control_mean=control_means,
        state_var=state_vars,
        # control_var=control_vars,
    )
    if marginalise_state:
        h_mean, h_var = mode_optimiser.dynamics.uncertain_predict_gating(
            state_mean=state_means,
            control_mean=control_means,
            state_var=state_vars,
            # control_var=control_vars,
        )
        print("h_mean.shape 1")
        print(h_mean.shape)
    else:
        h_mean, h_var = mode_optimiser.dynamics.mosvgpe.gating_network.predict_h(
            input_mean
        )
        print("h_mean.shape 2")
        print(h_mean.shape)
    if not marginalise_gating_func:
        h_var = None

    mode_probs = (
        mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs_given_h(
            h_mean, h_var=h_var
        )
    )[:, mode_optimiser.desired_mode]
    print("mode_probs")
    print(mode_probs.shape)
    if sum:
        return tf.reduce_sum(mode_probs)
    else:
        return mode_probs


# def mode_probability(
#     mode_optimiser: ModeOpt,
#     marginalise_state: bool = True,
#     marginalise_gating_func: bool = True,
#     sum: bool = False,
# ):
#     state_means, state_vars = rollout_controller_in_dynamics(
#         dynamics=mode_optimiser.dynamics,
#         controller=mode_optimiser.mode_controller,
#         start_state=mode_optimiser.start_state,
#     )
#     control_means, control_vars = append_zero_control(
#         *mode_optimiser.mode_controller(variance=True)
#     )
#     input_mean, input_var = combine_state_controls_to_input(
#         state_mean=state_means,
#         control_mean=control_means,
#         state_var=state_vars,
#         control_var=control_vars,
#     )
#     if marginalise_gating_func:
#         if marginalise_state:
#             mode_probs = mode_optimiser.dynamics.predict_mode_probability(
#                 state_mean=state_means,
#                 control_mean=control_means,
#                 state_var=state_vars,
#                 control_var=control_vars,
#             )
#         else:
#             mode_probs = (
#                 mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs(
#                     input_mean
#                 )[:, mode_optimiser.desired_mode]
#             )
#     else:
#         h_mean, _ = mode_optimiser.dynamics.mosvgpe.gating_network.predict_h(
#             input_mean
#         )[:, mode_optimiser.desired_mode]

#     # def predict_mixing_probs_given_h(
#         mode_probs = (
#             mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs_given_h(
#                 h_mean, h_var=None
#             )[:, mode_optimiser.desired_mode]
#         )
#     if sum:
#         return tf.reduce_sum(mode_probs)
#     else:
#         return mode_probs


def gating_function_variance(
    mode_optimiser: ModeOpt, marginalise_state: bool = True, sum: bool = False
):
    state_means, state_vars = rollout_controller_in_dynamics(
        dynamics=mode_optimiser.dynamics,
        controller=mode_optimiser.mode_controller,
        start_state=mode_optimiser.start_state,
        variance=False,
    )
    control_means, control_vars = append_zero_control(
        *mode_optimiser.mode_controller(variance=True)
    )
    if marginalise_state:
        _, h_vars = mode_optimiser.dynamics.uncertain_predict_gating(
            state_mean=state_means,
            control_mean=control_means,
            state_var=state_vars,
            # control_var=control_vars,
        )
    else:
        input_mean, input_var = combine_state_controls_to_input(
            state_mean=state_means,
            control_mean=control_means,
            state_var=state_vars,
            # control_var=control_vars,
        )
        _, h_vars = mode_optimiser.dynamics.mosvgpe.gating_network.predict_h(input_mean)
    h_vars = h_vars[:, mode_optimiser.desired_mode]
    if sum:
        return tf.reduce_sum(h_vars)
    else:
        return h_vars


def state_variance(mode_optimiser: ModeOpt, sum: bool = False):
    state_means, state_vars = rollout_controller_in_dynamics(
        dynamics=mode_optimiser.dynamics,
        controller=mode_optimiser.mode_controller,
        start_state=mode_optimiser.start_state,
        variance=False,
    )
    if sum:
        return tf.reduce_sum(state_vars)
    else:
        return state_vars


def approximate_riemannian_energy(
    mode_optimiser: ModeOpt, covariance_weight: float = 1.0, sum: bool = False
):
    riemannian_metric_weight_matrix = weight_to_matrix(
        1.0, dim=mode_optimiser.dynamics.state_dim
    )
    state_means, state_vars = rollout_controller_in_dynamics(
        dynamics=mode_optimiser.dynamics,
        controller=mode_optimiser.mode_controller,
        start_state=mode_optimiser.start_state,
        variance=False,
    )
    control_means, control_vars = mode_optimiser.mode_controller(variance=True)

    cost_fn = RiemannianEnergyCostFunction(
        gp=mode_optimiser.dynamics.desired_mode_gating_gp,
        riemannian_metric_weight_matrix=riemannian_metric_weight_matrix,
        covariance_weight=covariance_weight,
    )
    energy = cost_fn(
        state=state_means,
        control=control_means,
        state_var=state_vars,
        # control_var=control_vars,
    )
    if sum:
        return tf.reduce_sum(energy)
    else:
        return energy


def riemannian_length(mode_optimiser: ModeOpt, covariance_weight: float = 1.0):
    state_means, state_vars = rollout_controller_in_dynamics(
        dynamics=mode_optimiser.dynamics,
        controller=mode_optimiser.mode_controller,
        start_state=mode_optimiser.start_state,
    )
    control_means, control_vars = mode_optimiser.mode_controller(variance=True)

    manifold = GPManifold(
        gp=mode_optimiser.dynamics.gating_gp, covariance_weight=covariance_weight
    )
    manifold.length()

    state_means, state_vars = rollout_controller_in_dynamics(
        mode_optimiser.mode_controller,
        dynamics=mode_optimiser.dynamics,
        start_state=mode_optimiser.start_state,
        variance=False,
    )
    control_means, control_vars = mode_optimiser.mode_controller()
    h_means, h_vars = mode_optimiser.dynamics.uncertain_predict_gating(
        state_mean=state_means,
        control_mean=control_means,
        state_var=state_vars,
        # control_var=control_vars,
    )
    if sum:
        return tf.reduce_sum(h_vars)
    else:
        return h_vars


def euclidean_length(mode_optimiser: ModeOpt):
    state_means, state_vars = rollout_controller_in_dynamics(
        dynamics=mode_optimiser.dynamics,
        controller=mode_optimiser.mode_controller,
        start_state=mode_optimiser.start_state,
        variance=False,
    )
    control_means, control_vars = mode_optimiser.mode_controller(variance=True)
    h_means, h_vars = mode_optimiser.dynamics.uncertain_predict_gating(
        state_mean=state_means,
        control_mean=control_means,
        state_var=state_vars,
        # control_var=control_vars,
    )
    if sum:
        return tf.reduce_sum(h_vars)
    else:
        return h_vars
