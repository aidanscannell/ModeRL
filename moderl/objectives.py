#!/usr/bin/env python3
import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from moderl.custom_types import ControlTrajectory, State
from moderl.dynamics import ModeRLDynamics
from moderl.rollouts import rollout_ControlTrajectory_in_ModeRLDynamics
from moderl.utils import combine_state_controls_to_input

tfd = tfp.distributions


def joint_gating_function_entropy(
    dynamics: ModeRLDynamics, initial_solution: ControlTrajectory, start_state: State
) -> ttf.Tensor0:
    """Calculates joint gating function entropy along mean of state trajectory"""
    control_dists = initial_solution()
    state_dists = rollout_ControlTrajectory_in_ModeRLDynamics(
        dynamics=dynamics, control_trajectory=initial_solution, start_state=start_state
    )
    input_dists = combine_state_controls_to_input(
        state=state_dists[1:], control=control_dists
    )
    h_means, h_vars = dynamics.mosvgpe.gating_network.gp.predict_f(
        input_dists.mean(), full_cov=True
    )
    h_vars = (
        h_vars + tf.eye(h_vars.shape[1], h_vars.shape[2], dtype=default_float()) * 1e-6
    )
    h_dist = tfd.MultivariateNormalFullCovariance(h_means, h_vars[0, :, :] ** 2)
    gating_entropy = h_dist.entropy()
    return tf.reduce_sum(gating_entropy)


def independent_gating_function_entropy(
    dynamics: ModeRLDynamics, initial_solution: ControlTrajectory, start_state: State
) -> ttf.Tensor0:
    """Calculates independent gating function entropy along mean of state trajectory"""
    control_dists = initial_solution()
    state_dists = rollout_ControlTrajectory_in_ModeRLDynamics(
        dynamics=dynamics, control_trajectory=initial_solution, start_state=start_state
    )
    input_dists = combine_state_controls_to_input(
        state=state_dists[1:], control=control_dists
    )
    h_means, h_vars = dynamics.mosvgpe.gating_network.gp.predict_f(
        input_dists.mean(), full_cov=True
    )
    h_vars = (
        h_vars + tf.eye(h_vars.shape[1], h_vars.shape[2], dtype=default_float()) * 1e-6
    )
    h_dist = tfd.Normal(h_means, h_vars[0, :, :] ** 2)
    gating_entropy = h_dist.entropy()
    return tf.reduce_sum(gating_entropy)


# def explorative_objective(
#     dynamics: ModeRLDynamics, initial_solution: ControlTrajectory, start_state: State
# ) -> ttf.Tensor0:
#     control_dists = initial_solution()
#     state_dists = rollout_ControlTrajectory_in_ModeRLDynamics(
#         dynamics=dynamics, control_trajectory=initial_solution, start_state=start_state
#     )
#     input_dists = combine_state_controls_to_input(
#         state=state_dists[1:], control=control_dists
#     )
#     h_means, h_vars = dynamics.mosvgpe.gating_network.gp.predict_f(
#         input_dists.mean(), full_cov=True
#     )

#     control_means = initial_solution(variance=False)
#     control_vars = None
#     state_means, state_vars = rollout_controls_in_dynamics(
#         dynamics=mode_optimiser.dynamics,
#         start_state=start_state,
#         control_means=control_means,
#         control_vars=control_vars,
#     )
#     h_means, h_vars = conditional_gating(
#         state_means[1:, :], control_means, state_vars[1:, :], control_vars
#     )
#     print("h_vars")
#     print(h_vars)
#     #     h_means, h_vars = conditional_gating_temporal(state_means[1:, :], control_means, state_vars[1:, :], control_vars)
#     mode_probs = (
#         mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs_given_h(
#             h_means, h_vars
#         )
#     )
#     print("mode_probs.shape")
#     print(mode_probs.shape)
#     gating_entropy = tfd.Bernoulli(mode_probs[:, mode_optimiser.desired_mode]).entropy()
#     #     gating_entropy = tfd.Categorical(mode_probs).entropy()
#     print(gating_entropy.shape)
#     #     h_dist = tfd.MultivariateNormalDiag(h_means, h_vars)
#     #     gating_entropy = h_dist.entropy()
#     tf.print("entropy yo")
#     tf.print(-tf.reduce_sum(gating_entropy))
#     #     tf.print(cost_fn(state_means, control_means, state_vars, control_vars))
#     return -tf.reduce_sum(gating_entropy)


# def bald_objective(initial_solution: ControlTrajectory) -> ttf.Tensor0:
#     def binary_entropy(probs):
#         return -probs * tf.math.log(probs) - (1 - probs) * tf.math.log(1 - probs)

#     def entropy_approx(h_means, h_vars, mode_probs):
#         C = tf.constant(np.sqrt(math.pi * np.log(2.0) / 2.0), dtype=default_float())
#         param_entropy = C * tf.exp(-(h_means**2) / (2 * (h_vars**2 + C**2)))
#         param_entropy = param_entropy / (tf.sqrt(h_vars**2 + C**2))
#         print("param_entropy")
#         print(param_entropy)
#         model_entropy = binary_entropy(mode_probs)
#         print(model_entropy)
#         return model_entropy - param_entropy

#     control_dists = initial_solution()
#     state_dists = rollout_ControlTrajectory_in_ModeRLDynamics(
#         dynamics=dynamics, control_trajectory=initial_solution, start_state=start_state
#     )
#     # control_means = initial_solution(variance=False)
#     # control_vars = None
#     # state_means, state_vars = rollout_controls_in_dynamics(
#     #     dynamics=mode_optimiser.dynamics,
#     #     start_state=start_state,
#     #     control_means=control_means,
#     #     control_vars=control_vars,
#     # )

#     #     h_means, h_vars = mode_optimiser.dynamics.uncertain_predict_gating(state_means[1:, :], control_means)
#     h_means, h_vars = conditional_gating_temporal(
#         state_means[1:, :], control_means, state_vars[1:, :], control_vars
#     )
#     mode_probs = dynamics.mosvgpe.gating_network.predict_mixing_probs_given_h(
#         h_means, h_vars
#     )
#     print("mode_probs.shape")
#     print(mode_probs.shape)
#     print(h_means.shape)
#     print(h_vars.shape)
#     bald_objective = entropy_approx(
#         h_means[:, mode_optimiser.desired_mode],
#         h_vars[:, mode_optimiser.desired_mode],
#         mode_probs[:, mode_optimiser.desired_mode],
#     )
#     print(bald_objective)
#     tf.print("entropy")
#     tf.print(-tf.reduce_sum(bald_objective))
#     tf.print("cost")
#     tf.print(cost_fn(state_means, control_means, state_vars, control_vars))
#     return -tf.reduce_sum(bald_objective)
