#!/usr/bin/env python3
import numpy as np
import math
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float, default_jitter
from moderl.custom_types import ControlTrajectory, State
from moderl.dynamics import ModeRLDynamics
from moderl.dynamics.conditionals import svgp_covariance_conditional
from gpflow.conditionals import base_conditional
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


def bald_objective(
    dynamics: ModeRLDynamics, initial_solution: ControlTrajectory, start_state: State
) -> ttf.Tensor0:
    def binary_entropy(probs):
        return -probs * tf.math.log(probs) - (1 - probs) * tf.math.log(1 - probs)

    def bald_objective_sampling(input_dists):
        mode_probs = dynamics.mosvgpe.gating_network.predict_mixing_probs(
            Xnew=input_dists.mean()
        )
        print("yo mode probs")
        print(mode_probs)
        model_entropy = binary_entropy(mode_probs[:, dynamics.desired_mode])
        print(model_entropy)

        h_samples = dynamics.mosvgpe.gating_network.gp.predict_f_samples(
            input_dists.mean(),
            num_samples=10,
            full_cov=False
            # input_dists.mean(), num_samples=10, full_cov=True
        )
        print("h_samples")
        print(h_samples.shape)
        mode_probs_samples = (
            dynamics.mosvgpe.gating_network.gp.likelihood.conditional_mean(
                X=input_dists.mean(), F=h_samples
            )
        )
        print("mode_probs_samples.shape")
        print(mode_probs_samples.shape)
        param_entropy = binary_entropy(mode_probs_samples)
        print("param_entropy")
        print(param_entropy)
        param_entropy = tf.reduce_sum(param_entropy)
        print(param_entropy)
        # return model_entropy - param_entropy
        return -model_entropy + param_entropy

    control_dists = initial_solution()
    state_dists = rollout_ControlTrajectory_in_ModeRLDynamics(
        dynamics=dynamics, control_trajectory=initial_solution, start_state=start_state
    )
    input_dists = combine_state_controls_to_input(
        state=state_dists[1:], control=control_dists
    )

    def bald_objective_closed_form():
        h_means, h_vars = dynamics.mosvgpe.gating_network.gp.predict_f(
            input_dists.mean(),
            full_cov=False
            # input_dists.mean(), full_cov=True
        )

        input_means = input_dists.mean()

        def entropy_approx(h_means, h_vars, mode_probs):
            C = tf.constant(np.sqrt(math.pi * np.log(2.0) / 2.0), dtype=default_float())
            param_entropy = C * tf.exp(-(h_means**2) / (2 * (h_vars**2 + C**2)))
            param_entropy = param_entropy / (tf.sqrt(h_vars**2 + C**2))
            print("param_entropy")
            print(param_entropy)
            model_entropy = binary_entropy(mode_probs)
            print(model_entropy)
            return model_entropy - param_entropy

        print("h_means.shape")
        print(h_means.shape)
        print(h_vars.shape)
        h_means_conditioned, h_vars_conditioned = h_means[0:1, :], h_vars[0:1, :]
        print("h_means_conditioned.shape")
        print(h_means_conditioned.shape)
        print(h_vars_conditioned.shape)
        for t in range(1, initial_solution.horizon):
            Xnew = input_means[t : t + 1, :]
            print("Xnew.shape")
            print(Xnew.shape)
            Xobs = tf.concat([input_means[:t, :], input_means[t + 1 :, :]], 0)
            print(Xobs.shape)
            f = tf.concat([h_means[:t, :], h_means[t + 1 :, :]], 0)
            print("f.shape")
            print(f.shape)

            Knn = svgp_covariance_conditional(
                X1=Xnew, X2=Xnew, svgp=dynamics.desired_mode_gating_gp
            )[0, 0, :]
            Kmm = svgp_covariance_conditional(
                X1=Xobs, X2=Xobs, svgp=dynamics.desired_mode_gating_gp
            )[0, :, :]
            Kmn = svgp_covariance_conditional(
                X1=Xobs, X2=Xnew, svgp=dynamics.desired_mode_gating_gp
            )[0, :, :]
            Kmm += tf.eye(Kmm.shape[0], dtype=default_float()) * default_jitter()
            h_mean, h_var = base_conditional(
                Kmn=Kmn,
                Kmm=Kmm,
                Knn=Knn,
                f=f,
                full_cov=False,
                q_sqrt=None,  # TODO make this h_var??
                white=False,
            )
            print("hmean.shape")
            print(h_mean.shape)
            print(h_var.shape)
            # h_mean = tf.expand_dims(h_mean, 0)
            # h_var = tf.expand_dims(h_var, 0)
            h_means_conditioned = tf.concat([h_means_conditioned, h_mean], 0)
            h_vars_conditioned = tf.concat([h_vars_conditioned, h_var], 0)
        # h_means = h_means_conditioned
        # h_vars = h_vars_conditioned

        # h_vars = (
        #     h_vars + tf.eye(h_vars.shape[1], h_vars.shape[2], dtype=default_float()) * 1e-6
        # )
        # h_dist = tfd.Normal(h_means, h_vars[0, :, :] ** 2)
        # gating_entropy = h_dist.entropy()

        # #     h_means, h_vars = mode_optimiser.dynamics.uncertain_predict_gating(state_means[1:, :], control_means)
        # h_means, h_vars = conditional_gating_temporal(
        #     state_dists=state_dists[1:], control_dists=control_dists
        # )
        # #     state_means[1:, :], control_means, state_vars[1:, :], control_vars
        # # )
        (
            mode_probs,
            _,
        ) = dynamics.mosvgpe.gating_network.gp.likelihood.predict_mean_and_var(
            X=input_dists.mean(), Fmu=h_means, Fvar=h_vars
        )
        print("mode_probs.shape")
        print(mode_probs.shape)
        print(h_means.shape)
        print(h_vars.shape)
        return entropy_approx(
            h_means_conditioned[:, dynamics.desired_mode],
            h_vars_conditioned[:, dynamics.desired_mode],
            mode_probs[:, dynamics.desired_mode],
        )

    bald_objective = bald_objective_closed_form()
    # bald_objective = bald_objective_sampling(input_dists)
    print(bald_objective)
    tf.print("entropy")
    tf.print(tf.reduce_sum(bald_objective))
    return tf.reduce_sum(bald_objective)


# def conditional_gating(state_means, control_means, state_vars, control_vars):
#     # h_means, h_vars = dynamics.mosvgpe.gating_network.gp.predict_f(
#     #     input_dists.mean(), full_cov=True
#     # )
#     # h_means_prior, h_vars_prior = mode_optimiser.dynamics.uncertain_predict_gating(
#     #     state_means, control_means
#     # )
#     # gating_gp = mode_optimiser.dynamics.desired_mode_gating_gp

#     # input_means, input_vars = combine_state_controls_to_input(
#     #     state_means, control_means, state_vars, control_vars
#     # )

#     input_means = input_dists.mean()
#     # h_means, h_vars = h_means_prior[0:1, :], h_vars_prior[0:1, :]
#     for t in range(1, horizon):
#         Xnew = input_means[t : t + 1, :]
#         Xobs = tf.concat([input_means[:t, :], input_means[t + 1 :, :]], 0)
#         f = tf.concat([h_means[:t, :], h_means[t + 1 :, :]], 0)

#         Knn = svgp_covariance_conditional(X1=Xnew, X2=Xnew, svgp=gating_gp)[0, 0, :]
#         Kmm = svgp_covariance_conditional(X1=Xobs, X2=Xobs, svgp=gating_gp)[0, :, :]
#         Kmn = svgp_covariance_conditional(X1=Xobs, X2=Xnew, svgp=gating_gp)[0, :, :]
#         Kmm += tf.eye(Kmm.shape[0], dtype=default_float()) * default_jitter()
#         h_mean, h_var = base_conditional(
#             Kmn=Kmn,
#             Kmm=Kmm,
#             Knn=Knn,
#             f=f,
#             full_cov=False,
#             q_sqrt=None,
#             white=False,
#         )
#         h_means_conditioned = tf.concat([h_means_conditioned, h_mean], 0)
#         h_vars_conditioned = tf.concat([h_vars_conditioned, h_var], 0)
#     return h_means_conditioned, h_vars_conditioned


# def conditional_gating_temporal(
#     dynamics: ModelRLDynamics, state_means, control_means, state_vars, control_vars
# ):
#     control_dists = initial_solution()
#     state_dists = rollout_ControlTrajectory_in_ModeRLDynamics(
#         dynamics=dynamics, control_trajectory=initial_solution, start_state=start_state
#     )
#     input_dists = combine_state_controls_to_input(
#         state=state_dists[1:], control=control_dists
#     )
#     h_traj_mean, h_traj_cov = dynamics.mosvgpe.gating_network.gp.predict_f(
#         input_dists.mean(), full_cov=True
#     )

#     h_means_prior, h_vars_prior = dynamics.uncertain_predict_gating(
#         state_means, control_means
#     )
#     gating_gp = dynamics.desired_mode_gating_gp

#     input_means, input_vars = combine_state_controls_to_input(
#         state_means, control_means, state_vars, control_vars
#     )

#     h_means, h_vars = h_means_prior[0:1, :], h_vars_prior[0:1, :]
#     for t in range(1, horizon):
#         Xnew = input_means[t : t + 1, :]
#         Xobs = input_means[0:t, :]
#         f = h_means_prior[0:t, :]

#         Knn = svgp_covariance_conditional(X1=Xnew, X2=Xnew, svgp=gating_gp)[0, 0, :]
#         Kmm = svgp_covariance_conditional(X1=Xobs, X2=Xobs, svgp=gating_gp)[0, :, :]
#         Kmn = svgp_covariance_conditional(X1=Xobs, X2=Xnew, svgp=gating_gp)[0, :, :]
#         Kmm += tf.eye(Kmm.shape[0], dtype=default_float()) * default_jitter()
#         h_mean, h_var = base_conditional(
#             Kmn=Kmn,
#             Kmm=Kmm,
#             Knn=Knn,
#             f=f,
#             full_cov=False,
#             q_sqrt=None,
#             white=False,
#         )
#         h_means = tf.concat([h_means, h_mean], 0)
#         h_vars = tf.concat([h_vars, h_var], 0)
#     return h_means, h_vars
