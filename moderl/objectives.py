#!/usr/bin/env python3
import math

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float, default_jitter
from gpflow.conditionals import base_conditional
from moderl.custom_types import ControlTrajectory, State
from moderl.dynamics import ModeRLDynamics
from moderl.dynamics.conditionals import svgp_covariance_conditional
from moderl.rollouts import rollout_ControlTrajectory_in_ModeRLDynamics
from moderl.utils import combine_state_controls_to_input


tfd = tfp.distributions


# def upper_confidence_bound(
#     dynamics: ModeRLDynamics, initial_solution: ControlTrajectory, start_state: State
# ) -> ttf.Tensor0:
#     control_dists = initial_solution()
#     state_dists = rollout_ControlTrajectory_in_ModeRLDynamics(
#       dynamics=dynamics, control_trajectory=initial_solution, start_state=start_state
#     )
#     input_dists = combine_state_controls_to_input(
#         state=state_dists[1:], control=control_dists
#     )
#     h_means, h_vars = dynamics.mosvgpe.gating_network.gp.predict_f(
#         input_dists.mean(), full_cov=True
#     )
#     beta = 1.0
#     h_ucb = h_means + beta * h_vars * initial_solution.hallucinated_controls
#     mode_probs = dynamics.mosvgpe.gating_network.gp.likelihood.conditional_mean(
#         input_dists.mean(), F=h_ucb
#     )
#     alpha = tfd.Bernoulli(probs=mode_probs)
#     h_vars = (
#     h_vars + tf.eye(h_vars.shape[1], h_vars.shape[2], dtype=default_float()) * 1e-6
#     )
#     h_dist = tfd.Normal(h_means, h_vars[0, :, :] ** 2)
#     gating_entropy = h_dist.entropy()
#     return tf.reduce_sum(gating_entropy)


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
        h_vars
        + tf.eye(h_vars.shape[1], h_vars.shape[2], dtype=default_float())
        * default_jitter()
    )
    # TODO make this use cholesky and lower triangular
    h_dist = tfd.MultivariateNormalTriL(
        loc=h_means, scale_tril=tf.linalg.cholesky(h_vars[0, :, :] ** 2)
    )
    # h_dist = tfd.MultivariateNormalFullCovariance(h_means, h_vars[0, :, :] ** 2)
    gating_entropy = h_dist.entropy()
    return tf.reduce_sum(gating_entropy)


def conditional_gating_function_entropy(
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
    input_means = input_dists.mean()

    h_means_conditioned, h_vars_conditioned = [], []
    for t in range(1, initial_solution.horizon):
        state = input_means[t : t + 1, :]
        print("state.shape")
        print(state.shape)
        state_traj = input_means[:t, :]
        f, h_covs = dynamics.mosvgpe.gating_network.gp.predict_f(
            state_traj, full_cov=True
        )
        print(state_traj.shape)
        print("f.shape")
        print(f.shape)
        Knn = svgp_covariance_conditional(
            X1=state, X2=state, svgp=dynamics.desired_mode_gating_gp
        )[0, 0, :]
        Kmm = svgp_covariance_conditional(
            X1=state_traj, X2=state_traj, svgp=dynamics.desired_mode_gating_gp
        )[0, :, :]
        Kmn = svgp_covariance_conditional(
            X1=state_traj, X2=state, svgp=dynamics.desired_mode_gating_gp
        )[0, :, :]
        Kmm += tf.eye(Kmm.shape[0], dtype=default_float()) * default_jitter()
        print("Knn.shape")
        print(Knn.shape)
        print(Kmm.shape)
        print(Kmn.shape)
        h_mean_conditioned, h_var_conditioned = base_conditional(
            Kmn=Kmn,
            Kmm=Kmm,
            Knn=Knn,
            f=f,
            full_cov=False,
            white=False,
        )
        h_means_conditioned.append(h_mean_conditioned)
        h_vars_conditioned.append(h_var_conditioned)
    h_means_conditioned = tf.concat(h_means_conditioned, 0)
    h_vars_conditioned = tf.concat(h_vars_conditioned, 0)

    h_dist = tfd.Normal(loc=h_means_conditioned, scale=h_vars_conditioned**2)
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
    input_means = input_dists.mean()

    def bald_objective_closed_form_traj():
        def entropy_approx(h_means, h_vars, mode_probs):
            C = tf.constant(np.sqrt(math.pi * np.log(2.0) / 2.0), dtype=default_float())
            # param_entropy = C * tf.exp(-(h_means**2) / (2 * (h_vars**2 + C**2)))
            # param_entropy = param_entropy / (tf.sqrt(h_vars**2 + C**2))
            param_entropy = C * tf.exp(-(h_means**2) / (2 * (h_vars + C**2)))
            param_entropy = param_entropy / (tf.sqrt(h_vars + C**2))
            print("param_entropy")
            print(param_entropy)
            model_entropy = binary_entropy(mode_probs)
            print(model_entropy)
            return model_entropy - param_entropy

        # print("h_covs.shape yoyyoyoyoy")
        # print(h_covs.shape)
        h_means_conditioned, h_vars_conditioned = [], []
        for t in range(1, initial_solution.horizon):
            state = input_means[t : t + 1, :]
            print("state.shape")
            print(state.shape)
            state_traj = tf.concat([input_means[:t, :], input_means[t + 1 :, :]], 0)
            f, h_covs = dynamics.mosvgpe.gating_network.gp.predict_f(
                state_traj, full_cov=True
            )
            print(state_traj.shape)
            # f = tf.concat([h_means[:t, :], h_means[t + 1 :, :]], 0)
            print("f.shape")
            print(f.shape)
            Knn = svgp_covariance_conditional(
                X1=state, X2=state, svgp=dynamics.desired_mode_gating_gp
            )[0, 0, :]
            Kmm = svgp_covariance_conditional(
                X1=state_traj, X2=state_traj, svgp=dynamics.desired_mode_gating_gp
            )[0, :, :]
            Kmn = svgp_covariance_conditional(
                X1=state_traj, X2=state, svgp=dynamics.desired_mode_gating_gp
            )[0, :, :]
            Kmm += tf.eye(Kmm.shape[0], dtype=default_float()) * default_jitter()
            print("Knn.shape")
            print(Knn.shape)
            print(Kmm.shape)
            print(Kmn.shape)
            # Lm = tf.linalg.cholesky(Kmm)
            h_mean_conditioned, h_var_conditioned = base_conditional(
                Kmn=Kmn,
                Kmm=Kmm,
                Knn=Knn,
                f=f,
                full_cov=False,
                # q_sqrt=tf.linalg.cholesky(h_covs),
                # q_sqrt=None,  # TODO make this h_var??
                white=False,
            )
            h_means_conditioned.append(h_mean_conditioned)
            h_vars_conditioned.append(h_var_conditioned)
        h_means_conditioned = tf.concat(h_means_conditioned, 0)
        h_vars_conditioned = tf.concat(h_vars_conditioned, 0)

        h_means, h_covs = dynamics.mosvgpe.gating_network.gp.predict_f(
            input_dists.mean(),
            # full_cov=False
            full_cov=True,
        )
        mode_probs = dynamics.mosvgpe.gating_network.gp.likelihood.predict_mean_and_var(
            input_means, Fmu=h_means, Fvar=h_covs
        )[0]
        print("mode_probs.shape")
        print(mode_probs.shape)
        mode_probs = tf.transpose(tf.linalg.diag_part(mode_probs))[1:, :]
        print(mode_probs.shape)
        # print(h_means.shape)
        # print(h_vars.shape)
        print(h_means_conditioned.shape)
        print(h_vars_conditioned.shape)
        return entropy_approx(
            h_means_conditioned,
            h_vars_conditioned,
            mode_probs,
            # h_means_conditioned[:, dynamics.desired_mode],
            # h_vars_conditioned[:, dynamics.desired_mode],
            # mode_probs[:, dynamics.desired_mode],
        )

    def bald_objective_closed_form():
        h_means, h_vars = dynamics.mosvgpe.gating_network.gp.predict_f(
            input_dists.mean(),
            full_cov=True
            # full_cov=False
            # input_dists.mean(), full_cov=True
        )
        print("YOYOYO")
        print(h_means.shape)
        print(h_vars.shape)

        def entropy_approx(h_means, h_vars, mode_probs):
            C = tf.constant(np.sqrt(math.pi * np.log(2.0) / 2.0), dtype=default_float())
            # param_entropy = C * tf.exp(-(h_means**2) / (2 * (h_vars**2 + C**2)))
            # param_entropy = param_entropy / (tf.sqrt(h_vars**2 + C**2))
            param_entropy = C * tf.exp(-(h_means**2) / (2 * (h_vars + C**2)))
            param_entropy = param_entropy / (tf.sqrt(h_vars + C**2))
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
            # Kmm += tf.eye(Kmm.shape[0], dtype=default_float()) * default_jitter()
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
        # h_vars+tf.eye(h_vars.shape[1], h_vars.shape[2], dtype=default_float()) * 1e-6
        # )
        # h_dist = tfd.Normal(h_means, h_vars[0, :, :] ** 2)
        # gating_entropy = h_dist.entropy()

        # h_means, h_vars = conditional_gating_temporal(
        #     state_dists=state_dists[1:], control_dists=control_dists
        # )
        # #     state_means[1:, :], control_means, state_vars[1:, :], control_vars
        # # )
        mode_probs = dynamics.mosvgpe.gating_network.gp.likelihood.predict_mean_and_var(
            input_means, Fmu=h_means, Fvar=h_vars
        )[0]

        print("mode_probs.shape")
        print(mode_probs.shape)
        print(h_means.shape)
        print(h_vars.shape)
        print(h_means_conditioned.shape)
        print(h_vars_conditioned.shape)
        return entropy_approx(
            h_means_conditioned,
            h_vars_conditioned,
            mode_probs,
            # h_means_conditioned[:, dynamics.desired_mode],
            # h_vars_conditioned[:, dynamics.desired_mode],
            # mode_probs[:, dynamics.desired_mode],
        )

    bald_objective = bald_objective_closed_form_traj()
    # bald_objective = bald_objective_closed_form()
    # bald_objective = bald_objective_sampling(input_dists)
    print("bald_objective")
    print(bald_objective)
    tf.print("entropy")
    tf.print(tf.reduce_sum(bald_objective))
    return tf.reduce_sum(bald_objective)
