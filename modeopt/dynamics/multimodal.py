#!/usr/bin/env python3
import copy
from dataclasses import dataclass
from typing import Callable, NewType, Optional

import gin
import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import Module, default_float
from gpflow.conditionals import base_conditional

# from gpflow.conditionals import uncertain_conditional
from modeopt.dynamics import SVGPDynamics
from modeopt.dynamics.conditionals import (
    FakeInducingPoints,
    svgp_covariance_conditional,
    uncertain_conditional,
)
from modeopt.dynamics.gp import SVGPDynamics
from modeopt.dynamics.utils import create_tf_dataset
from mogpe.mixture_of_experts import MixtureOfSVGPExperts
from mogpe.training import MixtureOfSVGPExperts_from_toml
from mogpe.training.utils import update_model_from_checkpoint
from tensor_annotations import axes
from tensor_annotations.axes import Batch

# from modeopt.dynamics.utils import create_tf_dataset

StateDim = NewType("StateDim", axes.Axis)
ControlDim = NewType("ControlDim", axes.Axis)
StateControlDim = NewType("StateControlDim", axes.Axis)
One = NewType("One", axes.Axis)


@gin.configurable
def init_ModeOptDynamics_from_mogpe_ckpt(
    mogpe_config_file: str,
    dataset: Tuple,
    mogpe_ckpt_dir: str = None,
    nominal_dynamics: Callable = None,
    desired_mode: int = 0,
    optimiser: tf.optimizers.Optimizer = tf.optimizers.Adam(),
):
    X, Y = dataset
    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    model = MixtureOfSVGPExperts_from_toml(mogpe_config_file, dataset=(X, Y))

    if mogpe_ckpt_dir is not None:
        model = update_model_from_checkpoint(model, mogpe_ckpt_dir)
    dynamics = ModeOptDynamics(
        mosvgpe=model,
        desired_mode=desired_mode,
        state_dim=state_dim,
        control_dim=control_dim,
        nominal_dynamics=nominal_dynamics,
        optimiser=optimiser,
    )
    return dynamics


@dataclass
class ModeOptDynamicsTrainingSpec:
    """
    Specification data class for model training. Models that require additional parameters for
    training should create a subclass of this class and add additional properties.
    """

    num_epochs: int
    batch_size: int
    logging_epoch_freq: int
    monitor: gpf.monitor.Monitor = None
    manager: tf.train.CheckpointManager = None


class ModeOptDynamics(Module):
    def __init__(
        self,
        mosvgpe: MixtureOfSVGPExperts,
        desired_mode: int,
        state_dim: int,
        control_dim: int,
        nominal_dynamics: Callable = None,
        # optimizer: tf.optimizers.Optimizer = tf.optimizers.Adam(),
    ):
        self.mosvgpe = mosvgpe
        self._nominal_dynamics = nominal_dynamics
        self.set_desired_mode(desired_mode)
        self.state_dim = state_dim
        self.control_dim = control_dim

        self._gating_gp = self.mosvgpe.gating_network

        # self.optimizer = optimizer

    @property
    def desired_mode(self):
        return self._desied_mode

    @property
    def dynamics_gp(self):
        return self._dynamics_gp

    @property
    def gating_gp(self):
        return self._gating_gp

    def set_desired_mode(self, desired_mode):
        """Set the desired dynamics mode GP and create a SVGPDynamics"""
        assert desired_mode < self.mosvgpe.num_experts
        self._desied_mode = desired_mode
        self._dynamics_gp = self.mosvgpe.experts.experts_list[desired_mode]
        # self._desired_mode_dynamics = SVGPDynamics(dynamics_gp)
        self._desired_mode_dynamics = SVGPDynamics(
            self._dynamics_gp, nominal_dynamics=self._nominal_dynamics
        )

    def desired_mode_dynamics(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
        predict_state_difference: bool = False,
        add_noise: bool = False,
    ):
        return self._desired_mode_dynamics(
            state_mean,
            control_mean,
            state_var=state_var,
            control_var=control_var,
            predict_state_difference=predict_state_difference,
            add_noise=add_noise,
        )

    def __call__(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
        predict_state_difference: bool = False,
        add_noise: bool = False,
    ):
        """Call the desired mode's GP dynamics model"""
        return self.desired_mode_dynamics(
            state_mean,
            control_mean,
            state_var=state_var,
            control_var=control_var,
            predict_state_difference=predict_state_difference,
            add_noise=add_noise,
        )

    def predict_mode_probability(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ):
        input_mean = tf.concat([state_mean, control_mean], -1)
        if state_var is None or control_var is None:
            h_mean, h_var = self.gating_gp.predict_f(input_mean, full_cov=False)
        else:
            input_var = tf.concat([state_var, control_var], -1)
            h_mean, h_var = uncertain_conditional(
                input_mean,
                input_var,
                self.gating_gp.inducing_variable,
                kernel=self.gating_gp.kernel,
                q_mu=self.gating_gp.q_mu,
                q_sqrt=self.gating_gp.q_sqrt,
                mean_function=self.gating_gp.mean_function,
                full_output_cov=False,
                full_cov=False,
                white=self.gating_gp.whiten,
            )

        return self.predict_mode_probability_given_latent(h_mean, h_var)

    def predict_mode_probability_given_latent(
        self,
        h_mean: ttf.Tensor2[Batch, One],
        h_var: ttf.Tensor2[Batch, One] = None,
    ):
        # probs = self.gating_gp.predict_y(h_mean, h_var)
        probs = self.gating_gp.predict_mixing_probs_given_h(h_mean, h_var)
        print("probs")
        print(probs)
        if probs.shape[-1] == 1:
            return probs
        else:
            return probs[:, self.desired_mode]

    def mode_variational_expectation(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ):
        """Calculate expected log mode probability under trajectory distribution given by,

        \sum_{t=1}^T \E_{p(\state_t, \control_t)} [\log \Pr(\alpha=k_* \mid \state_t, \control_t)]

        \sum_{t=1}^T \E_{p(\state_t, \control_t, h)} [\log \Pr(\alpha=k_* \mid h(\state_t, \control_t) )]
        """
        # Calculate expected mode prob
        gating_means, gating_vars = self.uncertain_predict_gating(
            state_mean, control_mean, state_var, control_var
        )
        print("gating_means")
        print(gating_means)
        print(gating_vars)
        Y = tf.ones(gating_means.shape, dtype=default_float()) * self.desired_mode
        gating_var_exp = self.gating_gp.likelihood.variational_expectations(
            gating_means, gating_vars, Y
        )
        print("gating_var_exp")
        print(gating_var_exp)
        sum_gating_var_exp = tf.reduce_sum(gating_var_exp)
        print(sum_gating_var_exp)
        return sum_gating_var_exp

    def uncertain_predict_gating(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ):
        input_mean = tf.concat([state_mean, control_mean], -1)
        if state_var is None or control_var is None:
            h_mean, h_var = self.gating_gp.predict_f(input_mean, full_cov=False)
        else:
            input_var = tf.concat([state_var, control_var], -1)
            h_mean, h_var = uncertain_conditional(
                input_mean,
                input_var,
                # state_var,
                self.gating_gp.inducing_variable,
                kernel=self.gating_gp.kernel,
                q_mu=self.gating_gp.q_mu,
                q_sqrt=self.gating_gp.q_sqrt,
                mean_function=self.gating_gp.mean_function,
                full_output_cov=False,
                full_cov=False,
                white=self.gating_gp.whiten,
            )
        return h_mean, h_var

    def gating_conditional_entropy(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ):
        # inducing_variable_z = self.gating_gp.inducing_variable.Z
        # num_inducing = 90
        # inducing_variable = self.gating_gp.inducing_variable
        # q_mu = self.gating_gp.q_mu
        # q_sqrt = self.gating_gp.q_sqrt

        # h_mean, h_var = self.uncertain_predict_gating(
        #     state_mean[0:1, :],
        #     control_mean[0:1, :],
        #     # state_var[0:1, :],
        #     # control_var[0:1, :],
        # )
        h_means_prior, h_vars_prior = self.uncertain_predict_gating(
            state_mean,
            control_mean,
            # state_var[0:1, :],
            # control_var[0:1, :],
        )
        # print("h_mean_0.shape")
        # print(h_mean.shape)
        # print(h_var.shape)
        # tf.print("h_means_ init")
        # tf.print(h_means_)
        # tf.print(h_vars_)

        def unc_cond(input_mean, input_var, q_mu, q_sqrt, inducing_variable):
            print("inside unc_cond")
            print(input_mean.shape)
            print(input_var.shape)
            h_mean, h_var = uncertain_conditional(
                input_mean,
                input_var,
                inducing_variable,
                kernel=self.gating_gp.kernel,
                q_mu=q_mu,
                q_sqrt=q_sqrt,
                mean_function=self.gating_gp.mean_function,
                full_output_cov=False,
                full_cov=False,
                white=self.gating_gp.whiten,
            )
            print("after unc_cond")
            print(h_mean.shape)
            print(h_var.shape)
            return h_mean, h_var

        input_means = tf.concat([state_mean, control_mean], -1)
        input_vars = tf.concat([state_var, control_var], -1)
        print("input_means.shape")
        print(input_means.shape)
        print(input_vars.shape)
        tf.print("input_means")
        tf.print(input_means)
        horizon = state_mean.shape[0]
        h_means = h_means_prior[0:1, :]
        h_vars = h_vars_prior[0:1, :]
        for t in range(1, horizon):
            # h_mean, h_var = self._gating_gp.predict_f(
            #     input_means[t : t + 1, :], full_cov=False, full_output_cov=False
            # )
            Xnew = input_means[t : t + 1, :]
            Xobs = input_means[0:t, :]
            f = h_means_prior[0:t, :]
            # Kxs = self._gating_gp.kernel(Xobs, Xnew)
            # # Kxs = self._gating_gp.kernel(Xnew, Xobs)
            # Kxx = self._gating_gp.kernel(Xobs)
            # h_cov = h_var[0, :]
            # tf.print("prior cov")
            # tf.print(h_cov)
            # f = h_means_[0:t, :]

            Knn = svgp_covariance_conditional(X1=Xnew, X2=Xnew, svgp=self.gating_gp)[
                0, 0, :
            ]
            print("Knn.shape")
            print(Knn.shape)
            Kmm = svgp_covariance_conditional(X1=Xobs, X2=Xobs, svgp=self.gating_gp)[
                0, :, :
            ]
            print("Kmm.shape")
            print(Kmm.shape)
            Kmn = svgp_covariance_conditional(X1=Xobs, X2=Xnew, svgp=self.gating_gp)[
                0, :, :
            ]
            print("Kmn.shape")
            print(Kmn.shape)
            tf.print("prior term")
            tf.print(Knn)
            # Lm = tf.broadcast_to(Lm, tf.concat([leading_dims, tf.shape(Lm)], 0))  # [..., M, M]
            Lm = tf.linalg.cholesky(Kmm)
            A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]
            tf.print("A")
            tf.print(A)
            post_term = tf.linalg.matmul(A, A, transpose_a=True)  # [..., N1, N2]
            # post_term = tf.reduce_sum(tf.square(A), -2)  # [..., N]
            tf.print("post_term")
            tf.print(post_term)
            h_mean, h_var = base_conditional(
                Kmn=Kmn,
                Kmm=Kmm,
                Knn=Knn,
                f=f,
                full_cov=False,
                q_sqrt=None,
                # white=False,
                white=True,
            )

            # # svgp_conditional(X1=Xnew,X2=Xnew, kernel=self.gating_gp.kernel,q_mu=q_mu,q_sqrt=q_sqrt)
            # Kmm = svgp_conditional(X1=Xobs, X2=Xobs, svgp=self.gating_gp)
            # print("Kmm new")
            # print(Kmm.shape)
            # tf.print("Kmm")
            # tf.print(Kmm)
            # Kmn = svgp_conditional(X1=Xobs, X2=Xnew, svgp=self.gating_gp)
            # print("Kmn new")
            # print(Kmn.shape)
            # tf.print("Kmn")
            # tf.print(Kmn)
            # Knn = svgp_conditional(X1=Xnew, X2=Xnew, svgp=self.gating_gp)
            # # Knn = tf.expand_dims(Knn, 0)
            # Knn = Knn[:, 0]
            # print("Knn new")
            # print(Knn.shape)
            # tf.print("Knn")
            # tf.print(Knn)

            # covariance_conditional(
            #     Km1=Km1,
            #     Km2=Km2,
            #     Lm=Lm,
            #     Knn=Knn,
            #     f=f,
            #     full_cov=False,
            #     q_sqrt=q_sqrt,
            #     white=False,
            # )

            # # tf.print("f")
            # # tf.print(f)
            # h_mean, h_var = base_conditional(
            #     Kmn=Kmn,
            #     Kmm=Kmm,
            #     Knn=Knn,
            #     f=f,
            #     full_cov=False,
            #     # q_sqrt=self.gating_gp.q_sqrt,
            #     q_sqrt=None,
            #     white=False,
            #     # white=True,
            # )
            # h_mean, h_var = base_conditional(
            #     Kmn=Kxs,
            #     Kmm=Kxx,
            #     Knn=h_cov,
            #     f=f,
            #     full_cov=False,
            #     # q_sqrt=self.gating_gp.q_sqrt,
            #     q_sqrt=None,
            #     white=False,
            #     # white=True,
            # )
            print("h_mean")
            print(h_mean.shape)
            print(h_var.shape)
            tf.print("h_mean")
            tf.print(h_mean)
            tf.print("h_var")
            tf.print(h_var)
            # h_mean, h_var = uncertain_conditional(
            #     input_means[t : t + 1, :],
            #     input_vars[t : t + 1, :],
            #     inducing_variable,
            #     kernel=self.gating_gp.kernel,
            #     q_mu=q_mu,
            #     q_sqrt=q_sqrt,
            #     mean_function=self.gating_gp.mean_function,
            #     full_output_cov=False,
            #     full_cov=False,
            #     white=self.gating_gp.whiten,
            # )
            h_means = tf.concat([h_means, h_mean], 0)
            h_vars = tf.concat([h_vars, h_var], 0)
            print("h_means")
            print(h_means.shape)
            print(h_vars.shape)
        # self._gating_gp.inducing_variable.Z = inducing_variable.Z[:num_inducing, :]
        # self._gating_gp.q_mu = q_mu[:num_inducing, :]
        # self._gating_gp.q_sqrt = q_sqrt[:, :num_inducing, :num_inducing]
        return h_means, h_vars

    # def _train(
    #     self,
    #     state_control_inputs: ttf.Tensor2[Batch, StateControlDim],
    #     delta_state_outputs: ttf.Tensor2[Batch, StateDim],
    #     training_spec: ModeOptDynamicsTrainingSpec,
    # ):
    #     """Train the transition_model given the trajectories and a training_spec

    #     :param training_spec: training specifications with `batch_size`, `epochs`, `callbacks` etc.
    #     """

    #     training_loss, num_batches_per_epoch = self._build_training_loss(
    #         state_control_inputs, delta_state_outputs, training_spec=training_spec
    #     )
    #     # TODO make mosvgpe trainable?

    #     @tf.function
    #     def tf_optimization_step():
    #         return self.optimizer.minimize(
    #             training_loss, self.mosvgpe.trainable_variables
    #         )

    #     for epoch in range(training_spec.num_epochs):
    #         for _ in range(num_batches_per_epoch):
    #             tf_optimization_step()
    #         if self.monitor is not None:
    #             training_spec.monitor(epoch)
    #         epoch_id = epoch + 1
    #         if epoch_id % training_spec.logging_epoch_freq == 0:
    #             tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")
    #             if training_spec.manager is not None:
    #                 training_spec.manager.save()

    # def _build_training_loss(
    #     self,
    #     state_control_inputs: ttf.Tensor2[Batch, StateControlDim],
    #     delta_state_outputs: ttf.Tensor2[Batch, StateDim],
    #     training_spec: ModeOptDynamicsTrainingSpec,
    # ) -> Callable:
    #     """Creates a tf dataset that can be iterated and build training loss closure"""
    #     train_dataset, num_batches_per_epoch = create_tf_dataset(
    #         dataset=(state_control_inputs, delta_state_outputs),
    #         batch_size=training_spec.batch_size,
    #     )
    #     training_loss = model.training_loss_closure(iter(train_dataset))

    #     num_train_data = state_control_inputs.shape[0]
    #     prefetch_size = tf.data.experimental.AUTOTUNE
    #     shuffle_buffer_size = num_train_data // 2
    #     num_batches_per_epoch = num_train_data // training_spec.batch_size

    #     train_dataset = (
    #         train_dataset.repeat()
    #         .prefetch(prefetch_size)
    #         .shuffle(buffer_size=shuffle_buffer_size)
    #         .batch(training_spec.training_batch_size)
    #     )

    #     print(f"prefetch_size={prefetch_size}")
    #     print(f"shuffle_buffer_size={shuffle_buffer_size}")
    #     print(f"num_batches_per_epoch={num_batches_per_epoch}")

    #     training_loss = self._model.training_loss_closure(iter(train_dataset))
    #     return training_loss, num_batches_per_epoch

    # def _build_training_loss(
    #     self,
    #     latent_trajectories: Trajectory,
    #     training_spec: TransitionModelTrainingSpec,
    # ) -> Callable:
    #     transition = extract_transitions_from_trajectories(
    #         latent_trajectories,
    #         self.latent_observation_space_spec,
    #         self.action_space_spec,
    #         self.predict_state_difference,
    #     )

    #     train_dataset = transitions_to_tf_dataset(
    #         transition, predict_state_difference=self.predict_state_difference
    #     )
    #     num_train_data = train_dataset[0].shape[0]

    #     prefetch_size = tf.data.experimental.AUTOTUNE
    #     shuffle_buffer_size = num_train_data // 2
    #     num_batches_per_epoch = num_train_data // training_spec.training_batch_size

    #     train_dataset = (
    #         train_dataset.repeat()
    #         .prefetch(prefetch_size)
    #         .shuffle(buffer_size=shuffle_buffer_size)
    #         .batch(training_spec.training_batch_size)
    #     )

    #     print(f"prefetch_size={prefetch_size}")
    #     print(f"shuffle_buffer_size={shuffle_buffer_size}")
    #     print(f"num_batches_per_epoch={num_batches_per_epoch}")

    #     training_loss = self._model.training_loss_closure(iter(train_dataset))
    #     return training_loss, num_batches_per_epoch

    # def _train(
    #     self,
    #     latent_trajectories: Trajectory,
    #     training_spec: TransitionModelTrainingSpec,
    # ):
    #     """Train the transition_model given the trajectories and a training_spec

    #     :param latent_trajectories: Trajectories of latent states, actions, rewards and next latent
    #         states.
    #     :param training_spec: training specifications with `batch_size`, `epochs`, `callbacks` etc.
    #     """

    #     training_loss, num_batches_per_epoch = self._build_training_loss(
    #         latent_trajectories, training_spec
    #     )

    #     @tf.function
    #     def tf_optimization_step():
    #         return self.optimizer.minimize(
    #             training_loss, self._model.trainable_variables
    #         )

    #     for epoch in range(training_spec.num_epochs):
    #         for _ in range(num_batches_per_epoch):
    #             tf_optimization_step()
    #         if self.monitor is not None:
    #             self.monitor(epoch)
    #         epoch_id = epoch + 1
    #         if epoch_id % training_spec.logging_epoch_freq == 0:
    #             tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")
    #             if self.manager is not None:
    #                 self.manager.save()

    # def unwhiten(q_mu, q_sqrt, Z):
    #     Kzz = self.gating_gp.kernel(Z, Z)
    #     Lz = tf.linalg.cholesky(Kzz)
    #     u_tilde = tf.linalg.triangular_solve(Lz, q_mu, lower=True)
    #     u = u_tilde + self.gating_gp.mean_function(Z)
    #     S_tilde = tf.linalg.triangular_solve(Lz, q_sqrt, lower=True)
    #     tf.print("S_tilde")
    #     tf.print(S_tilde.shape)
    #     S_tilde = S_tilde @ tf.transpose(S_tilde)
    #     Lu = tf.linalg.cholesky(S_tilde)
    #     # S_tilde = tf.linalg.triangular_solve(Lz, tf.transpose(S_tilde), lower=True)
    #     # tf.print(S_tilde.shape)
    #     # tf.print(S_tilde)
    #     # S_tilde = tf.transpose(S_tilde)
    #     # tf.print(S_tilde.shape)
    #     # tf.print(S_tilde)
    #     return u, Lu

    # def whiten(u, Lu, Z):
    #     Kzz = self.gating_gp.kernel(Z, Z)
    #     Lz = tf.linalg.cholesky(Kzz)
    #     print("Lz")
    #     print(Lz.shape)
    #     u_tilde = u - self.gating_gp.mean_function(Z)
    #     print("u_tilde.shape")
    #     print(u_tilde.shape)
    #     u_hat = Lz @ u_tilde
    #     return u_hat
