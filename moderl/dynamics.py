#!/usr/bin/env python3
from dataclasses import dataclass
from functools import partial
from typing import Optional

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float, posteriors
from gpflow.conditionals import uncertain_conditional
from gpflow.models import SVGP
from gpflow.quadrature import NDiagGHQuadrature
from mosvgpe.custom_types import DatasetBatch
from mosvgpe.mixture_of_experts import MixtureOfSVGPExperts
from tensor_annotations.axes import Batch

from moderl.custom_types import (
    ControlDim,
    ControlMean,
    ControlTrajectoryMean,
    ControlTrajectoryVariance,
    ControlVariance,
    Dataset,
    One,
    StateDim,
    StateMean,
    StateMeanAndVariance,
    StateTrajectoryMean,
    StateTrajectoryVariance,
    StateVariance,
)
from moderl.utils import combine_state_controls_to_input


tfd = tfp.distributions

DEFAULT_NUM_GAUSS_HERMITE_POINTS = 20  # Uses too much memory!
DEFAULT_NUM_GAUSS_HERMITE_POINTS = 4

DEFAULT_DYNAMICS_FIT_KWARGS = {
    "batch_size": 16,
    "epochs": 1000,
    "verbose": True,
    "validation_split": 0.2,
}


# class ModeRLDynamics(tf.keras.Model):
class ModeRLDynamics:
    def __init__(
        self,
        mosvgpe: MixtureOfSVGPExperts,
        state_dim: int,
        desired_mode: int = 1,
        dataset: Dataset = None,
        learning_rate: float = 0.01,
        epsilon: float = 1e-8,
        dynamics_fit_kwargs: dict = DEFAULT_DYNAMICS_FIT_KWARGS,
        name: str = "ModeRLDynamics",
    ):
        self.mosvgpe = mosvgpe
        self.state_dim = state_dim
        self.desired_mode = desired_mode
        self.dataset = dataset
        self.dynamics_fit_kwargs = dynamics_fit_kwargs
        self.name = name

        # Config the optimiser and compile mosvgpe using keras
        optimiser = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, epsilon=epsilon
        )
        self.mosvgpe.compile(optimizer=optimiser)

    def call(
        self,
        state_control,
        training: Optional[bool] = False,
        predict_state_difference: Optional[bool] = False,
    ):
        """Call the desired mode's GP dynamics model"""
        if not training:
            return self.desired_mode_dynamics_gp(
                state_mean=state_control[:, : self.state_dim],
                control_mean=state_control[:, self.state_dim :],
                # state_var=state_var,
                # control_var=control_var,
                predict_state_difference=predict_state_difference,
                add_noise=False,
            )

    def forward(
        self,
        state_mean: StateTrajectoryMean,
        control_mean: ControlTrajectoryMean,
        state_var: StateTrajectoryVariance = None,
        control_var: ControlTrajectoryVariance = None,
        predict_state_difference: Optional[bool] = False,
    ):
        return self.desired_mode_dynamics_gp(
            state_mean=state_mean,
            control_mean=control_mean,
            state_var=state_var,
            control_var=control_var,
            predict_state_difference=predict_state_difference,
            add_noise=False,
        )

    def optimise(self):
        X, Y = self.dataset
        self.mosvgpe(X)  # Needs to be called to build shapes
        # TODO: if callbacks in self.dynamics_fit_kwargs extract and append them
        self.mosvgpe.fit(X, Y, callbacks=self.callbacks, **self.dynamics_fit_kwargs)

    def update_dataset(self, dataset: Dataset):
        if self.dataset is not None:
            Xold, Yold = self.dataset
            Xnew, Ynew = dataset
            X = np.concatenate([Xold, Xnew], 0)
            Y = np.concatenate([Yold, Ynew], 0)
            self.dataset = (X, Y)
        else:
            self.dataset = dataset
        self.mosvgpe.num_data = self.dataset[0].shape[0]

    def predict_mode_probability(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ):
        h_mean, h_var = self.uncertain_predict_gating(
            state_mean=state_mean,
            control_mean=control_mean,
            state_var=state_var,
            control_var=control_var,
        )

        probs = self.mosvgpe.gating_network.predict_mixing_probs_given_h(h_mean, h_var)
        if probs.shape[-1] == 1:
            return probs
        else:
            return probs[:, self.desired_mode]

    def uncertain_predict_gating(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ):
        # TODO make this handle softmax likelihood (k>2). Just need to map over gps
        input_mean, input_var = combine_state_controls_to_input(
            state_mean=state_mean,
            control_mean=control_mean,
            state_var=state_var,
            control_var=control_var,
        )
        if input_var is None:
            h_mean, h_var = self.desired_mode_gating_gp.predict_f(
                input_mean, full_cov=False
            )
        else:
            h_mean, h_var = uncertain_conditional(
                input_mean,
                input_var,
                self.desired_mode_gating_gp.inducing_variable,
                kernel=self.desired_mode_gating_gp.kernel,
                q_mu=self.desired_mode_gating_gp.q_mu,
                q_sqrt=self.desired_mode_gating_gp.q_sqrt,
                mean_function=self.desired_mode_gating_gp.mean_function,
                full_output_cov=False,
                full_cov=False,
                white=self.desired_mode_gating_gp.whiten,
            )
        if self.mosvgpe.gating_network.num_gating_gps == 1:
            h_mean = tf.concat([h_mean, -h_mean], -1)
            h_var = tf.concat([h_var, h_var], -1)
        return h_mean, h_var

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

        input_mean, input_var = combine_state_controls_to_input(
            state_mean, control_mean, state_var, control_var
        )

        def f(Xnew):
            print("Xnew.shape")
            print(Xnew.shape)
            # TODO this only works for Bernoulli likelihood
            # gating_means, gating_vars = self.gating_gp.predict_fs(Xnew, full_cov=False)
            gating_means, gating_vars = self.mosvgpe.gating_network.gp.predict_f(
                Xnew, full_cov=False
            )
            print("gating_means.shape")
            print(gating_means.shape)
            # TODO how to set Y shape if more than two modes?
            Y = tf.ones(gating_means.shape, dtype=default_float()) * (
                self.desired_mode + 1
            )
            # Y = tf.ones(gating_means.shape, dtype=default_float()) * 2
            # Y = tf.ones(gating_means.shape, dtype=default_float())
            # Y = tf.zeros(gating_means.shape, dtype=default_float())
            gating_var_exp = self.mosvgpe.gating_network.gp.likelihood.predict_log_density(
                gating_means,
                gating_vars,
                Y,
                # gating_means[..., self.desired_mode],
                # gating_vars[..., self.desired_mode],
                # Y[..., self.desired_mode],
            )
            return tf.expand_dims(gating_var_exp, -1)

        gauss_quadrature = NDiagGHQuadrature(
            dim=input_mean.shape[-1], n_gh=DEFAULT_NUM_GAUSS_HERMITE_POINTS
        )
        var_exp = gauss_quadrature(f, input_mean, input_var)
        mode_var_exp = tf.reduce_sum(var_exp)

        print("input_mean.shape")
        print(input_mean.shape)
        print(input_var.shape)
        print("var_exp: {}".format(var_exp))
        print("mode_var_exp: {}".format(mode_var_exp))
        return mode_var_exp

    @property
    def desired_mode_dynamics_gp(self) -> SVGPDynamicsWrapper:
        return self._desired_mode_dynamics_gp

    @desired_mode_dynamics_gp.setter
    def desired_mode_dynamics_gp(self, dynamics_gp: SVGP) -> SVGPDynamicsWrapper:
        self._desired_mode_dynamics_gp = SVGPDynamicsWrapper(dynamics_gp)
        gpf.utilities.print_summary(self._desired_mode_dynamics_gp.svgp_posterior)

    @property
    def desired_mode(self):
        return self._desired_mode

    @desired_mode.setter
    def desired_mode(self, desired_mode: int):
        """Set the desired dynamics mode GP (and build GP posterior)"""
        print("setting desired mode to {}".format(desired_mode))
        assert desired_mode < self.mosvgpe.num_experts
        self._desired_mode = desired_mode
        self.desired_mode_dynamics_gp = self.mosvgpe.experts_list[desired_mode].gp
        self.desired_mode_gating_gp = self.mosvgpe.gating_network.gp

    @property
    def desired_mode_gating_gp(self):
        return self._desired_mode_gating_gp

    @desired_mode_gating_gp.setter
    def desired_mode_gating_gp(self, gp: SVGP):
        # TODO set this differently when K>2
        if self.mosvgpe.gating_network.num_gating_gps == 1:
            self._desired_mode_gating_gp = gp
        else:
            # TODO build a single output gp from a multi output gp
            raise NotImplementedError("How to convert multi output gp to single dim")

    @property
    def gating_gp(self):
        return self.mosvgpe.gating_network.gp

    # @gating_gp.setter
    # def gating_gp(self, gp: SVGP):
    #     # TODO set this differently when K>2
    #     if self.mosvgpe.gating_network.num_gating_gps == 1:
    #         self._gating_gp = gp
    #     else:
    #         # TODO build a single output gp from a multi output gp
    #         raise NotImplementedError("How to convert multi output gp to single dim")


class SVGPDynamicsWrapper:
    def __init__(self, svgp: SVGP):
        self.svgp_posterior = svgp.posterior(
            precompute_cache=posteriors.PrecomputeCacheType.TENSOR
        )
        # TODO make posterior work with hydra config
        # self.svgp_posterior = svgp

        def uncertain_predict_f(input_mean, input_var):
            # TODO use multidispatch to handle multi-output uncertain conditionals
            f_mean, f_var = multioutput_uncertain_conditional(
                input_mean,
                input_var,
                # inducing_variables=svgp.inducing_variable,
                inducing_variables=svgp.inducing_variable.inducing_variable,
                kernel=svgp.kernel,
                q_mu=svgp.q_mu,
                q_sqrt=svgp.q_sqrt,
                # mean_function=svgp.mean_function,
                mean_function=None,
                full_output_cov=False,
                full_cov=False,
                whiten=svgp.whiten,
            )
            # TODO propogate state-control uncertianty through mean function?
            f_mean = f_mean + svgp.mean_function(input_mean)
            return f_mean, f_var

        self.predict_f = partial(svgp.predict_f, full_cov=False, full_output_cov=False)
        self.uncertain_predict_f = uncertain_predict_f

    def __call__(
        self,
        state_mean: StateMean,
        control_mean: ControlMean,
        state_var: StateVariance = None,
        control_var: ControlVariance = None,
        predict_state_difference: bool = False,
        add_noise: bool = False,
    ) -> StateMeanAndVariance:

        input_mean, input_var = combine_state_controls_to_input(
            state_mean, control_mean, state_var=state_var, control_var=control_var
        )
        # print("input_mean.shape")
        # print(input_mean.shape)
        # print(input_var)
        if input_var is None:
            delta_state_mean, delta_state_var = self.predict_f(input_mean)
        else:
            delta_state_mean, delta_state_var = self.uncertain_predict_f(
                input_mean, input_var
            )
        # delta_state_mean, delta_state_var = self.predict_f(input_mean)
        if add_noise:
            delta_state_mean, delta_state_var = self.gp.likelihood.predict_mean_and_var(
                delta_state_mean, delta_state_var
            )

        if predict_state_difference:
            return delta_state_mean, delta_state_var
        else:
            next_state_mean = state_mean + delta_state_mean
            if state_var is None:
                next_state_var = delta_state_var
            else:
                next_state_var = state_var + delta_state_var
            return next_state_mean, next_state_var


def multioutput_uncertain_conditional(
    input_mean,
    input_var,
    inducing_variables,
    kernel,
    q_mu,
    q_sqrt=None,
    mean_function=None,
    full_output_cov=False,
    full_cov=False,
    whiten=False,
):
    # TODO map instead of for loop
    f_means, f_vars = [], []
    for i, (kernel) in enumerate(kernel.kernels):
        if len(q_sqrt.shape) == 2:
            q_sqrt_i = q_sqrt[:, i : i + 1]
        elif len(q_sqrt.shape) == 3:
            q_sqrt_i = q_sqrt[i : i + 1, :, :]
        f_mean, f_var = uncertain_conditional(
            input_mean,
            input_var,
            inducing_variables,
            # gp.inducing_variable.inducing_variables[0],
            kernel=kernel,
            q_mu=q_mu[:, i : i + 1],
            q_sqrt=q_sqrt_i,
            mean_function=mean_function,
            full_output_cov=full_output_cov,
            full_cov=full_cov,
            white=whiten,
        )
        f_means.append(f_mean)
        f_vars.append(f_var)
    if len(f_means) > 1:
        f_means = tf.concat(f_means, -1)
        f_vars = tf.concat(f_vars, -1)
    else:
        f_means = tf.constant(f_means)
        f_vars = tf.constant(f_vars)
    return f_means, f_vars


from typing import Optional

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float, posteriors
from gpflow.conditionals import uncertain_conditional
from gpflow.models import SVGP
from gpflow.quadrature import NDiagGHQuadrature
from mosvgpe.custom_types import DatasetBatch
from mosvgpe.mixture_of_experts import MixtureOfSVGPExperts
from tensor_annotations.axes import Batch

from moderl.custom_types import (
    ControlDim,
    ControlMean,
    ControlTrajectoryMean,
    ControlTrajectoryVariance,
    ControlVariance,
    One,
    StateDim,
    StateMean,
    StateMeanAndVariance,
    StateTrajectoryMean,
    StateTrajectoryVariance,
    StateVariance,
)
from moderl.utils import combine_state_controls_to_input

from .svgp import SVGPDynamicsWrapper

tfd = tfp.distributions

DEFAULT_NUM_GAUSS_HERMITE_POINTS = 20  # Uses too much memory!
DEFAULT_NUM_GAUSS_HERMITE_POINTS = 4


@dataclass
class ModeRLDynamics(tf.keras.Model):
    mosvgpe: MixtureOfSVGPExperts
    state_dim: int
    desired_mode: int = 1
    name: str = "ModeRLDynamics"

    def call(
        self,
        state_control,
        training: Optional[bool] = False,
        predict_state_difference: Optional[bool] = False,
    ):
        """Call the desired mode's GP dynamics model"""
        if not training:
            return self.desired_mode_dynamics_gp(
                state_mean=state_control[:, : self.state_dim],
                control_mean=state_control[:, self.state_dim :],
                # state_var=state_var,
                # control_var=control_var,
                predict_state_difference=predict_state_difference,
                add_noise=False,
            )

    def forward(
        self,
        state_mean: StateTrajectoryMean,
        control_mean: ControlTrajectoryMean,
        state_var: StateTrajectoryVariance = None,
        control_var: ControlTrajectoryVariance = None,
        predict_state_difference: Optional[bool] = False,
    ):
        return self.desired_mode_dynamics_gp(
            state_mean=state_mean,
            control_mean=control_mean,
            state_var=state_var,
            control_var=control_var,
            predict_state_difference=predict_state_difference,
            add_noise=False,
        )

    def train_step(self, data: DatasetBatch):
        with tf.GradientTape() as tape:
            loss = -self.mosvgpe.maximum_log_likelihood_objective(data)

        trainable_vars = self.mosvgpe.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.mosvgpe.loss_tracker.update_state(loss)
        return {
            "loss": self.mosvgpe.loss_tracker.result(),
            # "gating_kl": self.gating_kl_tracker.result(),
            # "experts_kl": self.experts_kl_tracker.result(),
        }

    def test_step(self, data):
        loss = -self.mosvgpe.maximum_log_likelihood_objective(data)
        return {"loss": loss}

    def predict_mode_probability(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ):
        h_mean, h_var = self.uncertain_predict_gating(
            state_mean=state_mean,
            control_mean=control_mean,
            state_var=state_var,
            control_var=control_var,
        )

        probs = self.mosvgpe.gating_network.predict_mixing_probs_given_h(h_mean, h_var)
        if probs.shape[-1] == 1:
            return probs
        else:
            return probs[:, self.desired_mode]

    def uncertain_predict_gating(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ):
        # TODO make this handle softmax likelihood (k>2). Just need to map over gps
        input_mean, input_var = combine_state_controls_to_input(
            state_mean=state_mean,
            control_mean=control_mean,
            state_var=state_var,
            control_var=control_var,
        )
        if input_var is None:
            h_mean, h_var = self.desired_mode_gating_gp.predict_f(
                input_mean, full_cov=False
            )
        else:
            h_mean, h_var = uncertain_conditional(
                input_mean,
                input_var,
                self.desired_mode_gating_gp.inducing_variable,
                kernel=self.desired_mode_gating_gp.kernel,
                q_mu=self.desired_mode_gating_gp.q_mu,
                q_sqrt=self.desired_mode_gating_gp.q_sqrt,
                mean_function=self.desired_mode_gating_gp.mean_function,
                full_output_cov=False,
                full_cov=False,
                white=self.desired_mode_gating_gp.whiten,
            )
        if self.mosvgpe.gating_network.num_gating_gps == 1:
            h_mean = tf.concat([h_mean, -h_mean], -1)
            h_var = tf.concat([h_var, h_var], -1)
        return h_mean, h_var

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

        input_mean, input_var = combine_state_controls_to_input(
            state_mean, control_mean, state_var, control_var
        )

        def f(Xnew):
            print("Xnew.shape")
            print(Xnew.shape)
            # TODO this only works for Bernoulli likelihood
            # gating_means, gating_vars = self.gating_gp.predict_fs(Xnew, full_cov=False)
            gating_means, gating_vars = self.mosvgpe.gating_network.gp.predict_f(
                Xnew, full_cov=False
            )
            print("gating_means.shape")
            print(gating_means.shape)
            # TODO how to set Y shape if more than two modes?
            Y = tf.ones(gating_means.shape, dtype=default_float()) * (
                self.desired_mode + 1
            )
            # Y = tf.ones(gating_means.shape, dtype=default_float()) * 2
            # Y = tf.ones(gating_means.shape, dtype=default_float())
            # Y = tf.zeros(gating_means.shape, dtype=default_float())
            gating_var_exp = self.mosvgpe.gating_network.gp.likelihood.predict_log_density(
                gating_means,
                gating_vars,
                Y,
                # gating_means[..., self.desired_mode],
                # gating_vars[..., self.desired_mode],
                # Y[..., self.desired_mode],
            )
            return tf.expand_dims(gating_var_exp, -1)

        gauss_quadrature = NDiagGHQuadrature(
            dim=input_mean.shape[-1], n_gh=DEFAULT_NUM_GAUSS_HERMITE_POINTS
        )
        var_exp = gauss_quadrature(f, input_mean, input_var)
        mode_var_exp = tf.reduce_sum(var_exp)

        print("input_mean.shape")
        print(input_mean.shape)
        print(input_var.shape)
        print("var_exp: {}".format(var_exp))
        print("mode_var_exp: {}".format(mode_var_exp))
        return mode_var_exp

    @property
    def desired_mode_dynamics_gp(self) -> SVGPDynamicsWrapper:
        return self._desired_mode_dynamics_gp

    @desired_mode_dynamics_gp.setter
    def desired_mode_dynamics_gp(self, dynamics_gp: SVGP) -> SVGPDynamicsWrapper:
        self._desired_mode_dynamics_gp = SVGPDynamicsWrapper(dynamics_gp)
        gpf.utilities.print_summary(self._desired_mode_dynamics_gp.svgp_posterior)

    @property
    def desired_mode(self):
        return self._desired_mode

    @desired_mode.setter
    def desired_mode(self, desired_mode: int):
        """Set the desired dynamics mode GP (and build GP posterior)"""
        print("setting desired mode to {}".format(desired_mode))
        assert desired_mode < self.mosvgpe.num_experts
        self._desired_mode = desired_mode
        self.desired_mode_dynamics_gp = self.mosvgpe.experts_list[desired_mode].gp
        self.desired_mode_gating_gp = self.mosvgpe.gating_network.gp

    @property
    def desired_mode_gating_gp(self):
        return self._desired_mode_gating_gp

    @desired_mode_gating_gp.setter
    def desired_mode_gating_gp(self, gp: SVGP):
        # TODO set this differently when K>2
        if self.mosvgpe.gating_network.num_gating_gps == 1:
            self._desired_mode_gating_gp = gp
        else:
            # TODO build a single output gp from a multi output gp
            raise NotImplementedError("How to convert multi output gp to single dim")

    @property
    def gating_gp(self):
        return self.mosvgpe.gating_network.gp

    # @gating_gp.setter
    # def gating_gp(self, gp: SVGP):
    #     # TODO set this differently when K>2
    #     if self.mosvgpe.gating_network.num_gating_gps == 1:
    #         self._gating_gp = gp
    #     else:
    #         # TODO build a single output gp from a multi output gp
    #         raise NotImplementedError("How to convert multi output gp to single dim")


class SVGPDynamicsWrapper:
    def __init__(self, svgp: SVGP):
        self.svgp_posterior = svgp.posterior(
            precompute_cache=posteriors.PrecomputeCacheType.TENSOR
        )
        # TODO make posterior work with hydra config
        # self.svgp_posterior = svgp

        def uncertain_predict_f(input_mean, input_var):
            # TODO use multidispatch to handle multi-output uncertain conditionals
            f_mean, f_var = multioutput_uncertain_conditional(
                input_mean,
                input_var,
                # inducing_variables=svgp.inducing_variable,
                inducing_variables=svgp.inducing_variable.inducing_variable,
                kernel=svgp.kernel,
                q_mu=svgp.q_mu,
                q_sqrt=svgp.q_sqrt,
                # mean_function=svgp.mean_function,
                mean_function=None,
                full_output_cov=False,
                full_cov=False,
                whiten=svgp.whiten,
            )
            # TODO propogate state-control uncertianty through mean function?
            f_mean = f_mean + svgp.mean_function(input_mean)
            return f_mean, f_var

        self.predict_f = partial(svgp.predict_f, full_cov=False, full_output_cov=False)
        self.uncertain_predict_f = uncertain_predict_f

    def __call__(
        self,
        state_mean: StateMean,
        control_mean: ControlMean,
        state_var: StateVariance = None,
        control_var: ControlVariance = None,
        predict_state_difference: bool = False,
        add_noise: bool = False,
    ) -> StateMeanAndVariance:

        input_mean, input_var = combine_state_controls_to_input(
            state_mean, control_mean, state_var=state_var, control_var=control_var
        )
        # print("input_mean.shape")
        # print(input_mean.shape)
        # print(input_var)
        if input_var is None:
            delta_state_mean, delta_state_var = self.predict_f(input_mean)
        else:
            delta_state_mean, delta_state_var = self.uncertain_predict_f(
                input_mean, input_var
            )
        # delta_state_mean, delta_state_var = self.predict_f(input_mean)
        if add_noise:
            delta_state_mean, delta_state_var = self.gp.likelihood.predict_mean_and_var(
                delta_state_mean, delta_state_var
            )

        if predict_state_difference:
            return delta_state_mean, delta_state_var
        else:
            next_state_mean = state_mean + delta_state_mean
            if state_var is None:
                next_state_var = delta_state_var
            else:
                next_state_var = state_var + delta_state_var
            return next_state_mean, next_state_var


def multioutput_uncertain_conditional(
    input_mean,
    input_var,
    inducing_variables,
    kernel,
    q_mu,
    q_sqrt=None,
    mean_function=None,
    full_output_cov=False,
    full_cov=False,
    whiten=False,
):
    # TODO map instead of for loop
    f_means, f_vars = [], []
    for i, (kernel) in enumerate(kernel.kernels):
        if len(q_sqrt.shape) == 2:
            q_sqrt_i = q_sqrt[:, i : i + 1]
        elif len(q_sqrt.shape) == 3:
            q_sqrt_i = q_sqrt[i : i + 1, :, :]
        f_mean, f_var = uncertain_conditional(
            input_mean,
            input_var,
            inducing_variables,
            # gp.inducing_variable.inducing_variables[0],
            kernel=kernel,
            q_mu=q_mu[:, i : i + 1],
            q_sqrt=q_sqrt_i,
            mean_function=mean_function,
            full_output_cov=full_output_cov,
            full_cov=full_cov,
            white=whiten,
        )
        f_means.append(f_mean)
        f_vars.append(f_var)
    if len(f_means) > 1:
        f_means = tf.concat(f_means, -1)
        f_vars = tf.concat(f_vars, -1)
    else:
        f_means = tf.constant(f_means)
        f_vars = tf.constant(f_vars)
    return f_means, f_vars
