#!/usr/bin/env python3
from typing import Optional

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from gpflow.conditionals import uncertain_conditional
from gpflow.models import SVGP
from gpflow.quadrature import NDiagGHQuadrature
from modeopt.custom_types import (
    ControlDim,
    ControlTrajectoryMean,
    ControlTrajectoryVariance,
    One,
    StateDim,
    StateTrajectoryMean,
    StateTrajectoryVariance,
)
from modeopt.utils import combine_state_controls_to_input
from mogpe.custom_types import DatasetBatch
from mogpe.keras.mixture_of_experts import MixtureOfSVGPExperts
from tensor_annotations.axes import Batch

from .svgp import SVGPDynamicsWrapper

tfd = tfp.distributions

DEFAULT_NUM_GAUSS_HERMITE_POINTS = 20  # Uses too much memory!
DEFAULT_NUM_GAUSS_HERMITE_POINTS = 4


class ModeOptDynamics(tf.keras.Model):
    def __init__(
        self,
        mosvgpe: MixtureOfSVGPExperts,
        state_dim: int,
        desired_mode: int = 1,
        name: str = "ModeOptDynamics",
    ):
        super().__init__(name=name)
        self.mosvgpe = mosvgpe
        self.state_dim = state_dim
        self.desired_mode = desired_mode

    def call(
        self,
        # state_control: BatchedGaussianStateControl,
        state_control,
        # state_mean: StateTrajectoryMean,
        # control_mean: ControlTrajectoryMean,
        # state_var: StateTrajectoryVariance = None,
        # control_var: ControlTrajectoryVariance = None,
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

    def get_config(self):
        return {
            "mosvgpe": tf.keras.layers.serialize(self.mosvgpe),
            "state_dim": self.state_dim,
            "desired_mode": self.desired_mode,
        }

    @classmethod
    def from_config(cls, cfg: dict):
        mosvgpe = tf.keras.layers.deserialize(
            cfg["mosvgpe"],
            custom_objects={"MixtureOfSVGPExperts": MixtureOfSVGPExperts},
        )
        try:
            desired_mode = cfg["desired_mode"]
        except KeyError:
            desired_mode = 1
        return cls(
            mosvgpe=mosvgpe,
            state_dim=cfg["state_dim"],
            desired_mode=desired_mode,
        )
