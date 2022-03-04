#!/usr/bin/env python3
from typing import Optional
import gpflow as gpf

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import SVGP
from mogpe.custom_types import DatasetBatch
from mogpe.keras.mixture_of_experts import MixtureOfSVGPExperts
from modeopt.custom_types import (
    StateTrajectoryMean,
    StateTrajectoryVariance,
    ControlTrajectoryMean,
    ControlTrajectoryVariance,
)

from .svgp import SVGPDynamicsWrapper

tfd = tfp.distributions


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
