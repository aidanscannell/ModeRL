#!/usr/bin/env python3
import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from geoflow.manifolds import GPManifold
from gpflow import default_float
from gpflow.models import SVGP, GPModel
from gpflow.utilities.keras import try_val_except_none
from modeopt.constraints import hermite_simpson_collocation_constraints_fn
from modeopt.custom_types import ControlDim, Horizon, StateDim, TwoStateDim

from .base import BaseTrajectory

tfd = tfp.distributions


class GeodesicTrajectory(BaseTrajectory):
    def __init__(
        self,
        start_state: ttf.Tensor1[StateDim],
        target_state: ttf.Tensor1[StateDim],
        gp: GPModel,
        riemannian_metric_covariance_weight: float = 1.0,
        horizon: int = None,
        t_init: float = -1.0,
        t_end: float = 1.0,
        name: str = "GeodesicTrajectory",
    ):
        super().__init__(name=name)
        assert len(start_state.shape) == 2
        assert len(target_state.shape) == 2
        self.start_state = start_state
        self.target_state = target_state

        times = np.linspace(t_init, t_end, horizon)
        self.times = tf.constant(times, dtype=default_float())

        self.initial_states = tf.linspace(
            start_state[0, :], target_state[0, :], horizon
        )
        self.state_variables = tf.Variable(self.initial_states[1:-1, :])

        self.manifold = GPManifold(
            gp=gp, covariance_weight=riemannian_metric_covariance_weight
        )

    def geodesic_collocation_constraints(self):
        return hermite_simpson_collocation_constraints_fn(
            state_at_knots=self.states_and_first_derivatives,
            times=self.times,
            ode_fn=self.manifold.geodesic_ode,
        )

    @property
    def states(self) -> ttf.Tensor2[Horizon, StateDim]:
        return tf.concat(
            [
                tf.concat([self.start_state, self.state_variables], 0),
                self.target_state,
            ],
            0,
        )

    @property
    def states_and_first_derivatives(self) -> ttf.Tensor2[Horizon, TwoStateDim]:
        return tf.concat([self.states, self.state_derivatives], -1)

    @property
    def state_derivatives(self) -> ttf.Tensor2[Horizon, TwoStateDim]:
        diff = self.states[1:, :] - self.states[:-1, :]
        return tf.concat(
            [diff, tf.zeros([1, self.state_dim], dtype=default_float())], 0
        )

    @property
    def controls(self) -> ttf.Tensor2[Horizon, ControlDim]:
        raise NotImplementedError

    @property
    def horizon(self) -> int:
        return self.states.shape[0]

    @property
    def state_dim(self) -> int:
        return self.states.shape[-1]

    def copy(self):
        return GeodesicTrajectory(
            start_state=self.start_state,
            target_state=self.target_state,
            gp=self.manifold.gp,
            riemannian_metric_covariance_weight=self.manifold.covariance_weight,
            horizon=self.horizon,
            t_init=self.times[0],
            t_end=self.times[-1],
        )

    def get_config(self):
        return {
            "start_state": self.start_state.numpy(),
            "target_state": self.target_state.numpy(),
            "gp": tf.keras.layers.serialize(self.manifold.gp),
            "riemannian_metric_covariance_weight": self.manifold.covariance_weight,
            "horizon": self.horizon,
            "t_init": self.times[0].numpy(),
            "t_end": self.times[-1].numpy(),
        }

    @classmethod
    def from_config(cls, cfg: dict):
        # TODO this should accept any type of GP not just SVGP...
        gp = tf.keras.layers.deserialize(cfg["gp"], custom_objects={"SVGP": SVGP})
        return cls(
            start_state=tf.constant(
                np.array(cfg["start_state"]), dtype=default_float()
            ),
            target_state=tf.constant(
                np.array(cfg["target_state"]), dtype=default_float()
            ),
            gp=gp,
            riemannian_metric_covariance_weight=cfg[
                "riemannian_metric_covariance_weight"
            ],
            horizon=try_val_except_none(cfg, "horizon"),
            t_init=try_val_except_none(cfg, "t_init"),
            t_end=try_val_except_none(cfg, "t_end"),
        )
