#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from modeopt.custom_types import ControlTrajectory, ControlTrajectoryMean, Horizon

from .base import BaseTrajectory

tfd = tfp.distributions

SingleMeanAndVariance = None
TrajectoryMeanAndVariance = None


@dataclass
class ControlTrajectoryDist(BaseTrajectory):
    """Control trajectory distribution"""

    dist: Union[tfd.MultivariateNormalDiag, tfd.Deterministic]  # [horizon, control_dim]

    def __call__(
        self, timestep: Optional[int] = None, variance: Optional[bool] = False
    ) -> Union[ControlTrajectory, ControlTrajectoryMean]:
        if timestep is not None:
            idxs = [timestep, ...]
        else:
            idxs = [...]
        if variance:
            return self.controls[idxs], self.control_vars[idxs]
        else:
            return self.controls[idxs]

    def entropy(self, sum_over_traj=True) -> Union[ttf.Tensor0, ttf.Tensor1[Horizon]]:
        if sum_over_traj:
            return tf.reduce_sum(self.dist.entropy())
        else:
            return self.dist.entropy()

    @property
    def controls(self):
        return self.dist.mean()

    @property
    def control_vars(self):
        return self.dist.variance()

    def copy(self):
        return ControlTrajectoryDist(self.dist.copy())

    def get_config(self):
        if isinstance(self.dist, tfd.Deterministic):
            variance = None
        else:
            variance = self.dist.variance().numpy()
        return {"mean": self.dist.mean().numpy(), "variance": variance}

    @classmethod
    def from_config(cls, cfg: dict):
        mean = tf.Variable(np.array(cfg["mean"], dtype=default_float()))
        try:
            variance = tf.Variable(np.array(cfg["variance"]))
            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=variance)
        except (KeyError, ValueError):
            dist = tfd.Deterministic(loc=mean)
        return cls(dist=dist)


def initialise_gaussian_trajectory(
    horizon: int, control_dim: int, diag: Optional[bool] = True
) -> ControlTrajectoryDist:
    controls = tf.Variable(np.zeros((horizon, control_dim)))
    if diag:
        control_vars = tf.Variable(
            np.ones((horizon, control_dim)) * 0.2
            + np.random.random((horizon, control_dim)) * 0.01
        )
        dist = tfd.MultivariateNormalDiag(loc=controls, scale_diag=control_vars)
    else:
        raise NotImplementedError()
    return ControlTrajectoryDist(dist)


def initialise_deterministic_trajectory(
    horizon: int, control_dim: int
) -> ControlTrajectoryDist:
    controls = tf.Variable(np.zeros((horizon, control_dim)))
    controls = tf.Variable(
        np.ones((horizon, control_dim)) * 0.2
        + np.random.random((horizon, control_dim)) * 0.01
    )
    controls = tf.Variable(controls)
    dist = tfd.Deterministic(loc=controls)
    return ControlTrajectoryDist(dist)
