#!/usr/bin/env python3
import abc
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import Parameter, default_float
from gpflow.utilities.bijectors import positive
from moderl.custom_types import ControlTrajectory, ControlTrajectoryMean, Horizon


tfd = tfp.distributions

SingleMeanAndVariance = None
TrajectoryMeanAndVariance = None


class BaseTrajectory(tf.Module, abc.ABC):
    def __call__(
        self, timestep: Optional[int] = None, variance: Optional[bool] = False
    ) -> Union[ControlTrajectory, ControlTrajectoryMean]:
        if variance:
            return self.controls, None
        else:
            return self.controls

    @property
    def controls(self) -> ControlTrajectoryMean:
        raise NotImplementedError

    @property
    def horizon(self) -> int:
        return self.controls.shape[0]

    @property
    def control_dim(self) -> int:
        return self.controls.shape[1]

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError


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
    # controls = tf.Variable(np.zeros((horizon, control_dim)))
    controls = tf.Variable(
        np.ones((horizon, control_dim)) * 0.2
        + np.random.random((horizon, control_dim)) * 0.01
    )
    if diag:
        control_vars = Parameter(
            np.ones((horizon, control_dim)) * 0.2
            + np.random.random((horizon, control_dim)) * 0.01,
            dtype=default_float(),
            transform=positive(),
        )
        dist = tfd.MultivariateNormalDiag(loc=controls, scale_diag=control_vars)
    else:
        raise NotImplementedError
    return ControlTrajectoryDist(dist)


def initialise_deterministic_trajectory(
    horizon: int, control_dim: int
) -> ControlTrajectoryDist:
    # controls = tf.Variable(np.zeros((horizon, control_dim)))
    # controls = tf.Variable(controls)
    controls = tf.Variable(
        np.ones((horizon, control_dim)) * 0.2
        + np.random.random((horizon, control_dim)) * 0.01
    )
    dist = tfd.Deterministic(loc=controls)
    return ControlTrajectoryDist(dist)
