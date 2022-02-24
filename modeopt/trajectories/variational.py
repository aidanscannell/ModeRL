#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from modeopt.custom_types import ControlTrajectory, ControlTrajectoryMean, Horizon
from scipy.optimize import LinearConstraint

from .base import BaseTrajectory

tfd = tfp.distributions

SingleMeanAndVariance = None
TrajectoryMeanAndVariance = None


@dataclass
class ControlTrajectoryDist(BaseTrajectory):
    """Control trajectory distribution"""

    # dist: tfd.Distribution  # [horizon, control_dim]
    dist: Union[tfd.MultivariateNormalDiag, tfd.Deterministic]  # [horizon, control_dim]

    def __call__(
        self, timestep: Optional[int] = None, variance: Optional[bool] = False
    ) -> Union[ControlTrajectory, ControlTrajectoryMean]:
        if timestep is not None:
            idxs = [timestep, ...]
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


def initialise_gaussian_trajectory(
    horizon: int, control_dim: int, diag: Optional[bool] = True
) -> ControlTrajectoryDist:
    # controls = np.zeros((horizon, control_dim))
    # if diag:
    #     control_vars = (
    #         np.ones((horizon, control_dim)) * 0.2
    #         + np.random.random((horizon, control_dim)) * 0.01
    #     )
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


def build_control_constraints(
    trajectory: ControlTrajectoryDist,
    constraints_lower_bound=None,
    constraints_upper_bound=None,
):
    """Linear constraints on control trajectory"""
    if constraints_upper_bound is None or constraints_lower_bound is None:
        return None
    constraints_lower_bound = np.ones((trajectory.horizon, 1)) * constraints_lower_bound
    constraints_upper_bound = np.ones((trajectory.horizon, 1)) * constraints_upper_bound

    # TODO move this check to dispatcher?
    if isinstance(trajectory.dist, tfd.Deterministic):
        print("building constraints for tfd.Deterministic")
        control_constraint_matrix = np.eye(trajectory.horizon * trajectory.control_dim)
    else:
        control_constraint_matrix = np.eye(
            N=trajectory.horizon * trajectory.control_dim,
            M=trajectory.horizon * trajectory.control_dim * 2,
        )
    return LinearConstraint(
        control_constraint_matrix,
        constraints_lower_bound.reshape(-1),
        constraints_upper_bound.reshape(-1),
    )
