#!/usr/bin/env python3
import abc

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import Module, Parameter
from gpflow.config import default_float
from gpflow.utilities import positive
from scipy.optimize import LinearConstraint

tfd = tfp.distributions


class VariationalPolicy(abc.ABC, Module):
    """
    A trainable policy that can be optimised by a TrajectoryOptimiser and used for control
    """

    def __init__(self, controls, constraints_lower_bound, constraints_upper_bound):
        self.controls = controls
        self.num_time_steps, self.control_dim = controls.shape

        self.constraints_lower_bound = constraints_lower_bound
        self.constraints_upper_bound = constraints_upper_bound

    @property
    def variational_dist(self):
        return self._variational_dist

    @property
    def horizon(self):
        return self.num_time_steps

    def __call__(self, time_step: int = None):
        if time_step is None:
            return self.variational_dist
        else:
            return self.variational_dist[time_step : time_step + 1, :]

    def entropy(self):
        return tf.reduce_sum(self.variational_dist.entropy())

    def control_constraints(self):
        """Linear constraints on the mean of the control dist."""
        if self.constraints_upper_bound is None or self.constraints_lower_bound is None:
            return None
        constraints_lower_bound = (
            np.ones((self.num_time_steps, 1)) * self.constraints_lower_bound
        )
        constraints_upper_bound = (
            np.ones((self.num_time_steps, 1)) * self.constraints_upper_bound
        )
        control_constraint_matrix = np.eye(self.num_time_steps * self.control_dim)
        return LinearConstraint(
            control_constraint_matrix,
            constraints_lower_bound.reshape(-1),
            constraints_upper_bound.reshape(-1),
        )


class VariationalGaussianPolicy(VariationalPolicy):
    def __init__(
        self,
        means,
        vars,
        constraints_lower_bound=None,
        constraints_upper_bound=None,
    ):
        assert means.shape[0] == vars.shape[0]
        super().__init__(
            controls=means,
            constraints_lower_bound=constraints_lower_bound,
            constraints_upper_bound=constraints_upper_bound,
        )

        self.means = Parameter(means, dtype=default_float(), name="control_means")
        self.vars = Parameter(
            vars, dtype=default_float(), transform=positive(), name="control_vars"
        )

        self._variational_dist = tfd.MultivariateNormalDiag(
            loc=self.means, scale_diag=self.vars
        )

    def __call__(self, time_step: int = None):
        if time_step is None:
            return self.means, self.vars
        else:
            return (
                self.means[time_step : time_step + 1, :],
                self.vars[time_step : time_step + 1, :],
            )

    def control_constraints(self):
        if self.constraints_upper_bound is None or self.constraints_lower_bound is None:
            return None
        # Setup linear constraints on the controls
        constraints_lower_bound = (
            np.ones((self.num_time_steps, 1)) * self.constraints_lower_bound
        )
        constraints_upper_bound = (
            np.ones((self.num_time_steps, 1)) * self.constraints_upper_bound
        )
        # inf = np.ones(constraints_lower_bound.shape) * np.inf
        # constraints_lower_bound = np.concatenate([constraints_lower_bound, -inf], -1)
        # constraints_upper_bound = np.concatenate([constraints_upper_bound, inf], -1)
        # control_constraint_matrix = np.eye(self.num_time_steps * self.control_dim * 2)
        control_constraint_matrix = np.eye(
            N=self.num_time_steps * self.control_dim,
            M=self.num_time_steps * self.control_dim * 2,
        )
        return LinearConstraint(
            control_constraint_matrix,
            constraints_lower_bound.reshape(-1),
            constraints_upper_bound.reshape(-1),
        )


class DeterministicPolicy(VariationalPolicy):
    def __init__(
        self, controls=None, constraints_lower_bound=None, constraints_upper_bound=None
    ):
        super().__init__(
            controls=controls,
            constraints_lower_bound=constraints_lower_bound,
            constraints_upper_bound=constraints_upper_bound,
        )
        self.controls = Parameter(controls, dtype=default_float())
        self._variational_dist = tfd.Deterministic(loc=self.controls)

    def __call__(self, time_step: int = None):
        means = self.variational_dist.mean()
        vars = tf.zeros(means.shape, dtype=default_float())
        if time_step is None:
            return means, vars
        else:
            return (
                means[time_step : time_step + 1, :],
                vars[time_step : time_step + 1, :],
            )

    # def __call__(self, time_step=None):
    #     zeros = tf.zeros(self.knots.shape, dtype=default_float())
    #     if time_step is None:
    #         return
    #         # return self.knots, zeros
    #     else:
    #         return self.knots[time_step : time_step + 1, :], zeros
