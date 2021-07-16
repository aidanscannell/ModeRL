#!/usr/bin/env python3
import abc

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import Module, Parameter

# from gpflow.models import BayesianModel
from gpflow.config import default_float
from gpflow.utilities import positive, triangular

tfd = tfp.distributions


class VariationalPolicy(abc.ABC, Module):
    @property
    @abc.abstractmethod
    def variational_dist(self) -> tfd.Distribution:
        raise NotImplementedError

    def sample(self, num_samples):
        return self.variational_dist.sample(num_samples)  # [S, N, F]

    def entropy(self):
        return tf.reduce_sum(self.variational_dist.entropy())


class GaussianPolicy(VariationalPolicy):
    def __init__(self, means=None, vars=None):
        if means is None and vars is None:
            means = 0
            vars = 0
        elif means is None:
            num_time_steps = vars.shape[0]
            # self.means = init_means(num_time_steps)
        elif vars is None:
            num_time_steps = means.shape[0]
            # self.vars = init_vars(num_time_steps)
        else:
            assert means.shape[0] == vars.shape[0]
            # self.mean = means
            # self.vars = vars
        self.num_time_steps = means.shape[0]
        # self.means = tf.Variable(means)
        # self.vars = tf.Variable(vars)
        self.means = Parameter(means, dtype=default_float())
        self.vars = Parameter(vars, dtype=default_float(), transform=positive())
        # self.vars = vars
        # self._variational_dist = tfd.Normal(self.means, self.vars)
        self._variational_dist = tfd.MultivariateNormalDiag(
            loc=self.means, scale_diag=self.vars
        )

    def __call__(self, time_step=None):
        means = self.variational_dist.mean()
        # means = self.means
        if time_step is None:
            return means
        else:
            return means[time_step : time_step + 1, :]

    @property
    def variational_dist(self):
        return self._variational_dist


class GaussianMixturePolicy(VariationalPolicy):
    def __init__(self, means=None, vars=None, mixture_probs=None):
        # if means is None and vars is None:
        #     means = 0
        #     vars = 0
        # elif means is None:
        #     num_time_steps = vars.shape[0]
        #     # self.means = init_means(num_time_steps)
        # elif vars is None:
        #     num_time_steps = means.shape[0]
        #     # self.vars = init_vars(num_time_steps)
        # else:
        #     assert means.shape[0] == vars.shape[0]
        #     # self.mean = means
        #     # self.vars = vars
        self.num_time_steps = means[0].shape[0]
        self.means = Parameter(means, dtype=default_float())
        self.vars = Parameter(vars, dtype=default_float(), transform=positive())
        self.mixture_probs = Parameter(
            mixture_probs, dtype=default_float(), transform=positive()
        )
        mixture_distribution = tfd.Categorical(probs=self.mixture_probs)
        print(mixture_distribution)
        components_distribution = tfd.MultivariateNormalDiag(
            # TODO should vars be sqrt?
            loc=self.means,
            scale_diag=self.vars,
        )
        print(components_distribution)
        self._variational_dist = tfd.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            components_distribution=components_distribution,
        )
        print("variational_dist")
        print(self._variational_dist)
        print(self._variational_dist.sample(20))

    def __call__(self, time_step=None):
        means = self.variational_dist.mean()
        if time_step is None:
            return means
        else:
            return means[time_step : time_step + 1, :]

    @property
    def variational_dist(self):
        return self._variational_dist
