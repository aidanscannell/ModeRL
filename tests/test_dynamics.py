#!/usr/bin/env python3
from dataclasses import dataclass

import gpflow
import numpy as np
from numpy.testing import assert_array_less
from geocontrol.dynamics import GPDynamics


rng: np.random.RandomState = np.random.RandomState(0)
input_dim = 1
output_dim = 1
num_inducing = 3
X = rng.randn(20, input_dim)
Y = rng.randn(20, output_dim) ** 2
Z = rng.randn(num_inducing, input_dim)
q_sqrt = (rng.randn(output_dim, num_inducing, num_inducing) ** 2) * 0.01
q_mu = rng.randn(num_inducing, output_dim)
likelihood = gpflow.likelihoods.Exponential()
kernel = gpflow.kernels.SquaredExponential()
mean_function = gpflow.mean_functions.Constant()
data = (X, Y)
num_latent_gps = 1


gp = gpflow.models.SVGP(
    kernel=kernel,
    likelihood=likelihood,
    q_diag=False,
    num_latent_gps=num_latent_gps,
    inducing_variable=Z,
    q_mu=q_mu,
    q_sqrt=q_sqrt,
    whiten=False,
)


def test_gp_dynamics_():
    dynamics = GPDynamics(gp)
    print(dynamics)
    assert dynamics == 1
