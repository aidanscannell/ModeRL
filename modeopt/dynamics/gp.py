#!/usr/bin/env python3
import typing
from functools import partial

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import posteriors
from gpflow.conditionals import uncertain_conditional
from gpflow.models import SVGP
from modeopt.dynamics import Dynamics
from modeopt.utils import combine_state_contols_to_input
from tensor_annotations import axes
from tensor_annotations.axes import Batch

tfd = tfp.distributions

StateDim = typing.NewType("StateDim", axes.Axis)
ControlDim = typing.NewType("ControlDim", axes.Axis)
StateControlDim = typing.NewType("StateControlDim", axes.Axis)


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


def gp_predict_dynamics_wrapper(
    delta_state_mean: ttf.Tensor2[Batch, StateDim],
    delta_state_var: ttf.Tensor2[Batch, StateDim],
    state_mean: ttf.Tensor2[Batch, StateDim] = None,
    state_var: ttf.Tensor2[Batch, StateDim] = None,
    predict_state_difference: bool = False,
    gp_dims: ttf.Tensor1[StateDim] = None,
) -> ttf.Tensor2[Batch, StateDim]:
    if gp_dims is not None:
        # gp_delta_state_mean = tf.expand_dims(gp_delta_state_mean, gp_dims)
        raise ("implement functionality of gp only modelling certain state dimensions")

    if predict_state_difference:
        return delta_state_mean, delta_state_var
    else:
        next_state_mean = state_mean + delta_state_mean
        if state_var is None:
            next_state_var = delta_state_var
        else:
            next_state_var = state_var + delta_state_var
        return next_state_mean, next_state_var


class SVGPDynamics(Dynamics):
    def __init__(self, svgp: SVGP, gp_dims=None):
        super().__init__()
        self.svgp = svgp
        self.gp = svgp
        self.gp_dims = gp_dims
        self.state_dim = self.svgp.num_latent_gps
        self.svgp_posterior = svgp.posterior(
            precompute_cache=posteriors.PrecomputeCacheType.TENSOR
        )

        def uncertain_predict_f(input_mean, input_var):
            # TODO use multidispatch to handle multi-output uncertain conditionals
            f_mean, f_var = multioutput_uncertain_conditional(
                input_mean,
                input_var,
                # inducing_variables=svgp.inducing_variable,
                inducing_variables=svgp.inducing_variable.inducing_variables[0],
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
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
        predict_state_difference: bool = False,
        add_noise: bool = False,
    ) -> ttf.Tensor2[Batch, StateDim]:
        input_mean, input_var = combine_state_contols_to_input(
            state_mean, control_mean, state_var=state_var, control_var=state_var
        )
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
        next_state_mean, next_state_var = gp_predict_dynamics_wrapper(
            delta_state_mean,
            delta_state_var,
            state_mean=state_mean,
            state_var=state_var,
            predict_state_difference=predict_state_difference,
            gp_dims=self.gp_dims,
        )
        return next_state_mean, next_state_var
