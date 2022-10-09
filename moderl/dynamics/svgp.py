#!/usr/bin/env python3
from functools import partial
from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp

# from gpflow import posteriors
from gpflow.conditionals import uncertain_conditional
from gpflow.models import SVGP
from moderl.utils import combine_state_controls_to_input


tfd = tfp.distributions


class SVGPDynamicsWrapper:
    # class SVGPDynamicsWrapper(tf.keras.layers.Layer):
    def __init__(self, svgp: SVGP):
        # self.svgp_posterior = svgp.posterior(
        #     precompute_cache=posteriors.PrecomputeCacheType.TENSOR
        # )
        # TODO make posterior work with hydra config
        # self.svgp_posterior = svgp

        def uncertain_predict_f(input_mean, input_var):
            # TODO use multidispatch to handle multi-output uncertain conditionals
            f_mean, f_var = multioutput_uncertain_conditional(
                input_mean,
                input_var,
                # inducing_variables=svgp.inducing_variable,
                inducing_variables=svgp.inducing_variable.inducing_variable,
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
        state: Union[tfd.Normal, tfd.Deterministic],
        control: Union[tfd.Normal, tfd.Deterministic],
        predict_state_difference: bool = False,
        add_noise: bool = False,
        # ) -> StateMeanAndVariance:
    ) -> tfd.Normal:
        # print("hello")
        # print(state)
        # print(control)
        state_control_input = combine_state_controls_to_input(
            state=state, control=control
        )
        # print("state_control_input")
        # print(state_control_input)
        # input_mean, input_var = combine_state_controls_to_input(
        #     state_mean, control_mean, state_var=state_var, control_var=control_var
        # )
        # print("input_mean.shape")
        # print(input_mean.shape)
        # print(input_var)
        # if input_var is None:

        # if isinstance(state_control_input, tfd.Deterministic):
        #     print("here 1")
        #     delta_state_mean, delta_state_var = self.predict_f(
        #         state_control_input.mean()
        #     )
        # elif isinstance(state_control_input, tfd.Normal):
        #     print("here 2")
        #     print("state_control_input.mean()")
        #     print(state_control_input.mean())
        #     print(state_control_input.variance())
        #     delta_state_mean, delta_state_var = self.uncertain_predict_f(
        #         state_control_input.mean(), state_control_input.variance()
        #     )
        # else:
        #     raise NotImplementedError
        # TODO uncomment uncertainty propagation using moment mathcing
        delta_state_mean, delta_state_var = self.predict_f(state_control_input.mean())

        # print("delta_state_mean")
        # print(delta_state_mean)
        # print(delta_state_var)
        # delta_state_mean, delta_state_var = self.predict_f(input_mean)
        if add_noise:
            delta_state_mean, delta_state_var = self.gp.likelihood.predict_mean_and_var(
                delta_state_mean, delta_state_var
            )

        if predict_state_difference:
            # return delta_state_mean, delta_state_var
            return tfd.Normal(loc=delta_state_mean, scale=tf.math.sqrt(delta_state_var))
        else:
            next_state_mean = state.mean() + delta_state_mean
            # if state_var is None:
            #     next_state_var = delta_state_var
            # else:
            #     next_state_var = state.variance() + delta_state_var
            next_state_var = state.variance() + delta_state_var
            return tfd.Normal(loc=next_state_mean, scale=tf.math.sqrt(next_state_var))


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
