#!/usr/bin/env python3
import abc
import typing
from typing import Callable, Tuple

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.conditionals import uncertain_conditional
from gpflow.models import SVGP
from tensor_annotations import axes
from tensor_annotations.axes import Batch

tfd = tfp.distributions

StateDim = typing.NewType("StateDim", axes.Axis)
ControlDim = typing.NewType("ControlDim", axes.Axis)


class Dynamics(abc.ABC):
    """Dynamics model for discrete system."""

    @abc.abstractmethod
    def __call__(
        self,
        state: ttf.Tensor2[Batch, StateDim],
        control: ttf.Tensor2[Batch, ControlDim],
    ):
        """Transition dynamics function f(x, u)"""
        raise NotImplementedError


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
    means, vars = [], []
    for i, (kernel) in enumerate(kernel.kernels):
        if len(q_sqrt.shape) == 2:
            q_sqrt = q_sqrt[:, i : i + 1]
        elif len(q_sqrt.shape) == 3:
            q_sqrt = q_sqrt[i : i + 1, :, :]
        mean, var = uncertain_conditional(
            input_mean,
            input_var,
            inducing_variables,
            # gp.inducing_variable.inducing_variables[0],
            kernel=kernel,
            q_mu=q_mu[:, i : i + 1],
            q_sqrt=q_sqrt,
            mean_function=mean_function,
            full_output_cov=full_output_cov,
            full_cov=full_cov,
            white=whiten,
        )
        print("after uncertai")
        print(mean.shape)
        print(var.shape)
        means.append(mean)
        vars.append(var)
    if len(means) > 1:
        means = tf.concat(means, -1)
        vars = tf.concat(vars, -1)
    else:
        means = tf.constant(means)
        vars = tf.constant(vars)
    print("means")
    print(means.shape)
    print(vars.shape)
    return means, vars


def svgp_dynamics(
    svgp,
    state_mean: ttf.Tensor2[Batch, StateDim],
    control_mean: ttf.Tensor2[Batch, ControlDim],
    state_var: ttf.Tensor2[Batch, StateDim] = None,
    control_var: ttf.Tensor2[Batch, ControlDim] = None,
) -> ttf.Tensor2[Batch, StateDim]:
    assert len(state_mean.shape) == 2
    assert len(control_mean.shape) == 2
    input_mean = tf.concat([state_mean, control_mean], -1)
    if state_var is None and control_var is None:
        delta_state_mean, delta_state_var = svgp.predict_f(input_mean, full_cov=False)
    else:
        input_var = tf.concat([state_var, control_var], -1)
        delta_state_mean, delta_state_var = multioutput_uncertain_conditional(
            input_mean,
            input_var,
            # svgp.inducing_variable.inducing_variables,
            svgp.inducing_variable.inducing_variables[0],
            kernel=svgp.kernel,
            q_mu=svgp.q_mu,
            q_sqrt=svgp.q_sqrt,
            # mean_function=svgp.mean_function,
            mean_function=None,
            full_output_cov=False,
            full_cov=False,
            whiten=svgp.whiten,
        )
        delta_state_mean += svgp.mean_function(input_mean)
    return delta_state_mean, delta_state_var


def hybrid_svgp_dynamics(
    svgp,
    nominal_dynamics,
    gp_dims,
    state_mean: ttf.Tensor2[Batch, StateDim],
    control_mean: ttf.Tensor2[Batch, ControlDim],
    state_var: ttf.Tensor2[Batch, StateDim] = None,
    control_var: ttf.Tensor2[Batch, ControlDim] = None,
) -> Tuple[ttf.Tensor2[Batch, StateDim], ttf.Tensor2[Batch, StateDim]]:
    gp_delta_state_mean, gp_delta_state_var = svgp_dynamics(
        svgp,
        state_mean=state_mean,
        control_mean=control_mean,
        state_var=state_var,
        control_var=control_var,
    )
    # TODO propogate state-control uncertianty through nominal dynamics
    nominal_delta_state_mean, nominal_delta_state_var = nominal_dynamics(
        state_mean=state_mean,
        control_mean=control_mean,
        state_var=state_var,
        control_var=control_var,
    )

    if gp_dims is None:
        next_state_mean = state_mean + gp_delta_state_mean + nominal_delta_state_mean
        next_state_var = state_var + gp_delta_state_var + nominal_delta_state_var
    else:
        raise ("implement functionality of gp only modelling certain state dimensions")
        # gp_delta_state_mean = tf.expand_dims(gp_delta_state_mean, gp_dims)
    return next_state_mean, next_state_var


def zero_dynamics(
    state_mean: ttf.Tensor2[Batch, StateDim],
    control_mean: ttf.Tensor2[Batch, ControlDim],
    state_var: ttf.Tensor2[Batch, StateDim] = None,
    control_var: ttf.Tensor2[Batch, ControlDim] = None,
):
    return 0.0, 0.0


class SVGPDynamics(Dynamics):
    def __init__(
        self,
        svgp: SVGP,
        nominal_dynamics: Callable = None,
        gp_dims=None,
        # delta_time=0.05,
    ):
        super().__init(self)
        self.svgp = svgp
        self.gp = svgp
        self.gp_dims = gp_dims
        # self.delta_time = delta_time
        if nominal_dynamics is None:
            self._nominal_dynamics = zero_dynamics
        else:
            self._nominal_dynamics = nominal_dynamics

        self.state_dim = self.svgp.num_latent_gps

    def nominal_dynamics(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ):
        return self._nominal_dynamics(
            state_mean=state_mean,
            control_mean=control_mean,
            state_var=state_var,
            control_var=control_var,
        )

    def __call__(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ) -> ttf.Tensor2[Batch, StateDim]:
        return hybrid_svgp_dynamics(
            self.svgp,
            self.nominal_dynamics,
            self.gp_dims,
            state_mean=state_mean,
            control_mean=control_mean,
            state_var=state_var,
            control_var=control_var,
        )
