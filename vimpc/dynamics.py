#!/usr/bin/env python3
import gpflow
import abc
import typing
from typing import Callable

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.conditionals import uncertain_conditional
from gpflow.models import GPModel
from tensor_annotations import axes
from tensor_annotations.axes import Batch

tfd = tfp.distributions

StateDim = typing.NewType("StateDim", axes.Axis)
ControlDim = typing.NewType("ControlDim", axes.Axis)


class Dynamics(abc.ABC):
    """Dynamics model for discrete system."""

    # @property
    # @abc.abstractmethod
    # def time_step(self):
    #     """Time step."""
    #     raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self,
        state: ttf.Tensor2[Batch, StateDim],
        control: ttf.Tensor2[Batch, ControlDim],
    ):
        """Transition dynamics function f(x, u)"""
        raise NotImplementedError


class BayesianDynamics(Dynamics):
    """Dynamics model for discrete systems learned with Bayesian inference."""

    def __init__(self, x_train, y_train):
        self.x_train, self.y_train = x_train, y_train

    def update(self, x_new, y_new):
        self.x_train = tf.concat([x_train, x_new])
        self.y_train = tf.concat([y_train, y_new])
        training_loss = model.training_loss_closure(iter(train_dataset))

    def sample(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, StateDim],
        num_samples: int = 1,
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ):
        dist = self(state_mean, control_mean, state_var, control_var)
        samples = dist.sample(num_samples)
        return samples

    @abc.abstractmethod
    def __call__(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ) -> tfd.Distribution:
        """Transition dynamics function f(x, u) with uncertain inputs
        x ~ N(state, state_var) and u ~ N(control , control_var)
        """
        raise NotImplementedError


import copy


class EnvDynamics(Dynamics):
    """Dynamics model for discrete systems learned with Bayesian inference."""

    def __init__(self, env, gp):
        self.env = env
        self.gp = gp

    def reset(self):
        self.env.reset()

    def __call__(
        self,
        control_mean: ttf.Tensor2[Batch, ControlDim],
    ) -> tfd.Distribution:
        """Transition dynamics function f(x, u) with uncertain inputs
        x ~ N(state, state_var) and u ~ N(control , control_var)
        """
        num_controls = control_mean.shape[0]
        tf.reshape(control_mean, [-1])
        next_time_step = self.env.step(control_mean)
        # state = next_time_step.observation
        state = next_time_step
        return tf.reshape(state, [num_controls, -1])
        # states = []
        # for sample in range(state_mean.shape[0]):
        #     print("hello aidan")
        #     print(state_mean)
        #     self.env._reset(state_mean[sample, :])
        #     print("after reset")
        #     next_time_step = self.env.step(control_mean[sample, :])
        #     state = next_time_step.observation
        #     states.append(state)
        # return tf.concat(states, 0)


class FakeEnv:
    """Dynamics model for discrete systems learned with Bayesian inference."""

    def __init__(self, start_state):
        self.start_state = copy.deepcopy(start_state)
        self.state = copy.deepcopy(start_state)
        self.previous_velocity = 0.0
        self.delta_time = 0.05

    def reset(self):
        self.state = self.start_state
        self.previous_velocity = 0.0

    def step(
        self,
        control_mean: ttf.Tensor2[Batch, ControlDim],
    ) -> tfd.Distribution:
        """Transition dynamics function f(x, u) with uncertain inputs
        x ~ N(state, state_var) and u ~ N(control , control_var)
        """
        delta_state = self.transition_dynamics(self.state, control_mean)
        self.state = delta_state + self.state
        return self.state

    def transition_dynamics(self, state, action):
        # simulator parameters
        velocity = action  # as veloctiy controlled

        delta_state = (
            0.5 * (self.previous_velocity + velocity) * self.delta_time
        )  # internal dynamics (suvat)

        self.previous_velocity = velocity
        return delta_state


class GPDynamics(BayesianDynamics):
    # class GPDynamics:
    def __init__(self, gp: GPModel):
        x_train = None
        y_train = None
        super().__init__(x_train, y_train)
        self.gp = gp

    def __call__(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ) -> ttf.Tensor2[Batch, StateDim]:
        assert len(state_mean.shape) == 2
        assert len(control_mean.shape) == 2
        input_mean = tf.concat([state_mean, control_mean], -1)
        if state_var is None and control_var is None:
            # dynamics_input = tf.reshape(
            #     tf.concat([state_mean, control_mean], -1), [1, -1]
            # )
            # delta_state_mean, delta_state_var = self.gp.predict_y(
            delta_state_mean, delta_state_var = self.gp.predict_f(
                input_mean, full_cov=False
            )
            state_var = 0.0
        else:
            input_var = tf.concat([state_var, control_var], -1)
            means, vars = [], []
            for i, (inducing_points, kernel) in enumerate(
                zip(
                    self.gp.inducing_variable.inducing_variables, self.gp.kernel.kernels
                )
            ):
                if len(self.gp.q_sqrt.shape) == 2:
                    q_sqrt = self.gp.q_sqrt[:, i : i + 1]
                elif len(self.gp.q_sqrt.shape) == 3:
                    q_sqrt = self.gp.q_sqrt[i : i + 1, :, :]
                # print("q_sqrt")
                # print(q_sqrt.shape)
                # print("kernel")
                # print(self.gp.kernel)
                # print(self.gp.kernel.kernels[0])
                delta_state_mean, delta_state_var = uncertain_conditional(
                    input_mean,
                    input_var,
                    # self.gp.inducing_variable.inducing_variables,
                    inducing_points,
                    # kernel=self.gp.kernel,
                    kernel=kernel,
                    q_mu=self.gp.q_mu[:, i : i + 1],
                    q_sqrt=q_sqrt,
                    mean_function=None,
                    # mean_function=gpflow.mean_functions.Zero(),
                    # mean_function=self.gp.mean_function,
                    full_output_cov=False,
                    full_cov=False,
                    white=self.gp.whiten,
                )
                delta_state_mean += control_mean * 0.05
                means.append(delta_state_mean)
                vars.append(delta_state_var)
                # print("delta_state_mean")
                # print(delta_state_mean.shape)
                # print(delta_state_var.shape)
            delta_state_mean = tf.stack(means, 0)[0, :, :]
            delta_state_var = tf.stack(vars, 0)[0, :, :]
        next_state_mean = state_mean + delta_state_mean
        next_state_var = state_var + delta_state_var
        return next_state_mean, next_state_var

    def sample(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        num_samples: int = 1,
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ) -> ttf.Tensor2[Batch, StateDim]:
        state_dim = state_mean.shape[-1]
        control_dim = control_mean.shape[-1]
        if state_var is None and control_var is None:
            dynamics_input = tf.concat([state_mean, control_mean], -1)
            # state_mean = tf.transpose(state_mean)
            # control_mean = tf.transpose(control_mean)
            # xx, yy = tf.meshgrid(state_mean, control_mean)
            # print("xx")
            # print(xx.shape)
            # print(yy.shape)
            # dynamics_input = tf.concat(
            #     [tf.reshape(xx, [-1, state_dim]), tf.reshape(yy, [-1, control_dim])], -1
            # )

            # if len(dynamics_input.shape) == 1:
            #     dynamics_input = tf.reshape(dynamics_input, [1, -1])
            print("dynamics_input")
            print(dynamics_input.shape)
            delta_state_samples = self.gp.predict_f_samples(
                dynamics_input, num_samples=num_samples, full_cov=False
            )  # [S, N, D]
            return delta_state_samples
