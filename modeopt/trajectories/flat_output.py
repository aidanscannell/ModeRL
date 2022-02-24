#!/usr/bin/env python3
import abc
from typing import Optional, Union

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import default_float
from modeopt.custom_types import (
    ControlDim,
    ControlTrajectory,
    ControlTrajectoryMean,
    FlatOutputDim,
    Horizon,
    StateDim,
    Times,
)

from .base import BaseTrajectory

FlatOutput = ttf.Tensor2[Horizon, FlatOutputDim]


class FlatOutputTrajectory(BaseTrajectory):
    def __init__(
        self,
        flat_output: FlatOutput,
        times: Times,
        name: str = "FlatOutputTrajectory",
    ):
        self._flat_output = flat_output
        self.times = times
        super().__init__(name=name)

    def __call__(
        self, timestep: Optional[int], variance: Optional[bool] = False
    ) -> Union[ControlTrajectory, ControlTrajectoryMean]:
        if timestep is not None:
            idxs = [timestep, ...]
        else:
            idxs = [...]
        if variance:
            # TODO does it matter if this return zeros or None?
            return self.controls[idxs], None
        else:
            return self.controls[idxs]

    @property
    def flat_output_first_derivative(self):
        flat_output_plus_a_zero = tf.concat(
            [self.flat_output, tf.zeros([1, self.flat_dim], dtype=default_float())], 0
        )
        diff = flat_output_plus_a_zero[1:, :] - flat_output_plus_a_zero[:-1, :]
        return diff
        # diff = self.flat_output[1:, :] - self.flat_output[:-1, :]
        # return tf.concat([tf.zeros([1, self.flat_dim], dtype=default_float()), diff], 0)
        # return tf.concat([diff, tf.zeros([1, self.flat_dim], dtype=default_float())], 0)

    @property
    def flat_output(self):
        return self._flat_output

    @property
    def lagrange_multipliers(self):
        return tf.Variable(
            np.ones((self.horizon - 1, self.flat_dim)), dtype=default_float()
        )
        # return tf.Variable(np.ones(self.flat_output.shape), dtype=default_float())

    @property
    @abc.abstractmethod
    def controls(self) -> ttf.Tensor2[Horizon, ControlDim]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def states(self) -> ttf.Tensor2[Horizon, StateDim]:
        raise NotImplementedError

    @property
    def horizon(self) -> int:
        return self.flat_output.shape[0]

    @property
    def flat_dim(self):
        return self.flat_output.shape[-1]

    @property
    def state_dim(self):
        return self.states.shape[-1]


class VelocityControlledFlatOutputTrajectory(FlatOutputTrajectory):
    def __init__(
        self,
        start_state: ttf.Tensor1[FlatOutputDim],
        target_state: ttf.Tensor1[FlatOutputDim],
        horizon: int = None,
        t_init: float = -1.0,
        t_end: float = 1.0,
        lagrange_multipliers: Optional[
            bool
        ] = False,  # create lagrange multiplier variables?
        name: str = "FlatOutputTrajectory",
    ):
        times = np.linspace(t_init, t_end, horizon)
        times = tf.constant(times, dtype=default_float())
        super().__init__(flat_output=None, times=times, name=name)

        assert len(start_state.shape) == 2
        assert len(target_state.shape) == 2
        flat_output = tf.linspace(start_state[0, :], target_state[0, :], horizon)
        self.flat_output_vars = tf.Variable(flat_output[1:-1, :])
        self.start_state = start_state
        self.target_state = target_state

        # d_flat_output = tf.concat(
        #     [tf.concat([start_state, self.flat_output_vars], 0), target_state], 0
        # )
        # # tf.linspace(start_state[0, :], target_state[0, :], horizon)
        # self._d_flat_output_vars = tf.Variable(d_flat_output)

        # if lagrange_multipliers:
        #     self._lagrange_multipliers = tf.Variable(
        #         np.ones((self.horizon - 1, self.flat_dim * 2)), dtype=default_float()
        #     )
        # else:
        #     self._lagrange_multipliers = None

    @property
    def flat_output(self):
        return tf.concat(
            [
                tf.concat([self.start_state, self.flat_output_vars], 0),
                self.target_state,
            ],
            0,
        )

    @property
    def lagrange_multipliers(self):
        return self._lagrange_multipliers
        # return tf.Variable(np.ones(self.flat_output.shape), dtype=default_float())

    # @property
    # def flat_output_first_derivative(self):
    #     return self._d_flat_output_vars

    @property
    def controls(self) -> ttf.Tensor2[Horizon, ControlDim]:
        # flat_output_plus_a_zero = tf.concat(
        #     [self.flat_output, tf.zeros([1, self.flat_dim], dtype=default_float())], 0
        # )
        diff = self.states[1:, :] - self.states[:-1, :]
        return tf.concat([diff, tf.zeros([1, self.flat_dim], dtype=default_float())], 0)

    @property
    def states(self) -> ttf.Tensor2[Horizon, StateDim]:
        return self.flat_output

    @property
    def control_dim(self) -> int:
        return self.flat_dim

    def copy(self):
        return VelocityControlledFlatOutputTrajectory(
            start_state=tf.identity(self.flat_output)[0:1, :],
            target_state=tf.identity(self.flat_output)[-1:, :],
            horizon=self.horizon,
            t_init=self.times[0],
            t_end=self.times[-1],
        )
