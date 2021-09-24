#!/usr/bin/env python3
import abc
import typing

import tensor_annotations.tensorflow as ttf
import tensorflow_probability as tfp
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
