#!/usr/bin/env python3
import typing
from typing import Tuple

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import default_float
from tensor_annotations import axes
from tensor_annotations.axes import Batch

StateDim = typing.NewType("StateDim", axes.Axis)
ControlDim = typing.NewType("ControlDim", axes.Axis)


def combine_state_contols_to_input(
    state_mean: ttf.Tensor2[Batch, StateDim],
    control_mean: ttf.Tensor2[Batch, ControlDim],
    state_var: ttf.Tensor2[Batch, StateDim] = None,
    control_var: ttf.Tensor2[Batch, ControlDim] = None,
) -> Tuple[ttf.Tensor2[Batch, StateDim], ttf.Tensor2[Batch, StateDim]]:
    assert len(state_mean.shape) == 2
    assert len(control_mean.shape) == 2
    input_mean = tf.concat([state_mean, control_mean], -1)
    if state_var is None and control_var is None:
        input_var = None
        # state_var = tf.zeros(control_var.shape, dtype=default_float())
        # control_var = tf.zeros(state_var.shape, dtype=default_float())
    else:
        if state_var is None and control_var is not None:
            assert len(control_mean.shape) == 2
            state_var = tf.zeros(control_var.shape, dtype=default_float())
        elif state_var is not None and control_var is None:
            assert len(state_mean.shape) == 2
            control_var = tf.zeros(state_var.shape, dtype=default_float())
        input_var = tf.concat([state_var, control_var], -1)
    return input_mean, input_var
