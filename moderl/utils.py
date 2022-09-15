#!/usr/bin/env python3
import typing
from typing import Union

import tensorflow as tf
from gpflow import default_float
from modeopt.custom_types import (
    ControlMean,
    ControlVariance,
    StateControlMeanAndVariance,
    StateMean,
    StateVariance,
)
from tensor_annotations import axes

StateDim = typing.NewType("StateDim", axes.Axis)
ControlDim = typing.NewType("ControlDim", axes.Axis)


def combine_state_controls_to_input(
    state_mean: StateMean,
    control_mean: ControlMean,
    state_var: StateVariance = None,
    control_var: ControlVariance = None,
) -> StateControlMeanAndVariance:
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


def append_zero_control(control_means, control_vars):
    # Append zeros to control trajectory
    control_means = tf.concat(
        [
            control_means,
            tf.zeros([1, tf.shape(control_means)[1]], dtype=default_float()),
        ],
        0,
    )
    if control_vars is not None:
        control_vars = tf.concat(
            [
                control_vars,
                tf.zeros([1, tf.shape(control_vars)[1]], dtype=default_float()),
            ],
            0,
        )
    return control_means, control_vars


def weight_to_matrix(value: Union[list, float], dim: int):
    if isinstance(value, list):
        if len(value) == dim:
            value = tf.constant(value, dtype=default_float())
            return tf.linalg.diag(value)
        else:
            raise NotImplementedError
    elif value is None or value == 0.0:
        return None
    else:
        return tf.eye(dim, dtype=default_float()) * value
