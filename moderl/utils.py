#!/usr/bin/env python3
import json
import os
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float


tfd = tfp.distributions


def combine_state_controls_to_input(
    state: Union[tfd.Normal, tfd.Deterministic],
    control: Union[tfd.Normal, tfd.Deterministic],
) -> Union[tfd.Deterministic, tfd.Normal]:
    assert len(state.mean().shape) == 2
    assert len(control.mean().shape) == 2
    input_mean = tf.concat([state.mean(), control.mean()], -1)
    if state.variance() is None and control.variance() is None:
        input_var = None
        # state_var = tf.zeros(control_var.shape, dtype=default_float())
        # control_var = tf.zeros(state_var.shape, dtype=default_float())
    else:
        # if state.variance() is None and control.variance() is not None:
        #     assert len(control.mean().shape) == 2
        #     state_var = tf.zeros(control.variance().shape, dtype=default_float())
        # elif state.variance() is not None and control.variance() is None:
        #     assert len(state.mean().shape) == 2
        #     control_var = tf.zeros(state.variance().shape, dtype=default_float())
        input_var = tf.concat([state.variance(), control.variance()], -1)

    if isinstance(state, tfd.Deterministic) and isinstance(control, tfd.Deterministic):
        return tfd.Deterministic(loc=input_mean)
    elif input_var is None:
        return tfd.Deterministic(loc=input_mean)
    else:
        return tfd.Normal(loc=input_mean, scale=tf.math.sqrt(input_var))
    # return input_mean, input_var


# def combine_state_controls_to_input(
#     state_mean: StateMean,
#     control_mean: ControlMean,
#     state_var: StateVariance = None,
#     control_var: ControlVariance = None,
# ) -> StateControlMeanAndVariance:
#     assert len(state_mean.shape) == 2
#     assert len(control_mean.shape) == 2
#     input_mean = tf.concat([state_mean, control_mean], -1)
#     if state_var is None and control_var is None:
#         input_var = None
#         # state_var = tf.zeros(control_var.shape, dtype=default_float())
#         # control_var = tf.zeros(state_var.shape, dtype=default_float())
#     else:
#         if state_var is None and control_var is not None:
#             assert len(control_mean.shape) == 2
#             state_var = tf.zeros(control_var.shape, dtype=default_float())
#         elif state_var is not None and control_var is None:
#             assert len(state_mean.shape) == 2
#             control_var = tf.zeros(state_var.shape, dtype=default_float())
#         input_var = tf.concat([state_var, control_var], -1)
#     return input_mean, input_var


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


def save_json_config(obj, filename: str = "config.json"):
    """Save object to .json using get_config()"""

    class NumpyArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cfg = tf.keras.utils.serialize_keras_object(obj)
    with open(filename, "w") as f:
        json.dump(cfg, f, cls=NumpyArrayEncoder)


# def load_from_json_config(filename: str, custom_objects: dict):
#     """Load object from .json using from_config()"""
#     with open(filename, "r") as read_file:
#         json_cfg = read_file.read()
#     return tf.keras.models.model_from_json(json_cfg, custom_objects=custom_objects)
