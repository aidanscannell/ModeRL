#!/usr/bin/env python3
from gpflow.likelihoods import Softmax
import tensorflow as tf


# class SoftmaxSerializable(Softmax, tf.keras.layers.Layer):
class SoftmaxSerializable(Softmax):
    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, cfg: dict):
        return cls()
