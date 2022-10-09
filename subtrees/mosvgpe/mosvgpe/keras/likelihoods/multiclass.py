#!/usr/bin/env python3
import tensorflow as tf
from gpflow.likelihoods import Softmax


class SoftmaxSerializable(Softmax, tf.keras.layers.Layer):
    # class SoftmaxSerializable(Softmax):
    def get_config(self) -> dict:
        return {}

    # @classmethod
    # def from_config(cls, cfg: dict):
    #     return cls()
