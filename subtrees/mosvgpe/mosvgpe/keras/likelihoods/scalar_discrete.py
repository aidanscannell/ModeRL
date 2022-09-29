#!/usr/bin/env python3
from gpflow.likelihoods import Bernoulli
import tensorflow as tf


class BernoulliSerializable(Bernoulli, tf.keras.layers.Layer):
    # class BernoulliSerializable(Bernoulli):
    def get_config(self) -> dict:
        return {}

    # @classmethod
    # def from_config(cls, cfg: dict):
    #     return cls()
