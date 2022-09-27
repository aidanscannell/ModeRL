#!/usr/bin/env python3
from gpflow.likelihoods import Gaussian
import tensorflow as tf


# class GaussianSerializable(Gaussian, tf.keras.layers.Layer):
class GaussianSerializable(Gaussian):
    def get_config(self) -> dict:
        return {
            "variance": self.variance.numpy(),
            # "variance_lower_bound": self.variance.transform.bijectors.low
            # TODO chnage this depending on bijector
        }

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(variance=cfg["variance"])
