#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from gpflow.inducing_variables import InducingPoints


# class InducingPointsSerializable(InducingPoints, tf.keras.layers.Layer):
class InducingPointsSerializable(InducingPoints):
    def get_config(self) -> dict:
        return {"Z": self.Z.numpy()}

    @classmethod
    def from_config(cls, cfg: dict):
        try:
            Z = np.array(cfg["Z"])
        except KeyError:
            num_inducing = cfg["num_inducing"]
            input_dim = cfg["input_dim"]
            Z = np.random.rand(num_inducing, input_dim)
        return cls(Z=Z)
