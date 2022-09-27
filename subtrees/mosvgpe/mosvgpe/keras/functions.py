#!/usr/bin/env python3
from gpflow.functions import Zero, Constant
import tensorflow as tf


# class ConstantSerializable(Constant, tf.keras.layers.Layer):
class ConstantSerializable(Constant):
    def get_config(self) -> dict:
        return {"c": self.c.numpy()}

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(**cfg)


# class ZeroSerializable(Zero, tf.keras.layers.Layer):
class ZeroSerializable(Zero):
    def get_config(self) -> dict:
        return {"output_dim": self.output_dim}

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(**cfg)


MEAN_FUNCTIONS = [ConstantSerializable, ZeroSerializable]
MEAN_FUNCTION_OBJECTS = {
    mean_function.__name__: mean_function for mean_function in MEAN_FUNCTIONS
}
