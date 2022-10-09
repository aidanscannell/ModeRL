#!/usr/bin/env python3
import tensorflow as tf
from gpflow.kernels import SeparateIndependent, SharedIndependent

from .single_output_kernel_objects import SINGLE_OUTPUT_KERNEL_OBJECTS


class SeparateIndependentSerializable(SeparateIndependent, tf.keras.layers.Layer):
    def get_config(self) -> dict:
        kernels = []
        for kernel in self.kernels:
            kernels.append(tf.keras.layers.serialize(kernel))
        return {"kernels": kernels}

    @classmethod
    def from_config(cls, cfg: dict):
        kernels = []
        for kernel_cfg in cfg["kernels"]:
            kernels.append(
                tf.keras.layers.deserialize(
                    kernel_cfg, custom_objects=SINGLE_OUTPUT_KERNEL_OBJECTS
                )
            )
        return cls(kernels)


class SharedIndependentSerializable(SharedIndependent, tf.keras.layers.Layer):
    def get_config(self) -> dict:
        return {
            "kernels": tf.keras.layers.serialize(self.kernel),
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_config(cls, cfg: dict):
        kernel = tf.keras.layers.deserialize(
            cfg["kernels"], custom_objects=SINGLE_OUTPUT_KERNEL_OBJECTS
        )
        return cls(kernel=kernel, output_dim=cfg["output_dim"])
