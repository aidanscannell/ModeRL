#!/usr/bin/env python3
import tensorflow as tf
from gpflow.inducing_variables import (
    InducingPoints,
    SharedIndependentInducingVariables,
    SeparateIndependentInducingVariables,
)


class SharedIndependentInducingVariablesSerializable(
    SharedIndependentInducingVariables
):
    def get_config(self) -> dict:
        return {"inducing_variable": tf.keras.layers.serialize(self.inducing_variable)}

    @classmethod
    def from_config(cls, cfg: dict):
        inducing_variable = tf.keras.layers.deserialize(
            cfg["inducing_variable"],
            custom_objects={"InducingPoints": InducingPoints},
        )
        return cls(inducing_variable=inducing_variable)


class SeparateIndependentInducingVariablesSerializable(
    SeparateIndependentInducingVariables
):
    def get_config(self) -> dict:
        serialized_list = []
        for inducing_variable in self.inducing_variable_list:
            serialized_list.append(tf.keras.layers.serialize(inducing_variable))
        return {"inducing_variable_list": serialized_list}

    @classmethod
    def from_config(cls, cfg: dict):
        deserialized_list = []
        for inducing_variable in cfg["inducing_variable_list"]:
            deserialized_list.append(
                tf.keras.layers.deserialize(
                    inducing_variable,
                    custom_types={"InducingPoints": InducingPoints},
                )
            )
        return cls(inducing_variable_list=deserialized_list)
