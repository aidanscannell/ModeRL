#!/usr/bin/env python3
import json

import tensorflow as tf
from omegaconf import DictConfig, OmegaConf


def model_from_DictConfig(cfg: DictConfig, custom_objects: dict = None):
    cfg_dict = OmegaConf.to_container(cfg)
    layer_cfg = json.dumps(cfg_dict)
    return tf.keras.models.model_from_json(layer_cfg, custom_objects=custom_objects)


def deserialize(cfg, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        identifier=cfg,
        module_objects=globals(),
        custom_objects=custom_objects,
        name="MyObjectType",
    )
