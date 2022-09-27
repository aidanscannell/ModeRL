#!/usr/bin/env python3
# import json
# from json import JSONEncoder

import numpy as np

# import tensorflow as tf
# import yaml


def try_array_except_none(cfg: dict, key: str):
    # np.array(cfg["q_mu"]) works for deserializing model using keras
    # and setting q_mu=None/not setting them, allows users to write custom configs
    # without specifying q_mu/q_sqrt in the config
    try:
        return np.array(cfg[key]) if cfg[key] is not None else None
    except KeyError:
        return None


def try_val_except_none(cfg: dict, key: str):
    try:
        return cfg[key]
    except KeyError:
        return None


# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)


# def save_json_config(obj, filename: str = "config.json"):
#     """Save object to .json using get_config()"""
#     cfg = tf.keras.utils.serialize_keras_object(obj)
#     with open(filename, "w") as f:
#         json.dump(cfg, f, cls=NumpyArrayEncoder)


# def load_from_json_config(filename: str, custom_objects: dict):
#     """Load object from .json using from_config()"""
#     with open(filename, "r") as read_file:
#         json_cfg = read_file.read()
#     return tf.keras.models.model_from_json(json_cfg, custom_objects=custom_objects)


# def model_from_yaml(yaml_cfg_filename: str, custom_objects: dict = None):
#     with open(yaml_cfg_filename, "r") as fp:
#         cfg = yaml.load(fp, Loader=yaml.SafeLoader)
#         model_cfg = json.dumps(cfg)
#     return tf.keras.models.model_from_json(model_cfg, custom_objects=custom_objects)
