#!/usr/bin/env python3
import abc
import json
import os
from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend

# from keras.saving.saved_model import json_utils
from moderl.custom_types import State

# from mogpe.keras.utils import save_json_config
from scipy.optimize import LinearConstraint, NonlinearConstraint

tfd = tfp.distributions

# JSON_CONFIG_FILENAME = "config.json"


class Controller(tf.Module, abc.ABC):
    # class Controller(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State = None, timestep: int = None):
        raise NotImplementedError

    @abc.abstractmethod
    def optimise(self):
        raise NotImplementedError

    def constraints(self) -> Union[LinearConstraint, NonlinearConstraint]:
        raise NotImplementedError

    def control_dim(self) -> int:
        raise NotImplementedError


class FeedbackController(Controller, abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State):
        raise NotImplementedError


class NonFeedbackController(Controller, abc.ABC):
    @abc.abstractmethod
    def __call__(self, timestep: int):
        raise NotImplementedError
