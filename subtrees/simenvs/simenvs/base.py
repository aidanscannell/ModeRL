#!/usr/bin/env python3
import cv2
import numpy as np
import scipy as sp
from tf_agents.environments import (
    py_environment,
    tf_environment,
    tf_py_environment,
    utils,
)
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


float_type = np.float64

# state constraints (domain)
MIN_STATE = -3.0
MAX_STATE = 3.0

# environment parameters
LOW_PROCESS_NOISE_VAR = np.array([0.000001, 0.000002])
HIGH_PROCESS_NOISE_VAR = np.array([0.0001, 0.00004])
LOW_PROCESS_NOISE_MEAN = np.array([0.00001, 0.000002])
HIGH_PROCESS_NOISE_MEAN = np.array([0.1, 0.18])
BITMAP_RESOLUTION = 600  # if gating_bitmap=None then use np.ones(600)
GATING_BITMAP = None

# simulation parameters
DELTA_TIME = 0.001


class BaseGatingEnv(py_environment.PyEnvironment):
    def __init__(
        self,
        num_states,
        num_actions,
        min_observation,
        max_observation,
        min_action,
        max_action,
        low_process_noise_mean=LOW_PROCESS_NOISE_MEAN,
        high_process_noise_mean=HIGH_PROCESS_NOISE_MEAN,
        low_process_noise_var=LOW_PROCESS_NOISE_VAR,
        high_process_noise_var=HIGH_PROCESS_NOISE_VAR,
        gating_bitmap=None,
        constant_error=0.0,
        delta_time=DELTA_TIME,
    ):
        # simulation parameters
        self.num_dims = 2
        self.num_states = num_states
        self.num_actions = num_actions
        self.state_init = np.zeros([num_states], dtype=float_type)
        self._state = self.state_init
        self.delta_time = delta_time
        self.constant_error = constant_error

        # environment parameters
        if isinstance(low_process_noise_var, np.ndarray):
            self.low_process_noise_var = low_process_noise_var
        else:
            print("low_process_noise_var isn't array so broadcasting")
            self.low_process_noise_var = low_process_noise_var * np.ones(self.num_dims)
        if isinstance(high_process_noise_var, np.ndarray):
            self.high_process_noise_var = high_process_noise_var
        else:
            print("high_process_noise_var isn't array so broadcasting")
            self.high_process_noise_var = high_process_noise_var * np.ones(
                self.num_dims
            )
        if isinstance(low_process_noise_mean, np.ndarray):
            self.low_process_noise_mean = low_process_noise_mean
        else:
            print("low_process_noise_mean isn't array so broadcasting")
            self.low_process_noise_mean = low_process_noise_mean * np.ones(
                self.num_dims
            )
        if isinstance(high_process_noise_mean, np.ndarray):
            self.high_process_noise_mean = high_process_noise_mean
        else:
            print("high_process_noise_mean isn't array so broadcasting")
            self.high_process_noise_mean = high_process_noise_mean * np.ones(
                self.num_dims
            )

        # configure action spec
        if not isinstance(min_action, np.ndarray):
            min_action = min_action * np.ones(num_actions)
            print("min_action isn't array so broadcasting")
        if not isinstance(max_action, np.ndarray):
            max_action = max_action * np.ones(num_actions)
            print("max_action isn't array so broadcasting")
        self._action_spec = array_spec.BoundedArraySpec(
            # shape=(1, num_actions),
            shape=(num_actions,),
            dtype=float_type,
            minimum=min_action,
            maximum=max_action,
            name="action",
        )
        # configure observation spec
        if not isinstance(min_observation, np.ndarray):
            min_observation = min_observation * np.ones(num_states)
            print("min_observation isn't array so broadcasting")
        if not isinstance(max_observation, np.ndarray):
            max_observation = max_observation * np.ones(num_states)
            print("max_observation isn't array so broadcasting")
        self._observation_spec = array_spec.BoundedArraySpec(
            # shape=(1, num_states),
            shape=(num_states,),
            dtype=float_type,
            minimum=min_observation,
            maximum=max_observation,
            name="observation",
        )
        self.episode_ended = False

        if gating_bitmap is None:
            resolution = BITMAP_RESOLUTION
            self.gating_bitmap = np.ones([resolution, resolution])
        elif isinstance(gating_bitmap, str):
            self.gating_bitmap = cv2.imread(gating_bitmap, cv2.IMREAD_GRAYSCALE)
            self.gating_bitmap = self.gating_bitmap / 255
        elif isinstance(gating_bitmap, np.ndarray):
            self.gating_bitmap = gating_bitmap
        else:
            raise ("gating_bitmap must be np.ndarray or filepath string for bitmap")
        # TODO check x and y are the right way around
        self.num_pixels = np.array(
            # [self.gating_bitmap.shape[0] - 1, self.gating_bitmap.shape[1] - 1]
            [self.gating_bitmap.shape[1] - 1, self.gating_bitmap.shape[0] - 1]
        )

    def state_to_pixel(self, state):
        """Returns the bitmap pixel index associated with state"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        pixel = (
            (
                state[0, 0 : self.num_dims]
                - self.observation_spec().minimum[0 : self.num_dims]
            )
            / (
                self.observation_spec().maximum[0 : self.num_dims]
                - self.observation_spec().minimum[0 : self.num_dims]
            )
            * self.num_pixels
        )
        pixel = np.array([-pixel[1], pixel[0]])
        pixel = np.rint(pixel).astype(int)
        # TODO check these termination conditions are right
        if pixel[0] < -self.gating_bitmap.shape[0]:
            self._episode_ended = True
            return np.array([0, 0])
        if pixel[0] > 0:
            self._episode_ended = True
            return np.array([0, 0])
        if pixel[1] > self.gating_bitmap.shape[1] - 1:
            self._episode_ended = True
            return np.array([0, 0])
        if pixel[1] < 0:
            self._episode_ended = True
            return np.array([0, 0])
        return pixel

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reset(self, state_init=None):
        """Return initial_time_step."""
        self._current_time_step = self._reset(state_init)
        return self._current_time_step

    def _reset(self, state=None):
        # print("Reseting environment")
        if state is None:
            self._state = self.state_init
        else:
            self._state = state
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=float_type))

    def state_to_mixing_prob(self, state):
        pixel = self.state_to_pixel(state)
        gating_value = self.gating_bitmap[pixel[0], pixel[1]]
        return gating_value

    def _process_noise(self, state):
        # mean = np.array([0.0, 0.0])
        mean = np.zeros(state.shape).reshape(-1)
        pixel = self.state_to_pixel(state)
        gating_value = self.gating_bitmap[pixel[0], pixel[1]]
        if gating_value == 1.0:
            var = self.low_process_noise_var
        else:
            var = self.high_process_noise_var
        cov = np.diag(var)
        noise = sp.random.multivariate_normal(mean, cov)
        if gating_value == 1.0:
            noise += self.low_process_noise_mean
        else:
            noise += self.high_process_noise_mean
        return noise
