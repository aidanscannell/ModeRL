#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from simenvs.base import BaseGatingEnv
from tf_agents.environments import tf_py_environment, utils
from tf_agents.trajectories import time_step as ts


float_type = np.float64

# control constraints
MIN_THRUST = -2
MAX_THRUST = 2
MIN_TORQUE = -1
MAX_TORQUE = 1
MIN_ACTION = np.array([MIN_THRUST, MIN_TORQUE])
MAX_ACTION = np.array([MAX_THRUST, MAX_TORQUE])

# state constraints (domain)
# MIN_STATE = np.array([-3.0, -3.0, -50.0, -50.0, -10.0])
# MAX_STATE = np.array([3.0, 3.0, 50.0, 50.0, 10.0])
MIN_STATE = np.array([-3.0, -3.0, -10.0])
MAX_STATE = np.array([3.0, 3.0, 10.0])

# environment parameters
LOW_PROCESS_NOISE_VAR = np.array([0.000001, 0.000002])
HIGH_PROCESS_NOISE_VAR = np.array([0.0001, 0.00004])
LOW_PROCESS_NOISE_MEAN = np.array([0.00001, 0.000002])
HIGH_PROCESS_NOISE_MEAN = np.array([0.1, 0.18])
BITMAP_RESOLUTION = 600  # if gating_bitmap=None then use np.ones(600)
GATING_BITMAP = None

# simulation parameters
DELTA_TIME = 0.001


def deg2rad(angle):
    return angle * np.math.pi / 180.0


class RotatingPointMass2DEnv(BaseGatingEnv):
    def __init__(
        self,
        mass=1.4,
        moment_of_inertia=2.0,
        min_observation=MIN_STATE,
        max_observation=MAX_STATE,
        min_action=MIN_ACTION,
        max_action=MAX_ACTION,
        low_process_noise_var=LOW_PROCESS_NOISE_VAR,
        high_process_noise_var=HIGH_PROCESS_NOISE_VAR,
        low_process_noise_mean=LOW_PROCESS_NOISE_MEAN,
        high_process_noise_mean=HIGH_PROCESS_NOISE_MEAN,
        gating_bitmap=None,
        constant_error=0.0,
        delta_time=DELTA_TIME,
    ):
        # num_states = 5  # [x, y \dot{x}, \dot{y}, \phi]
        num_states = 3  # [x, y, \phi]
        num_actions = 2  # [thrust, torque]
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            min_observation=min_observation,
            max_observation=max_observation,
            min_action=min_action,
            max_action=max_action,
            low_process_noise_var=low_process_noise_var,
            high_process_noise_var=high_process_noise_var,
            low_process_noise_mean=low_process_noise_mean,
            high_process_noise_mean=high_process_noise_mean,
            gating_bitmap=gating_bitmap,
            constant_error=constant_error,
            delta_time=delta_time,
        )
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.inv_mass = 1 / mass
        self.inv_moment_of_inertia = 1 / moment_of_inertia

    def _step(self, action):
        delta_state = self.transition_dynamics(self._state, action)
        # print("delta state")
        # print(delta_state)
        # print(self._state)
        self._state += delta_state
        reward = 0
        # self._episode_ended = True  # remove this when term conds added
        if np.any(self._state > self.observation_spec().maximum):
            self._episode_ended = True
        elif np.any(self._state < self.observation_spec().minimum):
            self._episode_ended = True
        if self._episode_ended:
            # return ts.termination(np.array([self._state], dtype=float_type), reward)
            return ts.termination(self._state, reward)
        else:
            return ts.transition(
                self._state,
                # np.array([self._state], dtype=float_type),
                reward=reward,
                discount=1.0,
            )

    def transition_dynamics(self, state, action):
        # print("state")
        # print(state)
        # print(action)
        if len(action.shape) == 1:
            action = action.reshape(1, -1)
        # velocity = state[:, 2:4]
        yaw = state[:, -1:]

        # yaw = tf.math.atan2(velocity[:, 0], velocity[:, 1])

        thrust = action[:, 0:1]
        torque = action[:, 1:2]

        # velocity_x = -np.sin(yaw) * velocity[:, 0] + np.cos(yaw) * velocity[:, 1]
        # velocity_y = np.cos(yaw) * velocity[:, 0] + np.sin(yaw) * velocity[:, 1]
        # print("acceleration_x")
        # print(acceleration)

        # wind/turbulence blows quadcopter in world frame
        process_noise = self._process_noise(state) * 1  # external dynamics
        # acceleration += process_noise
        # print(acceleration)
        # print(acceleration)

        print("process_noise")
        print(process_noise)
        print("thrust")
        print(thrust)

        # accelerations in world frame
        acceleration_x = self.inv_mass * np.cos(yaw) * thrust + process_noise[0]
        acceleration_y = self.inv_mass * np.sin(yaw) * thrust + process_noise[1]
        acceleration = np.concatenate([acceleration_x, acceleration_y], -1)
        # print("acceleration")
        # print(acceleration)
        # drag = - @ velocity

        # velocity_noise = process_noise * self.delta_time * 10
        # velocity_noise = process_noise
        velocity = acceleration * self.delta_time
        # print("velocity_noise")
        # print(velocity_noise)
        # print("velocity")
        # print(velocity)
        # velocity += velocity_noise
        # print(velocity)

        angular_velocity = self.inv_moment_of_inertia * torque
        print("angular_velocity")
        print(angular_velocity)
        print(process_noise[2])
        angular_velocity += process_noise[2]

        # delta_state = (
        #     np.concatenate([velocity, acceleration], -1) + self.constant_error
        # ) * self.delta_time
        # delta_state = (
        #     np.concatenate([velocity, acceleration, angular_velocity], -1)
        #     + self.constant_error
        # ) * self.delta_time
        delta_state = (
            np.concatenate([velocity, angular_velocity], -1) + self.constant_error
        ) * self.delta_time
        # return delta_state
        print("delta_state.shape")
        print(delta_state.shape)
        return delta_state.reshape(-1)
