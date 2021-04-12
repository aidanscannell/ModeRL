import numpy as np
import toml
from bunch import Bunch

from simenvs.quadcopter_sim import VelocityControlledQuadcopter2DEnv

# control constraints
MIN_VELOCITY = -10
MAX_VELOCITY = 10
MIN_ACCELERATION = -10
MAX_ACCELERATION = 10
MIN_ACTION = MIN_VELOCITY
MAX_ACTION = MAX_VELOCITY

# observation constraints
MIN_STATE = -3.0
MAX_STATE = 3.0

# environment parameters
LOW_PROCESS_NOISE_VAR = np.array([0.000001, 0.000002])
HIGH_PROCESS_NOISE_VAR = np.array([0.0001, 0.00004])
# BITMAP_RESOLUTION = 600
GATING_BITMAP = None

# simulation parameters
DELTA_TIME = 0.05
VELOCITY_INIT = 0.0


def parse_list_to_array(list_):
    return np.array(list_)


def parse_min_state(config):
    try:
        min_state = config.min_state
        if isinstance(min_state, list):
            min_state = parse_list_to_array(min_state)
    except AttributeError:
        min_state = MIN_STATE
        print("No min_state found in toml config so using default :", min_state)
    return min_state


def parse_max_state(config):
    try:
        max_state = config.max_state
        if isinstance(max_state, list):
            max_state = parse_list_to_array(max_state)
    except AttributeError:
        max_state = MAX_STATE
        print("No max_state found in toml config so using default :", max_state)
    return max_state


def parse_min_velocity(config):
    try:
        min_velocity = config.min_velocity
        if isinstance(min_velocity, list):
            min_velocity = parse_list_to_array(min_velocity)
    except AttributeError:
        min_velocity = MIN_VELOCITY
        print(
            "No min_velocity found in toml config so using default :",
            min_velocity,
        )
    return min_velocity


def parse_max_velocity(config):
    try:
        max_velocity = config.max_velocity
        if isinstance(max_velocity, list):
            max_velocity = parse_list_to_array(max_velocity)
    except AttributeError:
        max_velocity = MAX_VELOCITY
        print(
            "No max_velocity found in toml config so using default :",
            max_velocity,
        )
    return max_velocity


def parse_min_acceleration(config):
    try:
        min_acceleration = config.min_acceleration
        if isinstance(min_acceleration, list):
            min_acceleration = parse_list_to_array(min_acceleration)
    except AttributeError:
        min_acceleration = MIN_ACCELERATION
        print(
            "No min_acceleration found in toml config so using default :",
            min_acceleration,
        )
    return min_acceleration


def parse_max_acceleration(config):
    try:
        max_acceleration = config.max_acceleration
        if isinstance(max_acceleration, list):
            max_acceleration = parse_list_to_array(max_acceleration)
    except AttributeError:
        max_acceleration = MAX_ACCELERATION
        print(
            "No max_acceleration found in toml config so using default :",
            max_acceleration,
        )
    return max_acceleration


def parse_low_process_noise_var(config):
    try:
        low_process_noise_var = config.low_process_noise_var
        if isinstance(low_process_noise_var, list):
            low_process_noise_var = parse_list_to_array(low_process_noise_var)
    except AttributeError:
        low_process_noise_var = LOW_PROCESS_NOISE_VAR
        print(
            "No low_process_noise_var found in toml config so using default :",
            low_process_noise_var,
        )
    return low_process_noise_var


def parse_high_process_noise_var(config):
    try:
        high_process_noise_var = config.high_process_noise_var
        if isinstance(high_process_noise_var, list):
            high_process_noise_var = parse_list_to_array(high_process_noise_var)
    except AttributeError:
        high_process_noise_var = HIGH_PROCESS_NOISE_VAR
        print(
            "No high_process_noise_var found in toml config so using default :",
            high_process_noise_var,
        )
    return high_process_noise_var


def parse_delta_time(config):
    try:
        delta_time = config.delta_time
    except AttributeError:
        delta_time = DELTA_TIME
        print("No delta_time found in toml config so using default :", delta_time)
    return delta_time


def parse_velocity_init(config):
    try:
        velocity_init = config.velocity_init
        if isinstance(velocity_init, list):
            velocity_init = parse_list_to_array(velocity_init)
    except AttributeError:
        velocity_init = VELOCITY_INIT
        print(
            "No velocity_init found in toml config so using default :",
            velocity_init,
        )
    return velocity_init


def parse_gating_bitmap(config):
    try:
        gating_bitmap = config.gating_bitmap
    except AttributeError:
        gating_bitmap = GATING_BITMAP
        print(
            "No gating_bitmap found in toml config so using default :",
            gating_bitmap,
        )
    return gating_bitmap


def parse_toml_config_to_VelocityControlledQuadcopter2DEnv(
    toml_config_filename, gating_bitmap_filename=None
):
    with open(toml_config_filename, "r") as config:
        config_dict = toml.load(config)
    config = Bunch(config_dict)

    min_state = parse_min_state(config)
    max_state = parse_max_state(config)
    min_velocity = parse_min_velocity(config)
    max_velocity = parse_max_velocity(config)
    min_acceleration = parse_min_acceleration(config)
    max_acceleration = parse_max_acceleration(config)

    if gating_bitmap_filename is None:
        gating_bitmap_filename = parse_gating_bitmap(config)
    low_process_noise_var = parse_low_process_noise_var(config)
    high_process_noise_var = parse_high_process_noise_var(config)

    delta_time = parse_delta_time(config)
    velocity_init = parse_velocity_init(config)

    env = VelocityControlledQuadcopter2DEnv(
        min_observation=min_state,
        max_observation=max_state,
        min_action=min_velocity,
        max_action=max_velocity,
        low_process_noise_var=low_process_noise_var,
        high_process_noise_var=high_process_noise_var,
        gating_bitmap=gating_bitmap_filename,
        velocity_init=velocity_init,
        delta_time=delta_time,
        min_acceleration=min_acceleration,
        max_acceleration=max_acceleration,
    )
    return env


if __name__ == "__main__":
    toml_config_filename = "../scenario-1/env_config.toml"
    env = parse_toml_config_to_VelocityControlledQuadcopter2DEnv(toml_config_filename)
