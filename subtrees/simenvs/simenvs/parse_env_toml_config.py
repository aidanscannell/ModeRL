import numpy as np
import toml
from bunch import Bunch

from simenvs.quadcopter import VelocityControlledQuadcopter2DEnv
from simenvs.velocity_controlled_point_mass import VelocityControlledPointMass2DEnv
from simenvs.point_mass import RotatingPointMass2DEnv

# control constraints
MIN_VELOCITY = -10
MAX_VELOCITY = 10
MIN_ACCELERATION = -10
MAX_ACCELERATION = 10
MIN_ACTION = MIN_VELOCITY
MAX_ACTION = MAX_VELOCITY
MIN_TORQUE = -1.0
MAX_TORQUE = -1.0
MIN_THRUST = -2.0
MAX_THRUST = 2.0

# observation constraints
MIN_STATE = -3.0
MAX_STATE = 3.0

# system parameters
MASS = 1.2
MOMENT_OF_INERTIA = 2.0

# environment parameters
LOW_PROCESS_NOISE_VAR = np.array([0.000001, 0.000002])
HIGH_PROCESS_NOISE_VAR = np.array([0.0001, 0.00004])
LOW_PROCESS_NOISE_MEAN = np.array([0.00001, 0.000002])
HIGH_PROCESS_NOISE_MEAN = np.array([0.001, 0.008])
# BITMAP_RESOLUTION = 600
GATING_BITMAP = None

# simulation parameters
DELTA_TIME = 0.05
VELOCITY_INIT = 0.0
# CONSTANT_ERROR = np.array([0.0, 0.0])
CONSTANT_ERROR = 0.0


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


def parse_min_thrust(config):
    try:
        min_thrust = config.min_thrust
        if isinstance(min_thrust, list):
            min_thrust = parse_list_to_array(min_thrust)
    except AttributeError:
        min_thrust = MIN_THRUST
        print(
            "No min_thrust found in toml config so using default :",
            min_thrust,
        )
    return min_thrust


def parse_max_thrust(config):
    try:
        max_thrust = config.max_thrust
        if isinstance(max_thrust, list):
            max_thrust = parse_list_to_array(max_thrust)
    except AttributeError:
        max_thrust = MAX_THRUST
        print(
            "No max_thrust found in toml config so using default :",
            max_thrust,
        )
    return max_thrust


def parse_min_torque(config):
    try:
        min_torque = config.min_torque
        if isinstance(min_torque, list):
            min_torque = parse_list_to_array(min_torque)
    except AttributeError:
        min_torque = MIN_TORQUE
        print(
            "No min_torque found in toml config so using default :",
            min_torque,
        )
    return min_torque


def parse_max_torque(config):
    try:
        max_torque = config.max_torque
        if isinstance(max_torque, list):
            max_torque = parse_list_to_array(max_torque)
    except AttributeError:
        max_torque = MAX_TORQUE
        print(
            "No max_torque found in toml config so using default :",
            max_torque,
        )
    return max_torque


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


def parse_low_process_noise_mean(config):
    try:
        low_process_noise_mean = config.low_process_noise_mean
        if isinstance(low_process_noise_mean, list):
            low_process_noise_mean = parse_list_to_array(low_process_noise_mean)
    except AttributeError:
        low_process_noise_mean = LOW_PROCESS_NOISE_MEAN
        print(
            "No low_process_noise_mean found in toml config so using default :",
            low_process_noise_mean,
        )
    return low_process_noise_mean


def parse_high_process_noise_mean(config):
    try:
        high_process_noise_mean = config.high_process_noise_mean
        if isinstance(high_process_noise_mean, list):
            high_process_noise_mean = parse_list_to_array(high_process_noise_mean)
    except AttributeError:
        high_process_noise_mean = HIGH_PROCESS_NOISE_MEAN
        print(
            "No high_process_noise_mean found in toml config so using default :",
            high_process_noise_mean,
        )
    return high_process_noise_mean


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


def parse_constant_error(config):
    try:
        constant_error = config.constant_error
        if isinstance(constant_error, list):
            constant_error = parse_list_to_array(constant_error)
    except AttributeError:
        constant_error = CONSTANT_ERROR
        print(
            "No constant_error found in toml config so using default :",
            constant_error,
        )
    return constant_error


def parse_mass(config):
    try:
        mass = config.mass
        if isinstance(mass, list):
            mass = parse_list_to_array(mass)
    except AttributeError:
        mass = MASS
        print(
            "No mass found in toml config so using default :",
            mass,
        )
    return mass


def parse_moment_of_inertia(config):
    try:
        moment_of_inertia = config.moment_of_inertia
        if isinstance(moment_of_inertia, list):
            moment_of_inertia = parse_list_to_array(moment_of_inertia)
    except AttributeError:
        moment_of_inertia = MOMENT_OF_INERTIA
        print(
            "No moment_of_inertia found in toml config so using default :",
            moment_of_inertia,
        )
    return moment_of_inertia


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


def parse_toml_config_to_VelocityControlledPointMass2DEnv(
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
    low_process_noise_mean = parse_low_process_noise_mean(config)
    high_process_noise_mean = parse_high_process_noise_mean(config)

    delta_time = parse_delta_time(config)
    # velocity_init = parse_velocity_init(config)
    constant_error = parse_constant_error(config)

    env = VelocityControlledPointMass2DEnv(
        min_observation=min_state,
        max_observation=max_state,
        min_action=min_velocity,
        max_action=max_velocity,
        low_process_noise_var=low_process_noise_var,
        high_process_noise_var=high_process_noise_var,
        low_process_noise_mean=low_process_noise_mean,
        high_process_noise_mean=high_process_noise_mean,
        gating_bitmap=gating_bitmap_filename,
        # velocity_init=velocity_init,
        constant_error=constant_error,
        delta_time=delta_time,
        min_acceleration=min_acceleration,
        max_acceleration=max_acceleration,
    )
    return env


def parse_toml_config_to_RotatingPointMass2DEnv(
    toml_config_filename, gating_bitmap_filename=None
):
    with open(toml_config_filename, "r") as config:
        config_dict = toml.load(config)
    config = Bunch(config_dict)

    min_state = parse_min_state(config)
    max_state = parse_max_state(config)
    min_thrust = parse_min_thrust(config)
    max_thrust = parse_max_thrust(config)
    min_torque = parse_min_torque(config)
    max_torque = parse_max_torque(config)
    min_action = np.array([min_thrust, min_torque])
    max_action = np.array([max_thrust, max_torque])

    mass = parse_mass(config)
    moment_of_inertia = parse_moment_of_inertia(config)

    if gating_bitmap_filename is None:
        gating_bitmap_filename = parse_gating_bitmap(config)
    low_process_noise_var = parse_low_process_noise_var(config)
    high_process_noise_var = parse_high_process_noise_var(config)
    low_process_noise_mean = parse_low_process_noise_mean(config)
    high_process_noise_mean = parse_high_process_noise_mean(config)

    delta_time = parse_delta_time(config)
    constant_error = parse_constant_error(config)

    env = RotatingPointMass2DEnv(
        mass=mass,
        moment_of_inertia=moment_of_inertia,
        min_observation=min_state,
        max_observation=max_state,
        min_action=min_action,
        max_action=max_action,
        low_process_noise_var=low_process_noise_var,
        high_process_noise_var=high_process_noise_var,
        low_process_noise_mean=low_process_noise_mean,
        high_process_noise_mean=high_process_noise_mean,
        gating_bitmap=gating_bitmap_filename,
        constant_error=constant_error,
        delta_time=delta_time,
    )
    return env
