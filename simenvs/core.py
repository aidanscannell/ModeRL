#!/usr/bin/env python3
import os

from simenvs.parse_env_toml_config import (
    parse_toml_config_to_VelocityControlledQuadcopter2DEnv,
    parse_toml_config_to_VelocityControlledPointMass2DEnv,
    parse_toml_config_to_RotatingPointMass2DEnv,
)


def make(env_name: str):
    this_dir, this_filename = os.path.split(__file__)
    split_name = env_name.split("/")
    if split_name[0] == "velocity-controlled-point-mass":
        if env_name == "velocity-controlled-point-mass/scenario-1":
            env_dir = "velocity_controlled_point_mass/scenario-1"
        elif env_name == "velocity-controlled-point-mass/scenario-2":
            env_dir = "velocity_controlled_point_mass/scenario-2"
        elif env_name == "velocity-controlled-point-mass/scenario-3":
            env_dir = "velocity_controlled_point_mass/scenario-3"
        elif env_name == "velocity-controlled-point-mass/scenario-4":
            env_dir = "velocity_controlled_point_mass/scenario-4"
        toml_env_config_file = os.path.join(this_dir, env_dir, "env_config.toml")
        gating_bitmap_file = os.path.join(this_dir, env_dir, "gating_mask.bmp")
        env = parse_toml_config_to_VelocityControlledPointMass2DEnv(
            toml_env_config_file, gating_bitmap_file
        )
    elif split_name[0] == "rotating-point-mass":
        if env_name == "rotating-point-mass/scenario-1":
            env_dir = "point_mass/scenario-1"
        elif env_name == "rotating-point-mass/scenario-2":
            env_dir = "point_mass/scenario-2"
        elif env_name == "rotating-point-mass/scenario-3":
            env_dir = "point_mass/scenario-3"
        elif env_name == "rotating-point-mass/scenario-4":
            env_dir = "point_mass/scenario-4"
        toml_env_config_file = os.path.join(this_dir, env_dir, "env_config.toml")
        gating_bitmap_file = os.path.join(this_dir, env_dir, "gating_mask.bmp")
        env = parse_toml_config_to_RotatingPointMass2DEnv(
            toml_env_config_file, gating_bitmap_file
        )
    elif split_name[0] == "quadcopter":
        if env_name == "quadcopter/scenario-1":
            env_dir = "quadcopter/scenario-1"
        elif env_name == "quadcopter/scenario-2":
            env_dir = "quadcopter/scenario-2"
        toml_env_config_file = os.path.join(
            this_dir, env_dir, "configs/env_config.toml"
        )
        gating_bitmap_file = os.path.join(this_dir, env_dir, "bitmaps/gating_mask.bmp")

        # configure environment from toml config file
        env = parse_toml_config_to_VelocityControlledQuadcopter2DEnv(
            toml_env_config_file, gating_bitmap_file
        )
    else:
        raise NotImplementedError("No environment named {}".format(env_name))

    # elif env_name == "scenario-1":
    #     env_dir = "scenario-1"
    # elif env_name == "scenario-2":
    #     env_dir = "scenario-2"
    # elif env_name == "3d-scenario-1":
    #     env_dir = "scenario-1"
    return env
