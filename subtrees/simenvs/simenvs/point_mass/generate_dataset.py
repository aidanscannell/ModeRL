import os

import matplotlib.pyplot as plt
import numpy as np
from simenvs.generate_dataset_from_env import (
    generate_random_transitions_dataset,
    generate_transitions_dataset,
)
from simenvs.parse_env_toml_config import parse_toml_config_to_RotatingPointMass2DEnv


def generate_point_mass_dataset(
    env_dir,
    num_states=1000,
    num_actions=4,
    omit_data=False,
    plot=False,
    overwrite=False,
):
    """Generate and save transition dataset of RotatingPointMass2DEnv

    :param env_dir: directory contraining toml config, gating bitmap etc
    :param num_states: number of states to randomly generate
    :param num_actions: number of actions to randomly generate
    :param omit_data: whether or not to omit data according to env_3ir/omit_data_mask.bmp
    :param plot: whether or not to plot dataset
    """
    # this_dir, this_filename = os.path.split(__file__)
    toml_env_config_file = os.path.join(
        this_dir, env_dir, "env_config.toml"
    )  # environment config to use
    gating_bitmap_file = os.path.join(env_dir, "gating_mask.bmp")
    save_dataset_filename = os.path.join(
        env_dir,
        "data/rotating_point_mass_" + str(num_states) + "_" + str(num_actions) + ".npz",
    )  # save dataset here
    if not overwrite:
        if os.path.exists(save_dataset_filename):
            print("File exists and no overwrite selected.")
            return

    if omit_data:
        omit_data_mask = "./omit_data_mask.bmp"  # remove states if bmp<255/2
    else:
        omit_data_mask = None

    # configure environment from toml config file
    env = parse_toml_config_to_RotatingPointMass2DEnv(
        toml_env_config_file, gating_bitmap_filename=gating_bitmap_file
    )

    (state_action_inputs, delta_state_outputs,) = generate_transitions_dataset(
        num_states=num_states,
        num_actions=num_actions,
        env=env,
        omit_data_mask=omit_data_mask,
    )

    np.savez(
        save_dataset_filename,
        x=state_action_inputs,
        y=delta_state_outputs,
    )

    if plot:
        plt.quiver(
            state_action_inputs[:, 0],
            state_action_inputs[:, 1],
            delta_state_outputs[:, 0],
            delta_state_outputs[:, 1],
        )
        plt.show()


def generate_random_point_mass_dataset(
    env_dir,
    num_samples=4000,
    omit_data=False,
    plot=False,
    overwrite=False,
):
    """Randomly generate and save transition dataset of PointMass2DEnv

    :param env_dir: directory contraining toml config, gating bitmap etc
    :param num_samples: number of transitions to randomly sample
    :param omit_data: whether or not to omit data according to env_3ir/omit_data_mask.bmp
    :param plot: whether or not to plot dataset
    """
    # this_dir, this_filename = os.path.split(__file__)
    toml_env_config_file = os.path.join(
        this_dir, env_dir, "env_config.toml"
    )  # environment config to use
    gating_bitmap_file = os.path.join(env_dir, "gating_mask.bmp")
    save_dataset_filename = os.path.join(
        env_dir, "data/rotating_point_mass_random_" + str(num_samples) + ".npz"
    )  # save dataset here
    if not overwrite:
        if os.path.exists(save_dataset_filename):
            print("File exists and no overwrite selected.")
            return

    if omit_data:
        omit_data_mask = "./omit_data_mask.bmp"  # remove states if bmp<255/2
    else:
        omit_data_mask = None

    # configure environment from toml config file
    env = parse_toml_config_to_RotatingPointMass2DEnv(
        toml_env_config_file, gating_bitmap_filename=gating_bitmap_file
    )

    state_action_inputs, delta_state_outputs = generate_random_transitions_dataset(
        num_samples=num_samples,
        env=env,
        omit_data_mask=omit_data_mask,
    )

    np.savez(
        save_dataset_filename,
        x=state_action_inputs,
        y=delta_state_outputs,
    )

    if plot:
        plt.quiver(
            state_action_inputs[:, 0],
            state_action_inputs[:, 1],
            delta_state_outputs[:, 0],
            delta_state_outputs[:, 1],
        )
        plt.show()


if __name__ == "__main__":
    # generate data sets for each PointMass2DEnv scenario (sub dirs)
    this_dir = os.getcwd()
    for path in os.scandir(this_dir):
        if path.is_dir():
            if "__pycache__" not in path.path:
                generate_random_point_mass_dataset(
                    env_dir=path.path,
                    # num_samples=10000,
                    num_samples=3000,
                    omit_data=False,
                    # plot=False,
                    # overwrite=False,
                    plot=True,
                    overwrite=True,
                )

                generate_point_mass_dataset(
                    env_dir=path.path,
                    num_states=1000,
                    num_actions=4,
                    omit_data=False,
                    # plot=False,
                    # overwrite=False,
                    plot=True,
                    overwrite=True,
                )

                generate_point_mass_dataset(
                    env_dir=path.path,
                    num_states=1000,
                    num_actions=1,
                    omit_data=False,
                    # plot=False,
                    # overwrite=False,
                    plot=True,
                    overwrite=True,
                )
