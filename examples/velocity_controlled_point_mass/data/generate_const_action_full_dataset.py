#!/usr/bin/env python3
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from simenvs.core import make
from simenvs.generate_dataset_from_env import generate_transitions_dataset_const_action

from .utils import generate_random_transitions_dataset_from_env


def generate_const_action_transitions_dataset_from_env(
    env_name: str,
    action,
    save_dir: str = None,
    num_states: int = 2000,
    omit_data_mask: Optional[str] = None,
    plot: bool = False,
    random_seed: int = 42,
):
    """Randomly generate and save transition dataset of VelocityControlledPointMass2DEnv

    :param env_dir: directory contraining toml config, gating bitmap etc
    :param num_samples: number of transitions to randomly sample
    :param omit_data: whether or not to omit data according to env_3ir/omit_data_mask.bmp
    :param plot: whether or not to plot dataset
    :param random_seed: random seed for numpy and tensorflow
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # configure environment from toml config file
    env = make(env_name)

    (
        state_action_inputs,
        delta_state_outputs,
    ) = generate_transitions_dataset_const_action(
        action=action, num_states=num_states, env=env, omit_data_mask=omit_data_mask
    )

    if save_dir is None:
        save_dir = "./"
    save_dataset_filename = (
        save_dir
        + str(state_action_inputs.shape[0])
        + "_samples_"
        + str(random_seed)
        + "_seed.npz"
    )

    state_dim = delta_state_outputs.shape[-1]
    np.savez(
        save_dataset_filename,
        x=state_action_inputs[:, 0:state_dim],
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
    # scenario = "5"
    scenario = "7"
    num_states = 2000
    action = np.array([[1.0, 1.0]])
    random_seed = 42

    env_name = "velocity-controlled-point-mass/scenario-" + scenario
    save_dir = (
        "./velocity_controlled_point_mass/data/scenario_"
        + scenario
        + "/const_action_full_dataset_t0p25_"
    )
    omit_data_mask = None

    generate_const_action_transitions_dataset_from_env(
        env_name=env_name,
        action=action,
        save_dir=save_dir,
        num_states=num_states,
        omit_data_mask=omit_data_mask,
        plot=True,
        # plot=False,
        random_seed=random_seed,
    )
