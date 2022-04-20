#!/usr/bin/env python3
import matplotlib.pyplot as plt
from simenvs.core import make
import numpy as np
import tensorflow as tf
from numpy import random

random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)


def get_initial_states(start_state, target_state, width=1.0, num_initial_states=20):
    state_dim = start_state.shape[-1]
    return random.uniform(
        [start_state[0, 0] - width, start_state[0, 1] - width],
        [start_state[0, 0] + width, start_state[0, 1] + width],
        (num_initial_states, state_dim),
    )


def sample_env_at_states(states, num_actions=1, verbose=False):
    # configure environment from toml config file
    env = make(env_name)

    delta_state_outputs = []
    state_control_inputs = []
    for state in states:
        for i in range(num_actions):
            action = np.random.uniform(
                env.action_spec().minimum,
                env.action_spec().maximum,
                (env.action_spec().shape[-1]),
            )
            delta_state = env.transition_dynamics(state, action)
            delta_state_outputs.append(delta_state)
            state_control_input = np.concatenate([state, action], -1)
            state_control_inputs.append(state_control_input)
    delta_state_outputs = np.stack(delta_state_outputs)
    state_control_inputs = np.stack(state_control_inputs)
    if verbose:
        print("State control inputs: ", state_control_inputs.shape)
        print("Delta state outputs: ", delta_state_outputs.shape)
    return state_control_inputs, delta_state_outputs


if __name__ == "__main__":

    scenario = "5"
    scenario = "10"
    # scenario = "7"
    # scenario = "8"
    # scenario = "9"

    start_state = np.array([[-1.0, -2.0]])
    target_state = np.array([[1.7, 3.0]])

    num_samples = 200
    num_initial_states = 200
    width = 0.8

    env_name = "velocity-controlled-point-mass/scenario-" + scenario
    save_dataset_filename = (
        "./velocity_controlled_point_mass/data/scenario_"
        + scenario
        + "/initial_dataset_start_"
        + str(start_state[0, 0])
        + "_end_"
        + str(start_state[0, 1])
        + ".npz"
    )

    initial_states = get_initial_states(
        start_state, target_state, width=width, num_initial_states=num_initial_states
    )
    initial_dataset = sample_env_at_states(initial_states, verbose=True)
    state_action_inputs, delta_state_outputs = initial_dataset
    dataset = initial_dataset

    np.savez(
        save_dataset_filename,
        x=state_action_inputs,
        y=delta_state_outputs,
    )

    plt.quiver(
        state_action_inputs[:, 0],
        state_action_inputs[:, 1],
        delta_state_outputs[:, 0],
        delta_state_outputs[:, 1],
    )
    plt.show()
