import matplotlib.pyplot as plt
import numpy as np
from simenvs.generate_dataset_from_env import generate_transitions_dataset
from simenvs.parse_env_toml_config import (
    parse_toml_config_to_VelocityControlledQuadcopter2DEnv,
)


toml_env_config_file = "./configs/env_config.toml"  # environment config to use
omit_data_mask = "./bitmaps/omit_data_mask.bmp"  # remove states if bmp<255/2
save_dataset_filename = "./data/quad_sim_scenario_1.npz"  # save dataset here

num_states = 1000  # number of states to randomly generate
num_actions = 4  # number of actions to randomly generate

# configure environment from toml config file
env = parse_toml_config_to_VelocityControlledQuadcopter2DEnv(toml_env_config_file)

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

plt.quiver(
    state_action_inputs[:, 0],
    state_action_inputs[:, 1],
    delta_state_outputs[:, 0],
    delta_state_outputs[:, 1],
)
plt.show()
