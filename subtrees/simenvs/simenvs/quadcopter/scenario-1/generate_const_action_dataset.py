import matplotlib.pyplot as plt
import numpy as np
from simenvs.generate_dataset_from_env import generate_transitions_dataset_const_action
from simenvs.parse_env_toml_config import (
    parse_toml_config_to_VelocityControlledQuadcopter2DEnv,
)


toml_env_config_file = "./configs/env_config.toml"  # environment config to use
omit_data_mask = "./bitmaps/omit_data_mask.bmp"  # remove states if bmp<255/2
save_dataset_filename = "./data/quad_sim_const_action_scenario_1.npz"

# omit_data_mask = None
# save_dataset_filename = (
#     "./data/quad_sim_const_action_scenario_1_all_observations.npz"  # save dataset here
# )

num_states = 4000  # number of states to randomly generate
action = np.array([0.05, -0.2])

# configure environment from toml config file
env = parse_toml_config_to_VelocityControlledQuadcopter2DEnv(toml_env_config_file)

(state_action_inputs, delta_state_outputs,) = generate_transitions_dataset_const_action(
    action, num_states=num_states, env=env, omit_data_mask=omit_data_mask
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
plt.xlabel("$x$")
plt.ylabel("$y$")
# plt.savefig("./images/quiver_const_action_scenario_1.pdf", transparent=True)
plt.savefig(
    "./images/quiver_const_action_scenario_1_all_observations.pdf", transparent=True
)
plt.show()
