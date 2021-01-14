import matplotlib.pyplot as plt
import numpy as np
from simenvs.generate_dataset_from_env import generate_transitions_dataset

num_data_per_dim = 40
num_actions_per_dim = 4
save_dataset_filename = "./data/quad_sim_scenario_1.npz"
gating_bitmap = "./gating_mask.bmp"
omit_data_mask = "./omit_data_mask.bmp"

state_action_inputs, delta_state_outputs = generate_transitions_dataset(
    gating_bitmap,
    omit_data_mask=omit_data_mask,
    num_data_per_dim=num_data_per_dim,
    num_actions_per_dim=num_actions_per_dim,
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
