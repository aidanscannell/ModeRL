import matplotlib.pyplot as plt
import numpy as np
from simenvs.generate_dataset_from_env import \
    generate_transitions_dataset_const_action

num_states = 4000
save_dataset_filename = "./data/quad_sim_const_action_scenario_1.npz"
gating_bitmap = "./gating_mask.bmp"
omit_data_mask = "./omit_data_mask.bmp"

action = np.array([0.05, -0.2])

state_action_inputs, delta_state_outputs = generate_transitions_dataset_const_action(
    action,
    gating_bitmap,
    omit_data_mask=omit_data_mask,
    num_states=num_states
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
