import cv2
import jax
import jax.numpy as jnp
import numpy as np

from simenvs.quadcopter_sim import VelocityControlledQuadcopter2DEnv

float_type = np.float64

# control constraints
MIN_VELOCITY = -10
MAX_VELOCITY = 10
MIN_ACCELERATION = -10
MAX_ACCELERATION = 10
MIN_ACTION = MIN_VELOCITY
MAX_ACTION = MAX_VELOCITY

# observation constraints
MIN_OBSERVATION = -3
MAX_OBSERVATION = 3

VELOCITY_INIT = 0.0
DELTA_TIME = 0.1

NUM_PIXELS = np.array([600 - 1, 600 - 1])



def generate_random_states(state_dim, num_states, min_state, max_state):
    """Generate num_states random states in domain [min_state, max_state]

    :returns: states [num_states, state_dim]
    """
    states = np.random.uniform(
        min_state,
        max_state,
        (num_states, state_dim),
    )
    return states


def generate_random_actions(action_dim, num_actions, min_action, max_action):
    """Generate num_actions random actions in domain [min_action, max_action]

    :returns: actions [num_actions, action_dim]
    """
    # actions = []
    # for _ in range(action_dim):
    #     actions.append(
    #         np.linspace(min_action, max_action, num_data_per_dim).reshape(-1)
    #     )
    # actions = np.stack(actions, -1)
    actions = np.random.uniform(
        min_action,
        max_action,
        (num_actions, action_dim),
    )
    return actions


# @jax.jit
def create_state_action_inputs(num_dims, states, actions):
    # states_x, states_y = np.meshgrid(states[:, 0], states[:, 1])
    # states = np.concatenate(
    #     [states_x.reshape(-1, 1), states_y.reshape(-1, 1)], -1
    # )
    # print("All combinations of states: ", states.shape)
def apply_mask_to_states(states, env, omit_data_mask=None):
    """Remove values from states where the associated omit_data_mask pixel is <0.5

    :param states: array of states [num_states, state_dim]
    :param env: an instance of VelocityControlledQuadcopter2DEnv
    :param omit_data_mask: filename for a bitmap or np.ndarray
    :returns: array of states with elements removed [new_num_states, state_dim]
    """
    if omit_data_mask is None:
        return states
    elif isinstance(omit_data_mask, str):
        omit_data_mask = cv2.imread(omit_data_mask, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('GFG', omit_data_mask)
        omit_data_mask = omit_data_mask / 255
    elif isinstance(omit_data_mask, np.ndarray):
        omit_data_mask = omit_data_mask
    else:
        raise (
            "omit_data_mask must be np.ndarray or filepath string for bitmap"
        )

    rows_to_delete = []
    for row in range(states.shape[0]):
        pixel = env.state_to_pixel(states[row, :])
        if omit_data_mask[pixel[0], pixel[1]] < 0.5:
            rows_to_delete.append(row)

    states = np.delete(states, rows_to_delete, 0)
    return states

    def grid_action(action):
        action = action.reshape(1, -1)
        num_test = states.shape[0]
        actions_x, actions_y = np.meshgrid(states[:, 0], states[:, 1])
        actions = np.concatenate(
            [actions_x.reshape(-1, 1), actions_y.reshape(-1, 1)], -1
        )
        action_broadcast = jnp.tile(action, (num_test, 1))
        state_action = jnp.concatenate([states, action_broadcast], -1)
        return state_action

    states_actions = jax.vmap(grid_action, in_axes=0)(actions)
    state_action_inputs = states_actions.reshape(-1, 2 * num_dims)
    return state_action_inputs


def transition_dynamics(state_action, env):
    """Wrapper for calling env.transition_dynamics(state, action) with state_action

    :param state_action: state-action tuple [state_dim+action_dim]
    :param env: instance of VelocityControlledQuadcopter2DEnv
    :returns: delta_state [state_dim]
    """
    state_action = state_action.reshape(1, -1)
    num_dims = int(state_action.shape[1] / 2)
    state = state_action[:, :num_dims]
    action = state_action[:, num_dims:]
    delta_state = env.transition_dynamics(state, action)
    return delta_state.reshape(-1)

def state_to_pixel(state, env):
    if len(state.shape) == 1:
        state = state.reshape(1, -1)
    pixel = (
        (state[0, :] - env.observation_spec().minimum)
        / (env.observation_spec().maximum - env.observation_spec().minimum)
        * NUM_PIXELS
    )
    # pixel *= np.array([-1, 1])
    return np.rint(pixel).astype(int)


def generate_transitions_dataset(
    gating_bitmap,
    omit_data_mask=None,
    num_data_per_dim=10,
    num_actions_per_dim=4,
):
    num_dims = 2
    env = VelocityControlledQuadcopter2DEnv(gating_bitmap=gating_bitmap)

    states = gen_dummy_states(num_dims, num_data_per_dim=num_data_per_dim)
    print("Initial states shape: ", states.shape)

    actions = gen_dummy_actions(num_dims, num_data_per_dim=num_actions_per_dim)
    print("Initial actions shape: ", actions.shape)
    state_action_inputs = create_state_action_inputs(num_dims, states, actions)
    print("State action inputs shape: ", state_action_inputs.shape)
    # print(state_action_inputs)

    num_data = state_action_inputs.shape[0]
    delta_state_outputs = []
    for row in range(num_data):
        delta_state = transition_dynamics(state_action_inputs[row, :], env)
        delta_state_outputs.append(delta_state)
    delta_state_outputs = np.stack(delta_state_outputs)

    print("Delta state outputs: ", delta_state_outputs.shape)
    return state_action_inputs, delta_state_outputs



def generate_transitions_dataset_const_action(
    action, gating_bitmap, omit_data_mask=None, num_states=10
):
    num_dims = 2
    env = VelocityControlledQuadcopter2DEnv(gating_bitmap=gating_bitmap)

    states = gen_dummy_states(num_dims, num_data_per_dim=num_data_per_dim)
    print("Initial states shape: ", states.shape)
    states = apply_mask_to_states(states, env, omit_data_mask)
    print("Initial states shape after applying mask: ", states.shape)

    print("Initial action shape: ", action.shape)
    if len(action.shape) == 1:
        action = action.reshape(1, -1)
    state_action_inputs = create_state_action_inputs(num_dims, states, action)
    print("State action inputs shape: ", state_action_inputs.shape)
    # print(state_action_inputs)

    num_data = state_action_inputs.shape[0]
    delta_state_outputs = []
    for row in range(num_data):
        delta_state = transition_dynamics(state_action_inputs[row, :], env)
        delta_state_outputs.append(delta_state)
    delta_state_outputs = np.stack(delta_state_outputs)

    print("Delta state outputs: ", delta_state_outputs.shape)
    return state_action_inputs, delta_state_outputs


# def generate_transitions_dataset_const_action(action):
#     num_data_per_dim = 10
#     num_actions_per_dim = 4
#     state_action_inputs, delta_state_outputs = generate_transitions_dataset(
#         num_data_per_dim=num_data_per_dim,
#         num_actions_per_dim=num_actions_per_dim,
#     )
#     np.savez(
#         "./data/quad_sim_data_new.npz",
#         x=state_action_inputs,
#         y=delta_state_outputs,
#     )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num_data_per_dim = 40
    num_actions_per_dim = 1
    save_dataset_filename = "./data/quad_sim_data_constant_action.npz"
    gating_bitmap = "./gating_network.bmp"

    state_action_inputs, delta_state_outputs = generate_transitions_dataset(
        gating_bitmap,
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
