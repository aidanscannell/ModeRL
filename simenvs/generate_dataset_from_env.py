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

# state constraints (domain)
MIN_STATE = -3.0
MAX_STATE = 3.0

# environment parameters
LOW_PROCESS_NOISE_VAR = np.array([0.000001, 0.000002])
HIGH_PROCESS_NOISE_VAR = np.array([0.0001, 0.00004])
BITMAP_RESOLUTION = 600  # if gating_bitmap=None then use np.ones(600)
GATING_BITMAP = None

# simulation parameters
DELTA_TIME = 0.05
VELOCITY_INIT = 0.0


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


def create_state_action_inputs(states, actions):
    """Create state-action inputs with every combination of states/actions

    :param states: [num_states, state_dim]
    :param actions: [num_actions, action_dim]
    :returns: state-action inputs [num_states*num_actions, state_dim+action_dim]
    """
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

    state_action_dim = actions.shape[1] + states.shape[1]
    states_actions = jax.vmap(grid_action, in_axes=0)(actions)
    state_action_inputs = states_actions.reshape(-1, state_action_dim)
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


def generate_transitions_dataset(
    num_states, num_actions, env, omit_data_mask=None
):
    """Generate dataset of state transitions

    :param num_states: number of states to randomly generate
    :param num_actions: number of actions to randomly generate
    :param env: an instance of VelocityControlledQuadcopter2DEnv
    :param omit_data_mask: bitmaap used to omit data from dataset
    :returns: state_action_inputs [num_states*num_actions, state_dim+action_dim]
              delta_state_outputs [num_states*num_actions, state_dim]
    """
    # generate states
    state_dim = env.observation_spec().shape[1]
    min_state = env.observation_spec().minimum
    max_state = env.observation_spec().maximum
    states = generate_random_states(
        state_dim,
        num_states=num_states,
        min_state=min_state,
        max_state=max_state,
    )
    print("Initial states shape: ", states.shape)
    states = apply_mask_to_states(states, env, omit_data_mask)
    print("Initial states shape after applying mask: ", states.shape)

    # generate actions
    action_dim = env.action_spec().shape[1]
    min_action = env.action_spec().minimum
    max_action = env.action_spec().maximum
    actions = generate_random_actions(
        action_dim,
        num_actions=num_actions,
        min_action=min_action,
        max_action=max_action,
    )
    print("Initial actions shape: ", actions.shape)

    # create every combination of states and actions to get inputs
    state_action_inputs = create_state_action_inputs(states, actions)
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
    action, num_states, env, omit_data_mask=None
):
    """Generate dataset of state transitions with a constant action

    :param action: the constant action used to generate transitions
    :param num_states: number of states to randomly generate
    :param env: an instance of VelocityControlledQuadcopter2DEnv
    :param omit_data_mask: bitmaap used to omit data from dataset
    :returns: state_action_inputs [num_states, state_dim+action_dim]
              delta_state_outputs [num_states, state_dim]
    """
    # generate states
    state_dim = env.observation_spec().shape[1]
    min_state = env.observation_spec().minimum
    max_state = env.observation_spec().maximum
    states = generate_random_states(
        state_dim,
        num_states=num_states,
        min_state=min_state,
        max_state=max_state,
    )
    print("Initial states shape: ", states.shape)
    states = apply_mask_to_states(states, env, omit_data_mask)
    print("Initial states shape after applying mask: ", states.shape)

    # generate actions
    print("Initial action shape: ", action.shape)
    if len(action.shape) == 1:
        action = action.reshape(1, -1)

    # create every combination of states and actions to get inputs
    state_action_inputs = create_state_action_inputs(states, action)
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


def test_generate_transitions_dataset_const_action():
    num_states = 1000
    action = np.array([0.1, 0.1])
    # gating_bitmap = "../scenario-1/gating_mask.bmp"
    # omit_data_mask = "../scenario-1/omit_data_mask.bmp"
    gating_bitmap = "../scenario-2/gating_mask.bmp"
    omit_data_mask = "../scenario-2/omit_data_mask.bmp"
    env = VelocityControlledQuadcopter2DEnv(gating_bitmap=gating_bitmap)

    (
        state_action_inputs,
        delta_state_outputs,
    ) = generate_transitions_dataset_const_action(
        action, num_states=num_states, env=env, omit_data_mask=omit_data_mask
    )
    assert state_action_inputs.shape[0] == delta_state_outputs.shape[0]
    assert state_action_inputs.shape[1] == 2 * delta_state_outputs.shape[1]
    plt.quiver(
        state_action_inputs[:, 0],
        state_action_inputs[:, 1],
        delta_state_outputs[:, 0],
        delta_state_outputs[:, 1],
    )
    plt.show()


def test_generate_transitions_dataset():
    num_states = 100
    num_actions = 16
    # gating_bitmap = "../scenario-1/gating_mask.bmp"
    # omit_data_mask = "../scenario-1/omit_data_mask.bmp"
    gating_bitmap = "../scenario-2/gating_mask.bmp"
    omit_data_mask = "../scenario-2/omit_data_mask.bmp"
    env = VelocityControlledQuadcopter2DEnv(gating_bitmap=gating_bitmap)

    (state_action_inputs, delta_state_outputs) = generate_transitions_dataset(
        num_states=num_states,
        num_actions=num_actions,
        env=env,
        omit_data_mask=omit_data_mask,
    )
    assert state_action_inputs.shape[0] == delta_state_outputs.shape[0]
    assert state_action_inputs.shape[1] == 2 * delta_state_outputs.shape[1]
    plt.quiver(
        state_action_inputs[:, 0],
        state_action_inputs[:, 1],
        delta_state_outputs[:, 0],
        delta_state_outputs[:, 1],
    )
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test_generate_transitions_dataset_const_action()
    test_generate_transitions_dataset()
