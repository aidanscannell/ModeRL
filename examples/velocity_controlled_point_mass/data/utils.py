#!/usr/bin/env python3
from typing import Optional

import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow import default_float
from simenvs.core import make
from simenvs.generate_dataset_from_env import generate_random_transitions_dataset
from sklearn.model_selection import train_test_split


@gin.configurable
def load_vcpm_dataset(
    filename: str,
    trim_coords: Optional[list] = None,  # [x1_low, x1_high, x2]
    standardise: bool = False,
    # standardise: bool = True,
    plot: Optional[bool] = False,
    test_split_size: float = 0.0,
):
    data = np.load(filename)
    # if num_inputs is not None:
    #     X = data["x"][:, 0:num_inputs]
    # else:
    #     X = data["x"]
    # if num_outputs is not None:
    #     Y = data["y"][:, 0:num_outputs]
    # else:
    #     Y = data["y"]
    X = data["x"]
    Y = data["y"]
    print("X: {}, Y: {}".format(X.shape, Y.shape))

    # remove some data points
    def trim_dataset(X, Y, trim_coords):
        mask_0 = X[:, 0] < trim_coords[0][0]
        mask_1 = X[:, 1] < trim_coords[0][1]
        mask_2 = X[:, 0] > trim_coords[1][0]
        mask_3 = X[:, 1] > trim_coords[1][1]
        mask = mask_0 | mask_1 | mask_2 | mask_3
        X_partial = X[mask, :]
        Y_partial = Y[mask, :]
        return X_partial, Y_partial

    if trim_coords is not None:
        # print("trim_coords")
        # print(trim_coords)
        X, Y = trim_dataset(X, Y, trim_coords)
    print("X_trimmed: {}, Y_trimmed: {}".format(X.shape, Y.shape))

    if standardise:
        X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
        Y[:, 0] = (Y[:, 0] - Y[:, 0].mean()) / Y[:, 0].std()
        Y[:, 1] = (Y[:, 1] - Y[:, 1].mean()) / Y[:, 1].std()
        # Y = (Y - Y.mean()) / Y.std()

    if test_split_size > 0:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_split_size, random_state=42
        )
        X_test = tf.convert_to_tensor(X_test, dtype=default_float())
        Y_test = tf.convert_to_tensor(Y_test, dtype=default_float())
        test_data = (X_test, Y_test)
    else:
        X_train, Y_train = X, Y

    X_train = tf.convert_to_tensor(X_train, dtype=default_float())
    Y_train = tf.convert_to_tensor(Y_train, dtype=default_float())
    train_data = (X_train, Y_train)

    # plot = False
    if plot:
        plt.quiver(
            X_train[:, 0],
            X_train[:, 1],
            Y_train[:, 0],
            Y_train[:, 1],
            label="Train",
            color="k",
        )
        if test_split_size > 0:
            plt.quiver(
                X_test[:, 0],
                X_test[:, 1],
                Y_test[:, 0],
                Y_test[:, 1],
                label="Test",
                color="r",
            )
        plt.legend()
        plt.show()

    # print("X_train[:, 0].min()")
    # print(tf.reduce_min(X_train[:, 0]))
    # print(tf.reduce_min(X_train[:, 1]))
    # print(tf.reduce_min(X_train[:, 2]))
    # print(tf.reduce_min(X_train[:, 3]))
    # print(tf.reduce_max(X_train[:, 0]))
    # print(tf.reduce_max(X_train[:, 1]))
    # print(tf.reduce_max(X_train[:, 2]))
    # print(tf.reduce_max(X_train[:, 3]))
    print("X_train: {}, Y_train: {}".format(X_train.shape, Y_train.shape))
    if test_split_size > 0:
        print("X_test: {}, Y_test: {}".format(X_test.shape, Y_test.shape))
        return train_data, test_data
    else:
        return train_data, None


def generate_random_transitions_dataset_from_env(
    env_name: str,
    save_dir: str = None,
    num_samples: int = 4000,
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
    # key = random.PRNGKey(random_seed)

    # configure environment from toml config file
    env = make(env_name)

    state_action_inputs, delta_state_outputs = generate_random_transitions_dataset(
        num_samples=num_samples,
        env=env,
        omit_data_mask=omit_data_mask,
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


# def generate_dynamics_dataset_from_env(
#     tf_env,
#     tf_policy=None,
#     num_steps_per_episode=500,
#     num_episodes=10,
#     max_num_data=10000,
# ):
#     # Init random policy
#     if tf_policy is None:
#         tf_policy = tf_agents.policies.random_tf_policy.RandomTFPolicy(
#             action_spec=tf_env.action_spec(), time_step_spec=tf_env.time_step_spec()
#         )

#     # Create replay buffer
#     model_training_buffer_capacity = max_num_data
#     collect_model_training_episodes = num_episodes
#     model_training_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#         tf_policy.trajectory_spec,
#         batch_size=tf_env.batch_size,
#         max_length=model_training_buffer_capacity,
#     )

#     # Driver loop
#     step_counter = tf.zeros(shape=(tf_env.batch_size,))
#     episode_counter = tf.zeros(shape=(tf_env.batch_size,))
#     time_step = tf_env.reset()
#     policy_state = tf_policy.get_initial_state()

#     while episode_counter < num_episodes and step_counter < num_steps_per_episode:
#         action_step = tf_policy.action(time_step, policy_state)
#         next_time_step = tf_env.step(action_step.action)

#         # traj = from_transition(time_step, action_step, next_time_step)
#         # for observer in self._transition_observers:
#         #     observer((time_step, action_step, next_time_step))
#         # for observer in self.observers:
#         #     observer(traj)

#     # Init lists for data collection
#     # state_control_inputs = []
#     # delta_state_outputs = []

#     # def dynamics_dataset_observer(args):
#     #     time_step, policy_step, next_time_step = args

#     #     previous_state = time_step.observation[0, :, :]
#     #     current_state = next_time_step.observation[0, :, :]
#     #     control = policy_step.action[0, :, :]

#     #     state_control = tf.concat([previous_state, control], -1)
#     #     delta_state = current_state - previous_state

#     #     state_control_inputs.append(state_control)
#     #     delta_state_outputs.append(delta_state)

#     model_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
#         tf_env,
#         tf_policy,
#         observers=[model_training_buffer.add_batch],
#         num_episodes=collect_model_training_episodes,
#     )

#     model_collect_driver.run()
#     print("finished running")
#     print(model_training_buffer)
#     return model_training_buffer

# # Add an observer that create dataset
# observer = [dynamics_dataset_observer]
# driver = dynamic_step_driver.DynamicStepDriver(
#     tf_env,
#     tf_policy,
#     transition_observers=observer,
#     num_steps=timesteps_per_episode,
#     # num_steps=10,
# )
# for e in range(num_episodes):
#     print(e)
#     driver.run(maximum_iterations=timesteps_per_episode)
#     # driver.run(maximum_iterations=1)

# state_control_inputs = tf.concat(state_control_inputs, 0)
# delta_state_outputs = tf.concat(delta_state_outputs, 0)
# return state_control_inputs, delta_state_outputs


# def generate_velocity_controlled_point_mass_dataset(
#     env_name,
#     start_state,
#     tf_policy=None,
#     timesteps_per_episode=500,
#     num_episodes=10,
#     max_num_data=10000,
#     save_dataset_filename=None,
#     plot=True,
# ):
#     # Configure environment
#     # env = make("velocity-controlled-point-mass/" + scenario)
#     env = make(env_name)
#     start_state = tf.constant(start_state, dtype=default_float())
#     env.state_init = start_state
#     tf_env = tf_py_environment.TFPyEnvironment(env)

#     # state_control_inputs, delta_state_outputs = generate_dynamics_dataset_from_env(
#     model_training_buffer = generate_dynamics_dataset_from_env(
#         tf_env,
#         tf_policy=tf_policy,
#         timesteps_per_episode=timesteps_per_episode,
#         num_episodes=num_episodes,
#         max_num_data=max_num_data,
#     )
#     print(
#         "X.shape: {}, Y.shape: {}".format(
#             state_control_inputs.shape, delta_state_outputs.shape
#         )
#     )

#     if save_dataset_filename is not None:
#         np.savez(
#             save_dataset_filename,
#             x=state_control_inputs,
#             y=delta_state_outputs,
#         )

#     if plot:
#         plt.quiver(
#             state_control_inputs[:, 0],
#             state_control_inputs[:, 1],
#             delta_state_outputs[:, 0],
#             delta_state_outputs[:, 1],
#         )
#         plt.show()
#     return state_control_inputs, delta_state_outputs
