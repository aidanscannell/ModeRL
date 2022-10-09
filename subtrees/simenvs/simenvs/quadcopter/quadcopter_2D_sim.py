#!/usr/bin/env python3
import cv2
import imageio

# Who's up for a gathering/small party/fire in the woods past Beese's on Saturday eve (into the night)? Will need to have a rain check nearer the time but rain atm isn't very heavy so we should be fine. We can take some tarps to be safe. Expect music, dancing, midnight swimming and general party vibes.
# import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import tensorflow as tf
import tf_agents
from simenvs.visualisation import EnvRenderer
from tf_agents.environments import (
    py_environment,
    tf_environment,
    tf_py_environment,
    utils,
)
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


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
DELTA_TIME = 0.001
VELOCITY_INIT = 0.0


class Quadcopter2DEnv(py_environment.PyEnvironment):
    def __init__(
        self,
        start_state,
        target_state,
        min_observation=MIN_STATE,
        max_observation=MAX_STATE,
        min_action=MIN_VELOCITY,
        max_action=MAX_VELOCITY,
        low_process_noise_var=LOW_PROCESS_NOISE_VAR,
        high_process_noise_var=HIGH_PROCESS_NOISE_VAR,
        gating_bitmap=None,
        velocity_init=0.0,
        delta_time=DELTA_TIME,
        min_acceleration=MIN_ACCELERATION,
        max_acceleration=MAX_ACCELERATION,
    ):
        # self.gravitational_acceleration = tf.constant(9.81, dtype=float_type)
        self.e3 = tf.constant([[0.0], [0.0], [1.0]], dtype=float_type)
        self.mass = tf.constant(1.0, dtype=float_type)
        self.inertia_matrix = tf.constant([[1.0]], dtype=float_type)
        self.inv_inertia_matrix = 1.0 / self.inertia_matrix

        # simulation parameters
        self.start_state = start_state
        self.target_state = target_state
        self.state_dim = 6
        self.control_dim = 2

        self.state_init = start_state
        print("self.state_init")
        print(self.state_init)
        self._state = self.state_init
        self.delta_time = delta_time

        # environment parameters
        if isinstance(low_process_noise_var, np.ndarray):
            self.low_process_noise_var = low_process_noise_var
        else:
            print("low_process_noise_var isn't array so broadcasting")
            self.low_process_noise_var = low_process_noise_var * np.ones(num_states)
        if isinstance(high_process_noise_var, np.ndarray):
            self.high_process_noise_var = high_process_noise_var
        else:
            print("high_process_noise_var isn't array so broadcasting")
            self.high_process_noise_var = high_process_noise_var * np.ones(num_states)

        # # configure action spec
        min_action = np.array([-10, -10])
        max_action = np.array([10, 10])
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(self.control_dim,),
            dtype=float_type,
            minimum=min_action,
            maximum=max_action,
            name="action",
        )
        # configure observation spec
        # if not isinstance(min_observation, np.ndarray):
        #     min_observation = min_observation * np.ones(state_dim)
        #     print("min_observation isn't array so broadcasting")
        # if not isinstance(max_observation, np.ndarray):
        #     max_observation = max_observation * np.ones(state_dim)
        #     print("max_observation isn't array so broadcasting")
        min_observation = np.array([-3.0, -3.0, -0.5, -0.5, -360, -5])
        max_observation = np.array([3.0, 3.0, 0.5, 0.5, 360, 5])
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.state_dim,),
            dtype=float_type,
            minimum=min_observation,
            maximum=max_observation,
            name="observation",
        )
        self.episode_ended = False

        if gating_bitmap is None:
            resolution = BITMAP_RESOLUTION
            self.gating_bitmap = np.ones([resolution, resolution])
        elif isinstance(gating_bitmap, str):
            self.gating_bitmap = cv2.imread(gating_bitmap, cv2.IMREAD_GRAYSCALE)
            self.gating_bitmap = self.gating_bitmap / 255
        elif isinstance(gating_bitmap, np.ndarray):
            self.gating_bitmap = gating_bitmap
        else:
            raise ("gating_bitmap must be np.ndarray or filepath string for bitmap")
        # TODO check x and y are the right way around
        self.num_pixels = np.array(
            # [self.gating_bitmap.shape[0] - 1, self.gating_bitmap.shape[1] - 1]
            [self.gating_bitmap.shape[1] - 1, self.gating_bitmap.shape[0] - 1]
        )

        self.viewer = EnvRenderer(self)

    def state_to_pixel(self, state):
        """Returns the bitmap pixel index associated with state"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        pixel = (
            (state[0, 0:2] - self.observation_spec().minimum[0:2])
            / (
                self.observation_spec().maximum[0:2]
                - self.observation_spec().minimum[0:2]
            )
            * self.num_pixels
        )
        pixel = np.array([-pixel[1], pixel[0]])
        return np.rint(pixel).astype(int)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reset(self, state_init=None):
        """Return initial_time_step."""
        self._current_time_step = self._reset(state_init)
        return self._current_time_step

    def _reset(self, state=None):
        self.step_num = 0
        # print("Reseting environment")
        if state is None:
            self._state = self.state_init
        else:
            self._state = state
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=float_type))

    def _step(self, action):
        self.step_num += 1
        delta_state = self.transition_dynamics(self._state, action)
        # print("delta state")
        # print(delta_state)
        self._state += delta_state
        reward = 0
        # self._episode_ended = True  # remove this when term conds added
        if (self._state[0:3] > self.observation_spec().maximum[0:3]).numpy().any():
            self._episode_ended = True  # remove this when term conds added
        elif (self._state[0:3] < self.observation_spec().minimum[0:3]).numpy().any():
            self._episode_ended = True  # remove this when term conds added
        elif self.step_num > 100:
            self._episode_ended = True  # remove this when term conds added
        if self._episode_ended:
            return ts.termination(np.array([self._state], dtype=float_type), reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=float_type),
                reward=reward,
                discount=1.0,
            )

    def state_to_mixing_prob(self, state):
        pixel = self.state_to_pixel(state)
        gating_value = self.gating_bitmap[pixel[0], pixel[1]]
        return gating_value

    def _process_noise(self, state):
        mean = np.array([0.0, 0.0])
        pixel = self.state_to_pixel(state)
        gating_value = self.gating_bitmap[pixel[0], pixel[1]]
        if gating_value == 1.0:
            var = self.low_process_noise_var
        else:
            var = self.high_process_noise_var
        cov = np.diag(var)
        noise = sp.random.multivariate_normal(mean, cov)
        return noise

    def control_to_thrust(self, control):
        """Total thrust force f_z from rotors"""
        return control[0]

    def control_to_torques(self, control):
        """Torque \tau=[\tau_x, \tau_y, \tau_z]^T relative to the body frame F_b"""
        return control[1:]

    def state_to_positions(self, state):
        """Position h=[x, y, z]^T in world frame F_w"""
        return state[0:3]

    def state_to_velocity(self, state):
        """Velocity v=[v_x, v_y, v_z]^T relative to the world frame F_w"""
        return state[3:6]

    def state_to_rotations(self, state):
        """Roll/pitch/yaw r=[\phi, \theta, \psi]^T"""
        return state[6:9]

    def state_to_angular_velocity(self, state):
        """Angular velocity \omega_b=[p, q, r]^T relative to the body frame F_b"""
        return state[9:12]

    def rotation_matrix(self, rotations):
        roll = rotations[0:1]
        pitch = rotations[1:2]
        yaw = rotations[2:3]
        rotation_matrix = rotation_x(roll) @ rotation_y(pitch) @ rotation_z(yaw)
        return rotation_matrix

    def transition_dynamics(self, state, control, delta_time=None):
        if delta_time is None:
            delta_time = self.delta_time

        # positions = self.state_to_positions(state)
        # print("positioons")
        # print(positions)
        velocity = tf.reshape(self.state_to_velocity(state), [-1, 1])
        print("velocity")
        print(velocity)

        # construct rotation matrix
        rotations = self.state_to_rotations(state)
        rotation_matrix = self.rotation_matrix(rotations)
        # print("rotation_matrix")
        # print(rotation_matrix.shape)

        # velocity update
        thrust = self.control_to_thrust(control)
        # print("thrust")
        # print(thrust.shape)
        thrust_e3 = thrust * self.e3
        delta_velocity = (
            self.gravitational_acceleration * self.e3
            + 1 / self.mass * rotation_matrix @ thrust_e3
        ) * delta_time
        print("delta_velocity")
        print(delta_velocity)

        # rotations update
        angular_velocity = self.state_to_angular_velocity(state)
        p = angular_velocity[0:1]
        q = angular_velocity[1:2]
        r = angular_velocity[2:3]
        angular_velocity_tensor = tf.zeros([3, 3], dtype=float_type)
        angular_velocity_tensor = tf.tensor_scatter_nd_update(
            angular_velocity_tensor, [[0, 1]], -r
        )
        angular_velocity_tensor = tf.tensor_scatter_nd_update(
            angular_velocity_tensor, [[0, 2]], q
        )
        angular_velocity_tensor = tf.tensor_scatter_nd_update(
            angular_velocity_tensor, [[1, 0]], r
        )
        angular_velocity_tensor = tf.tensor_scatter_nd_update(
            angular_velocity_tensor, [[1, 2]], -p
        )
        angular_velocity_tensor = tf.tensor_scatter_nd_update(
            angular_velocity_tensor, [[2, 0]], -q
        )
        angular_velocity_tensor = tf.tensor_scatter_nd_update(
            angular_velocity_tensor, [[2, 1]], p
        )
        # print("angular_velocity_tensor")
        # print(angular_velocity_tensor)
        delta_rotations_matrix = rotation_matrix @ angular_velocity_tensor * delta_time
        delta_rotations = tf.ones((3, 1), dtype=float_type)
        delta_rotations = tf.tensor_scatter_nd_update(
            delta_rotations, [[0, 0]], delta_rotations_matrix[2, 1:2]
        )
        delta_rotations = tf.tensor_scatter_nd_update(
            delta_rotations, [[1, 0]], delta_rotations_matrix[0, 2:3]
        )
        delta_rotations = tf.tensor_scatter_nd_update(
            delta_rotations, [[2, 0]], delta_rotations_matrix[0, 1:2]
        )
        # print("delta rotations")
        # print(delta_rotations.shape)

        # angular velocity update
        torques = self.control_to_torques(control)
        torques = tf.reshape(torques, [-1, 1])
        print("torques")
        print(torques)
        angular_velocity = self.state_to_angular_velocity(state)
        angular_velocity = tf.reshape(angular_velocity, [-1, 1])
        print(angular_velocity)
        delta_angular_velocity = (
            self.inv_inertia_matrix @ torques
            - self.inv_inertia_matrix
            @ angular_velocity_tensor
            @ self.inertia_matrix
            @ angular_velocity
        )
        print(delta_angular_velocity)
        delta_angular_velocity = delta_angular_velocity * delta_time
        print(delta_angular_velocity)
        # print("delta_angular_velocity")
        # print(delta_angular_velocity.shape)

        # positions update
        delta_positions = velocity * delta_time
        print("delta_positions")
        print(delta_positions.shape)

        delta_state = tf.concat(
            [delta_positions, delta_velocity, delta_rotations, delta_angular_velocity],
            0,
        )
        delta_state = tf.reshape(delta_state, [self.state_dim])
        print("delta_state")
        print(delta_state)

        # process_noise = self._process_noise(state)  # external dynamics
        # print("process_noise")
        # print(process_noise)
        # delta_state += process_noise

        return delta_state

    def render(self, state):
        self.viewer.update()
        return self.viewer.fig

    # def render(self, state):
    #     if self.viewer == None:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(projection="3d")
    #         ax.set_xlim(
    #             self.observation_spec().minimum[0], self.observation_spec().maximum[0]
    #         )
    #         ax.set_ylim(
    #             self.observation_spec().minimum[1], self.observation_spec().maximum[1]
    #         )
    #         ax.set_zlim(
    #             self.observation_spec().minimum[2], self.observation_spec().maximum[2]
    #         )
    #         ax.set_xlabel("$x$")
    #         ax.set_ylabel("$y$")
    #         ax.set_zlabel("$z$")
    #         ax.set_title("Quadcopter Simulation")
    #         self.viewer = fig, ax
    #     else:
    #         fig, ax = self.viewer
    #     # if self._line is not None:
    #     #     ax.lines.remove(self._line)
    #     # self._line.pop(0).remove()
    #     position = self.state_to_positions(state).numpy().flatten()
    #     self._line = ax.plot(position[0], position[1], position[2], color="magenta")
    #     # self._line = ax.scatter(
    #     #     position[0], position[1], position[2], marker="*", color="magenta"
    #     # )
    #     return fig, ax


def test_VelocityControlledQuadcopter2DEnv():
    env = VelocityControlledQuadcopter2DEnv()
    utils.validate_py_environment(env, episodes=2)


def test_Quadcopter3DEnv():
    env = Quadcopter3DEnv()
    utils.validate_py_environment(env, episodes=2)


def test_tf_env(env):
    tf_env = tf_py_environment.TFPyEnvironment(env)

    print(
        "Valid tf Environment? ",
        isinstance(tf_env, tf_environment.TFEnvironment),
    )
    print("TimeStep Specs:\n", tf_env.time_step_spec())
    print("Action Specs:\n", tf_env.action_spec())


def record_env():
    gif_path = "./images/test.gif"
    frames_path = "./images/episode-{i}-timestep-{t}.jpg"
    gating_bitmap = "./scenario-1/bitmaps/gating_mask.bmp"

    # pos_init = np.array([2.7, 2.0, 0.0])  # desired start state
    # pos_end_targ = np.array([-2.6, -1.5, 2.5])  # desired end state
    # state_init = np.array([2.7, 2.0, 0.0, 0.0, 0.0, 0.0])  # desired start state
    # state_end_targ = np.array([-2.6, -1.5, 2.5, 0.0, 0.0, 0.0])  # desired end state
    start_state = np.array(
        [2.7, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )  # desired start state
    target_state = np.array(
        [-2.6, -1.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )  # desired end state

    eval_py_env = Quadcopter3DEnv(
        start_state=start_state, target_state=target_state, gating_bitmap=gating_bitmap
    )
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    # eval_env = py_environment.PyEnvironment(eval_py_env)
    # eval_env = eval_py_env

    def policy(i):
        # return tf.reshape(
        #     tf.constant([-0.1, 0.0, 0.0, 0.0], dtype=float_type),
        #     [1, -1]
        #     # tf.constant([-0.1, 0.01, 0.01, 0.01], dtype=float_type), [1, -1]
        # )
        rng = np.random.default_rng(i)
        action = (
            tf_agents.specs.array_spec.sample_bounded_spec(
                eval_py_env.action_spec(), rng
            )
            * 1000
        )
        print(action)
        return tf.reshape(action, [1, -1])

    num_episodes = 3

    ts = []
    for i in range(num_episodes):
        t = 0
        print("Episode {i}".format(i=i))
        time_step = eval_env.reset()
        while not time_step.is_last():
            action = policy(i * t)
            time_step = eval_env.step(action)
            state = time_step.observation
            # fig, ax = eval_py_env.render(state)
            fig = eval_py_env.render(state)
            fig.savefig(frames_path.format(i=i, t=t))
            t = t + 1
        ts.append(t - 1)
        # writer.append_data(imageio.imread(frames_path.format(i=i, t=t)))
    # kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    # imageio.mimsave('./powers.gif', [plot_for_offset(i/4, 100) for i in range(10)], fps=1)

    with imageio.get_writer(gif_path, mode="I", fps=8) as writer:
        for i, t in zip(range(num_episodes), ts):
            for t_ in range(t):
                writer.append_data(imageio.imread(frames_path.format(i=i, t=t_)))


if __name__ == "__main__":

    record_env()
    # test_Quadcopter3DEnv()
    # test_VelocityControlledQuadcopter2DEnv()

    # env = VelocityControlledQuadcopter2DEnv()
    # test_tf_env(env)
