#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

cmap = cm.PiYG


class EnvRenderer:
    def __init__(self, env):
        self.env = env

        # create figure/axis
        self.fig = plt.figure()
        # self.fig = plt.subplots(2, 2)
        self.ax_3d = self.fig.add_subplot(2, 2, 1, projection="3d")
        self.ax_x = self.fig.add_subplot(2, 2, 2)
        self.ax_y = self.fig.add_subplot(2, 2, 3)
        self.ax_z = self.fig.add_subplot(2, 2, 4)

        self.quad = {}

        def init_2d_x(ax):
            ax.set_xlim(
                env.observation_spec().minimum[1], env.observation_spec().maximum[1]
            )
            ax.set_ylim(
                env.observation_spec().minimum[0], env.observation_spec().maximum[0]
            )
            ax.set_xlabel("$y$")
            ax.set_ylabel("$x$")
            ax.set_title("Projection")
            self.quad["hub-x"] = ax.plot(
                [],
                [],
                [],
                marker="o",
                color="green",
                markersize=6,
                antialiased=False,
                zorder=3,
            )
            return ax

        def init_2d_y(ax):
            ax.set_xlim(
                env.observation_spec().minimum[0], env.observation_spec().maximum[0]
            )
            ax.set_ylim(
                env.observation_spec().minimum[1], env.observation_spec().maximum[1]
            )
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_title("Projection")
            self.quad["hub-y"] = ax.plot(
                [],
                [],
                [],
                marker="o",
                color="green",
                markersize=6,
                antialiased=False,
                zorder=3,
            )
            return ax

        def init_2d_z(ax):
            ax.set_xlim(
                env.observation_spec().minimum[0], env.observation_spec().maximum[0]
            )
            ax.set_ylim(
                env.observation_spec().minimum[1], env.observation_spec().maximum[1]
            )
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_title("Projection")
            self.quad["hub-z"] = ax.plot(
                [],
                [],
                [],
                marker="o",
                color="green",
                markersize=6,
                antialiased=False,
                zorder=3,
            )
            return ax

        def init_3d(ax):
            # plot gating network
            x = np.arange(
                env.observation_spec().minimum[0],
                env.observation_spec().maximum[0],
                0.1,
            )
            y = np.arange(
                env.observation_spec().minimum[1],
                env.observation_spec().maximum[1],
                0.1,
            )
            X, Y = np.meshgrid(x, y)
            states = np.column_stack([X.reshape(-1), Y.reshape(-1)])
            mixing_probs = []
            for state in states:
                mixing_probs.append(env.state_to_mixing_prob(state))
            mixing_probs = np.stack(mixing_probs, 0)
            # ax.contourf(
            ax.contour(
                X,
                Y,
                mixing_probs.reshape(X.shape),
                zdir="z",
                # offset=env.observation_spec().minimum[2],
                offset=0.0,
                cmap=cmap,
                zorder=1,
            )

            ax.set_xlim(
                env.observation_spec().minimum[0], env.observation_spec().maximum[0]
            )
            ax.set_ylim(
                env.observation_spec().minimum[1], env.observation_spec().maximum[1]
            )
            ax.set_zlim(
                env.observation_spec().minimum[2], env.observation_spec().maximum[2]
            )
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_zlabel("$z$")
            ax.set_title("Quadcopter Simulation")

            # plot start/target positions
            start_pos = self.env.start_state[0:3]
            target_pos = self.env.target_state[0:3]
            ax.scatter(start_pos[0], start_pos[1], start_pos[2], marker="x", color="k")
            ax.scatter(
                target_pos[0], target_pos[1], target_pos[2], color="k", marker="x"
            )
            # ax.annotate(
            #     "Start $\mathbf{x}_0$",
            #     (start_pos[0], start_pos[1], start_pos[2]),
            #     horizontalalignment="left",
            #     verticalalignment="top",
            # )
            # ax.annotate(
            #     "End $\mathbf{x}_f$",
            #     (target_pos[0], target_pos[1], target_pos[2]),
            #     horizontalalignment="left",
            #     verticalalignment="top",
            # )

            # initialise quadcopter lines etc
            self.quad["l1"] = ax.plot(
                [], [], [], color="blue", linewidth=3, antialiased=False, zorder=3
            )
            self.quad["l2"] = ax.plot(
                [], [], [], color="red", linewidth=3, antialiased=False, zorder=3
            )
            self.quad["hub"] = ax.plot(
                [],
                [],
                [],
                marker="o",
                color="green",
                markersize=6,
                antialiased=False,
                zorder=3,
            )
            return ax

        self.ax_3d = init_3d(self.ax_3d)
        self.ax_x = init_2d_x(self.ax_x)
        self.ax_y = init_2d_y(self.ax_y)
        self.ax_z = init_2d_z(self.ax_z)

    def update(self):
        position = self.env.state_to_positions(self.env._state).numpy().flatten()
        rotations = self.env.state_to_rotations(self.env._state).numpy()
        R = self.env.rotation_matrix(rotations)
        # R = self.rotation_matrix(self.quads[key]['orientation'])
        # L = self.quads[key]['L']
        L = 0.5
        points = np.array(
            [[-L, 0, 0], [L, 0, 0], [0, -L, 0], [0, L, 0], [0, 0, 0], [0, 0, 0]]
        ).T
        points = np.dot(R, points)
        points[0, :] += position[0]
        points[1, :] += position[1]
        points[2, :] += position[2]
        self.quad["l1"][0].set_data(points[0, 0:2], points[1, 0:2])
        self.quad["l1"][0].set_zorder(5)
        self.quad["l2"][0].set_zorder(5)
        self.quad["l1"][0].set_3d_properties(points[2, 0:2])
        self.quad["l2"][0].set_data(points[0, 2:4], points[1, 2:4])
        self.quad["l2"][0].set_3d_properties(points[2, 2:4])
        self.quad["hub"][0].set_data(points[0, 5], points[1, 5])
        self.quad["hub"][0].set_3d_properties(points[2, 5])

        # update x/y/z projections
        # self.quad["hub-x"][0].set_data(points[0, 5], points[1, 5])
        # self.quad["hub-y"][0].set_data(points[0, 5], points[2, 5])
        # self.quad["hub-z"][0].set_data(points[2, 5], points[1, 5])
        self.quad["hub-x"][0].set_data(position[0], position[1])
        self.quad["hub-y"][0].set_data(position[0], position[2])
        self.quad["hub-z"][0].set_data(position[1], position[2])

        # self._line = self.ax.plot(
        #     points[0, :], position[1], position[2], color="magenta"
        # )
