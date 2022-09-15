#!/usr/bin/env python3
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import patches
from modeopt.mode_opt import ModeOpt
from modeopt.rollouts import (
    collect_data_from_env,
    rollout_controller_in_dynamics,
    rollout_controller_in_env,
)
from modeopt.trajectories import GeodesicTrajectory
from mogpe.keras.plotting import MixtureOfSVGPExpertsContourPlotter

LABELS = {"env": "Environment", "dynamics": "Dynamics"}
COLORS = {"env": "c", "dynamics": "m"}
LINESTYLES = {"env": "-.", "dynamics": ":"}
MARKERS = {"env": "^", "dynamics": "."}

LABELS = {"env": "Environment", "dynamics": "Dynamics", "collocation": "Collocation"}
COLORS = {"env": "c", "dynamics": "m", "collocation": "y"}
LINESTYLES = {"env": "-.", "dynamics": ":", "collocation": "-"}
MARKERS = {"env": "+", "dynamics": "o", "collocation": "x"}

LINESTYLES = {"env": "-", "dynamics": "-", "collocation": "-"}
MARKERS = {"env": "*", "dynamics": ".", "collocation": "d"}

# matplotlib.rcParams.update({"font.size": 10})
# matplotlib.rcParams.update({"font.size": 7})


class ModeOptContourPlotter:
    """Used to plot first two input dimensions using contour plots

    Can handle arbitrary number of experts and output dimensions.
    """

    def __init__(
        self,
        mode_optimiser: ModeOpt,
        test_inputs=None,
        static: bool = True,
        static_trajectories: bool = False,
        explorative: bool = False,  # whether to use explorative_controller or mode_controller
    ):
        self.mode_optimiser = mode_optimiser
        self.mosvgpe_plotter = MixtureOfSVGPExpertsContourPlotter(
            mode_optimiser.dynamics.mosvgpe, test_inputs=test_inputs, static=static
        )
        self.test_inputs = self.mosvgpe_plotter.test_inputs
        self.start_state = self.mode_optimiser.start_state
        self.target_state = self.mode_optimiser.target_state
        self.explorative = explorative

        if not isinstance(self.test_inputs, tf.Tensor):
            self.test_inputs = tf.constant(self.test_inputs)
        if self.mode_optimiser.mode_controller is not None:
            if isinstance(
                self.mode_optimiser.mode_controller.previous_solution,
                GeodesicTrajectory,
            ):
                metric = self.mode_optimiser.mode_controller.previous_solution.manifold.metric(
                    self.test_inputs[:, 0 : self.mode_optimiser.dynamics.state_dim]
                )
                self.metric_trace = tf.linalg.trace(metric)

        self.static_trajectories = static_trajectories
        self.trajectories = self.generate_trajectories()

    def plot_model(self):
        self.plot_trajectories_over_gating_network_gps()
        self.plot_trajectories_over_desired_gating_network_gp()
        self.plot_trajectories_over_mixing_probs()

    def plot_trajectories_over_gating_network_gps(
        self, plot_satisfaction_prob: bool = False
    ):
        fig = self.mosvgpe_plotter.plot_gating_network_gps()
        fig = self.plot_env_given_fig(fig)
        self.plot_trajectories_given_fig(
            fig, plot_satisfaction_prob=plot_satisfaction_prob
        )
        return fig

    def plot_trajectories_over_desired_gating_network_gp(
        self, plot_satisfaction_prob: bool = False
    ):
        fig = self.mosvgpe_plotter.plot_single_gating_network_gp(
            desired_mode=self.mode_optimiser.desired_mode
        )
        fig = self.plot_env_given_fig(fig)
        self.plot_trajectories_given_fig(
            fig, plot_satisfaction_prob=plot_satisfaction_prob
        )
        return fig

    def plot_trajectories_over_mixing_probs(self, plot_satisfaction_prob: bool = False):
        fig = self.mosvgpe_plotter.plot_mixing_probs()
        fig = self.plot_env_given_fig(fig)
        self.plot_trajectories_given_fig(
            fig, plot_satisfaction_prob=plot_satisfaction_prob
        )
        return fig

    def plot_trajectories_over_desired_mixing_prob(
        self, plot_satisfaction_prob: bool = False
    ):
        fig = self.mosvgpe_plotter.plot_single_mixing_prob(
            self.mode_optimiser.desired_mode
        )
        fig = self.plot_env_given_fig(fig)
        self.plot_trajectories_given_fig(
            fig, plot_satisfaction_prob=plot_satisfaction_prob
        )
        return fig

    def plot_trajectories_over_metric_trace(self):
        figsize = self.mosvgpe_plotter.figsize
        fig = plt.figure(figsize=(figsize[0] / 2, figsize[1]))
        gs = fig.add_gridspec(1, 1)
        ax = gs.subplots()
        # fig.suptitle("Trace of expected metric $\\text{tr}(\mathbb{E}[G(\mathbf{x})])$")
        self.mosvgpe_plotter.plot_contf(ax, z=self.metric_trace)
        fig = self.plot_env_given_fig(fig)
        self.plot_trajectories_given_fig(fig)
        return fig

    def plot_data_over_desired_gating_network_gp(self):
        fig = self.mosvgpe_plotter.plot_single_gating_network_gp(
            desired_mode=self.mode_optimiser.desired_mode
        )
        fig = self.plot_env_given_fig(fig)
        self.plot_data_given_fig(fig)
        return fig

    def plot_data_over_mixing_probs(self):
        fig = self.mosvgpe_plotter.plot_mixing_probs()
        fig = self.plot_env_given_fig(fig)
        self.plot_data_given_fig(fig)
        return fig

    def plot_data_given_fig(self, fig, plot_satisfaction_prob: bool = False):
        axs = fig.get_axes()
        mixing_probs = (
            self.mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                self.test_inputs
            )
        )
        for ax in axs:
            self.plot_data_given_ax(ax)
            if plot_satisfaction_prob:
                CS = ax.tricontour(
                    self.test_inputs[:, 0],
                    self.test_inputs[:, 1],
                    mixing_probs[:, self.mode_optimiser.desired_mode].numpy(),
                    [self.mode_optimiser.mode_satisfaction_probability],
                )
                ax.clabel(CS, inline=True, fontsize=10)
        self.plot_env_no_obs_start_end_given_fig(fig)

    def plot_data_given_ax(self, ax):
        states = self.mode_optimiser.dataset[0]
        ax.scatter(
            states[:, 0],
            states[:, 1],
            marker="x",
            color="b",
            linewidth=0.5,
            alpha=0.5,
            label="Observations",
        )

    def plot_trajectories_given_fig(self, fig, plot_satisfaction_prob: bool = False):
        if not self.static_trajectories:
            self.trajectories = self.generate_trajectories()
        for key in self.trajectories.keys():
            self.plot_trajectory_given_fig(fig, self.trajectories[key], key)
        self.plot_env_no_obs_start_end_given_fig(fig)
        if plot_satisfaction_prob:
            for ax in fig.get_axes():
                self.plot_mode_satisfaction_probability_given_ax(ax)

        # fig.legend(loc="lower center", bbox_transform=fig.transFigure)
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = OrderedDict(zip(labels, handles))
        # fig.legend(
        #     by_label.values(),
        #     by_label.keys(),
        #     # loc="upper center",
        #     bbox_to_anchor=(0.5, 0.05),
        #     # loc="lower center",
        #     loc="upper center",
        #     bbox_transform=fig.transFigure,
        #     ncol=len(by_label),
        # )
        # fig.legend(loc="lower center", bbox_transform=fig.transFigure, ncol=8)

    # def map_over_axs(fig):
    #     axs = fig.get_axes()
    #     if isinstance(axs, np.ndarray):
    #         axs = axs.flat
    #         # for ax in axs.flat:
    #         #     fn(ax, trajectory, key)
    #     # elif isinstance(axs, list):
    #     for ax in axs:
    #         fn(ax, trajectory, key)
    #     # else:
    #     # self.plot_trajectory_given_ax(axs, trajectory, key)

    def plot_trajectory_given_fig(self, fig, trajectory, key):
        axs = fig.get_axes()
        # if isinstance(axs, np.ndarray):
        #     for ax in axs.flat:
        #         self.plot_trajectory_given_ax(ax, trajectory, key)
        # elif isinstance(axs, list):
        #     for ax in axs:
        #         self.plot_trajectory_given_ax(ax, trajectory, key)
        # else:
        #     self.plot_trajectory_given_ax(axs, trajectory, key)
        # return fig
        if isinstance(axs, np.ndarray):
            axs = axs.flat
        for ax in axs:
            self.plot_trajectory_given_ax(ax, trajectory, key)
        return fig

    def plot_trajectory_given_ax(self, ax, trajectory, key):
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            label=LABELS[key],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=0.3,
            marker=MARKERS[key],
        )
        self.plot_start_end_pos_given_ax(ax)
        # self.plot_no_observations_given_fig_ax(fig, ax)

    def plot_mode_satisfaction_probability_given_ax(self, ax):
        mixing_probs = (
            self.mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                self.test_inputs
            )
        )
        CS = ax.tricontour(
            self.test_inputs[:, 0],
            self.test_inputs[:, 1],
            mixing_probs[:, self.mode_optimiser.desired_mode].numpy(),
            [self.mode_optimiser.mode_satisfaction_probability],
        )
        ax.clabel(CS, inline=True, fontsize=10)

    def plot_env(self):
        figsize = self.mosvgpe_plotter.figsize
        fig = plt.figure(figsize=(figsize[0] / 2, figsize[1]))
        gs = fig.add_gridspec(1, 1)
        ax = gs.subplots()
        if self.mode_optimiser.dataset is not None:
            ax.quiver(
                self.mode_optimiser.dataset[0][:, 0],
                self.mode_optimiser.dataset[0][:, 1],
                self.mode_optimiser.dataset[1][:, 0],
                self.mode_optimiser.dataset[1][:, 1],
                alpha=0.5,
            )
        self.plot_env_given_fig(fig)
        self.plot_start_end_pos_given_ax(ax, bbox=True)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        return fig

    def plot_env_given_fig(self, fig):
        axs = fig.get_axes()
        if isinstance(axs, np.ndarray):
            axs = axs.flat
        for ax in axs:
            self.plot_env_given_ax(ax)
        return fig

    def plot_env_given_ax(self, ax):
        test_states = self.test_inputs[:, 0 : self.mode_optimiser.dynamics.state_dim]
        mode_probs = []
        for test_state in test_states:
            pixel = self.mode_optimiser.env.state_to_pixel(test_state.numpy())
            mode_probs.append(self.mode_optimiser.env.gating_bitmap[pixel[0], pixel[1]])
        mode_probs = tf.stack(mode_probs, 0)
        CS = ax.tricontour(
            test_states[:, 0],
            test_states[:, 1],
            mode_probs.numpy(),
            [0.5],
        )
        try:
            # clabel = ax.clabel(CS, inline=True, fmt={0.5: "Mode boundary"})
            clabel = ax.clabel(CS, inline=True, fontsize=8, fmt={0.5: "Mode boundary"})
            clabel[0].set_bbox(dict(boxstyle="round,pad=0.1", fc="white", alpha=1.0))
        except IndexError:
            pass

    def plot_start_end_pos_given_ax(self, ax, bbox=False):
        if bbox:
            bbox = dict(boxstyle="round,pad=0.1", fc="thistle", alpha=1.0)
        else:
            bbox = None
        if len(self.start_state.shape) == 1:
            self.start_state = self.start_state[tf.newaxis, :]
        if len(self.target_state.shape) == 1:
            self.target_state = self.target_state[tf.newaxis, :]
        ax.annotate(
            "$\mathbf{x}_0$",
            (self.start_state[0, 0] + 0.1, self.start_state[0, 1]),
            horizontalalignment="left",
            verticalalignment="top",
            bbox=bbox,
        )
        ax.annotate(
            "$\mathbf{x}_f$",
            (self.target_state[0, 0] - 0.1, self.target_state[0, 1]),
            horizontalalignment="right",
            verticalalignment="bottom",
            bbox=bbox,
            # bbox=dict(boxstyle="round,pad=0.1", fc="thistle", alpha=1.0),
        )
        ax.scatter(
            self.start_state[0, 0], self.start_state[0, 1], marker="x", color="k", s=8.0
        )
        ax.scatter(
            self.target_state[0, 0],
            self.target_state[0, 1],
            color="k",
            marker="x",
            s=8.0,
        )

    def generate_trajectories(self):
        if self.explorative:
            dynamics_trajectory, _ = rollout_controller_in_dynamics(
                dynamics=self.mode_optimiser.dynamics,
                controller=self.mode_optimiser.explorative_controller,
                start_state=self.mode_optimiser.start_state.numpy(),
            )
            trajectories = {"dynamics": dynamics_trajectory}
            env_trajectory = rollout_controller_in_env(
                env=self.mode_optimiser.env,
                controller=self.mode_optimiser.explorative_controller,
                start_state=self.mode_optimiser.start_state.numpy(),
            )
            trajectories.update({"env": env_trajectory})
        else:

            dynamics_trajectory = self.mode_optimiser.dynamics_rollout()[0]
            trajectories = {"dynamics": dynamics_trajectory}
            if self.mode_optimiser.env is not None:
                trajectories.update({"env": self.mode_optimiser.env_rollout()})
            if isinstance(
                self.mode_optimiser.mode_controller.previous_solution,
                GeodesicTrajectory,
            ):
                print("is instance of GeodesicTrajectory")
                trajectories.update(
                    {
                        "collocation": self.mode_optimiser.mode_controller.previous_solution.states
                    }
                )
            else:
                print("is NOT instance of GeodesicTrajectory")
                # dynamics_trajectory = self.mode_optimiser.dynamics_rollout()[0]
                # env_trajectory = self.mode_optimiser.env_rollout()
                # trajectories = {"env": env_trajectory, "dynamics": dynamics_trajectory}
        self.trajectories = trajectories
        return trajectories

    def plot_env_no_obs_start_end_given_fig(self, fig, trim_coords=None):
        self.plot_env_given_fig(fig)
        axs = fig.get_axes()
        for ax in axs:
            self.plot_no_observations_given_ax(ax, trim_coords=trim_coords)
            self.plot_start_end_pos_given_ax(ax, bbox=True)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0.5, 0.05),
            loc="upper center",
            bbox_transform=fig.transFigure,
            ncol=len(by_label),
        )

    def plot_no_observations_given_ax(self, ax, trim_coords=None):
        color = "k"
        x_min = self.mode_optimiser.env.observation_spec().minimum[0]
        y_min = self.mode_optimiser.env.observation_spec().minimum[1]
        x_max = self.mode_optimiser.env.observation_spec().maximum[0]
        y_max = self.mode_optimiser.env.observation_spec().maximum[1]
        # y_max = 3
        # y_min = -3
        ax.add_patch(
            patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                edgecolor=color,
                facecolor=color,
                # hatch="///",
                # label="Environment boundary",
                # label="No observations",
                fill=False,
                ls="--",
            )
        )
        if trim_coords:
            if trim_coords[0][0] < x_min:
                trim_coords[0][0] = x_min
            if trim_coords[0][1] < y_min:
                trim_coords[0][1] = y_min
            if trim_coords[1][0] > x_max:
                trim_coords[1][0] = x_max
            if trim_coords[1][1] > y_max:
                trim_coords[1][1] = y_max
            ax.add_patch(
                patches.Rectangle(
                    (trim_coords[0][0], trim_coords[0][1]),
                    trim_coords[1][0] - trim_coords[0][0],
                    trim_coords[1][1] - trim_coords[0][1],
                    edgecolor=color,
                    facecolor=color,
                    hatch="///",
                    label="No observations",
                    fill=False,
                    linewidth=0,
                )
            )
