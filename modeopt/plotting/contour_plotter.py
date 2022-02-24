#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from modeopt.mode_opt import ModeOpt
from modeopt.trajectories import GeodesicTrajectory
from mogpe.keras.plotting import MixtureOfSVGPExpertsContourPlotter

LABELS = {"env": "Environment", "dynamics": "Dynamics"}
COLORS = {"env": "c", "dynamics": "m"}
LINESTYLES = {"env": "-.", "dynamics": ":"}
MARKERS = {"env": "^", "dynamics": "."}

LABELS = {"env": "Environment", "dynamics": "Dynamics", "collocation": "Collocation"}
COLORS = {"env": "c", "dynamics": "m", "collocation": "y"}
LINESTYLES = {"env": "-.", "dynamics": ":", "collocation": "-"}
MARKERS = {"env": "^", "dynamics": ".", "collocation": "D"}


class ModeOptContourPlotter:
    """Used to plot first two input dimensions using contour plots

    Can handle arbitrary number of experts and output dimensions.
    """

    def __init__(self, mode_optimiser: ModeOpt, test_inputs=None, static: bool = True):
        self.mode_optimiser = mode_optimiser
        self.mosvgpe_plotter = MixtureOfSVGPExpertsContourPlotter(
            mode_optimiser.dynamics.mosvgpe, test_inputs=test_inputs, static=static
        )
        self.test_inputs = self.mosvgpe_plotter.test_inputs
        self.start_state = self.mode_optimiser.start_state
        self.target_state = self.mode_optimiser.target_state

        if not isinstance(self.test_inputs, tf.Tensor):
            self.test_inputs = tf.constant(self.test_inputs)
        if isinstance(
            self.mode_optimiser.mode_controller.previous_solution, GeodesicTrajectory
        ):
            metric = (
                self.mode_optimiser.mode_controller.previous_solution.manifold.metric(
                    self.test_inputs[:, 0 : self.mode_optimiser.dynamics.state_dim]
                )
            )
            self.metric_trace = tf.linalg.trace(metric)

    def plot_model(self):
        self.plot_trajectories_over_gating_network_gps()
        self.plot_trajectories_over_mixing_probs()

    def plot_trajectories_over_gating_network_gps(self):
        fig = self.mosvgpe_plotter.plot_gating_network_gps()
        self.plot_trajectories_given_fig(fig)
        return fig

    def plot_trajectories_over_mixing_probs(self):
        fig = self.mosvgpe_plotter.plot_mixing_probs()
        self.plot_trajectories_given_fig(fig)
        return fig

    def plot_trajectories_over_metric_trace(self):
        fig = plt.figure(figsize=(self.mosvgpe_plotter.figsize))
        gs = fig.add_gridspec(1, 1, wspace=0.3)
        ax = gs.subplots()
        fig.suptitle("Trace of expected metric $\\text{tr}(\mathbb{E}[G(\mathbf{x})])$")
        self.mosvgpe_plotter.plot_contf(ax, z=self.metric_trace)
        self.plot_trajectories_given_fig(fig)
        return fig

    def plot_trajectories_given_fig(self, fig):
        trajectories = self.generate_trajectories()
        for key in trajectories.keys():
            self.plot_trajectory_given_fig(fig, trajectories[key], key)

    def plot_trajectory_given_fig(self, fig, trajectory, key):
        axs = fig.get_axes()
        if isinstance(axs, np.ndarray):
            for ax in axs.flat:
                self.plot_trajectory_given_ax(ax, trajectory, key)
        elif isinstance(axs, list):
            for ax in axs:
                self.plot_trajectory_given_ax(ax, trajectory, key)
        else:
            self.plot_trajectory_given_ax(axs, trajectory, key)
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

    def plot_start_end_pos_given_ax(self, ax):
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
        ax.annotate(
            "$\mathbf{x}_0$",
            (self.start_state[0, 0] + 0.05, self.start_state[0, 1]),
            horizontalalignment="left",
            verticalalignment="top",
        )
        ax.annotate(
            "$\mathbf{x}_f$",
            (self.target_state[0, 0] - 0.05, self.target_state[0, 1]),
            horizontalalignment="right",
            verticalalignment="bottom",
        )

    def generate_trajectories(self):
        if isinstance(
            self.mode_optimiser.mode_controller.previous_solution, GeodesicTrajectory
        ):
            collocation_trajectory = {
                "collocation": self.mode_optimiser.mode_controller.previous_solution.states
            }
            trajectories = collocation_trajectory
        else:
            dynamics_trajectory = self.mode_optimiser.dynamics_rollout()[0]
            env_trajectory = self.mode_optimiser.env_rollout()
            trajectories = {"env": env_trajectory, "dynamics": dynamics_trajectory}
        return trajectories
