#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from gpflow.monitor import (
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
from mogpe.helpers import Plotter2D
from mogpe.helpers.quadcopter_plotter import (
    QuadcopterPlotter,
    init_axis_labels_and_ticks,
)
from mogpe.training.monitor import ImageWithCbarToTensorBoard

from modeopt.mode_opt import ModeOpt

plt.style.use("science")
matplotlib.rcParams.update({"font.size": 10})

expert_colors = ["c", "m", "y"]
colors = {"initial": "c", "optimised": "m"}
linestyles = {"initial": "x", "optimised": "*"}
markers = {"initial": "x", "optimised": "*"}

styles = {"env": "c-.", "dynamics": "m-s"}
labels = {"env": "Environment", "dynamics": "Dynamics"}

# svgp_linestyle = "--"
# svgp_linestyle = "-"


class ModeOptPlotter:
    def __init__(
        self,
        mode_opt: ModeOpt,
        # mogpe_plotter: Plotter2D,
        mogpe_plotter: QuadcopterPlotter,
    ):
        self.mogpe_plotter = mogpe_plotter
        self.figsize = self.mogpe_plotter.figsize
        self.mode_opt = mode_opt
        self.desired_mode = self.mode_opt.desired_mode
        self.start_state = self.mode_opt.start_state
        self.target_state = self.mode_opt.target_state

    def plot_model(self, save_dir):
        try:
            os.makedirs(save_dir)
        except:
            print("save_dir already exists")
        # save_filename = os.path.join(save_dir, "env_rollout_over_prob.pdf")
        # self.plot_rollout_over_prob(save_filename=save_filename, dynamics=False)
        # save_filename = os.path.join(save_dir, "env_rollout_over_desired_prob.pdf")
        # self.plot_rollout_over_prob(
        #     save_filename=save_filename, desired_mode=self.desired_mode
        # )
        # save_filename = os.path.join(save_dir, "env_rollout_over_gating_gps.pdf")
        # self.plot_rollout_over_gating_gps(save_filename=save_filename, dynamics=False)
        # save_filename = os.path.join(
        #     save_dir, "env_rollout_over_desired_gating_gps.pdf"
        # )
        # self.plot_rollout_over_gating_gps(
        #     save_filename=save_filename, desired_mode=self.desired_mode, dynamics=False
        # )

        # save_filename = os.path.join(save_dir, "dynamics_rollout_over_prob.pdf")
        # self.plot_dynamics_rollout_over_prob(save_filename=save_filename)
        # save_filename = os.path.join(save_dir, "dynamics_rollout_over_desired_prob.pdf")
        # self.plot_dynamics_rollout_over_prob(
        #     save_filename=save_filename, desired_mode=self.desired_mode
        # )
        # save_filename = os.path.join(
        #     save_dir, "dynamics_rollout_over_desired_gating_gps.pdf"
        # )
        # self.plot_dynamics_rollout_over_gating_gps(
        #     save_filename=save_filename, desired_mode=self.desired_mode
        # )
        # save_filename = os.path.join(save_dir, "dynamics_rollout_over_gating_gps.pdf")
        # self.plot_dynamics_rollout_over_gating_gps(save_filename=save_filename)
        # # save_filename = os.path.join(
        # #     save_dir, "env_and_dynamics_rollout_over_gating_gps.pdf"
        # # )
        # # self.plot_env_and_dynamics_rollout_over_gating_gps(save_filename=save_filename)

        save_filename = os.path.join(
            save_dir, "env_and_dynamics_rollout_over_desired_prob.pdf"
        )
        self.plot_env_and_dynamics_rollout_over_prob(
            save_filename=save_filename, desired_mode=self.desired_mode
        )
        save_filename = os.path.join(
            save_dir, "env_and_dynamics_rollout_over_desired_gating_gps.pdf"
        )
        self.plot_env_and_dynamics_rollout_over_gating_gps(
            save_filename=save_filename, desired_mode=self.desired_mode
        )

    def create_fig_axs_plot_desired_prob(self):
        fig = plt.figure(figsize=(self.figsize[0] / 2, self.figsize[1] / 4))
        gs = fig.add_gridspec(1, 1, wspace=0.3)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def create_fig_axs_plot_desired_gating_gps(self):
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] / 4))
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def plot_rollout_over_prob(
        self, save_filename=None, dynamics=True, desired_mode=None
    ):
        if desired_mode is None:
            fig, axs = self.mogpe_plotter.create_fig_axs_plot_prob()
        else:
            fig, axs = self.create_fig_axs_plot_desired_prob()
        if dynamics:
            self.plot_dynamics_rollout_over_prob_given_fig_axs(
                fig, axs, desired_mode=desired_mode
            )
        else:
            self.plot_env_rollout_over_prob_given_fig_axs(
                fig, axs, desired_mode=desired_mode
            )
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_rollout_over_gating_gps(
        self, save_filename=None, dynamics=True, desired_mode=None
    ):
        if desired_mode is None:
            fig, axs = self.mogpe_plotter.create_fig_axs_plot_gating_gps()
        else:
            fig, axs = self.create_fig_axs_plot_desired_gating_gps()
        if dynamics:
            self.plot_dynamics_rollout_over_gating_gps_given_fig_axs(
                fig, axs, desired_mode=desired_mode
            )
        else:
            self.plot_env_rollout_over_gating_gps_given_fig_axs(
                fig, axs, desired_mode=desired_mode
            )
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_env_and_dynamics_rollout_over_gating_gps(
        self, save_filename=None, desired_mode=None
    ):
        fig, axs = self.create_fig_axs_plot_desired_gating_gps()
        self.plot_env_rollout_over_gating_gps_given_fig_axs(
            fig, axs, desired_mode=desired_mode
        )
        self.plot_dynamics_rollout_over_gating_gps_given_fig_axs(
            fig, axs, desired_mode=desired_mode
        )
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_env_and_dynamics_rollout_over_prob(
        self, save_filename=None, desired_mode=None
    ):
        if desired_mode is None:
            fig, axs = self.mogpe_plotter.create_fig_axs_plot_prob()
        else:
            fig, axs = self.create_fig_axs_plot_desired_prob()
        self.plot_env_rollout_over_prob_given_fig_axs(
            fig, axs, desired_mode=desired_mode
        )
        self.plot_dynamics_rollout_over_prob_given_fig_axs(
            fig, axs, desired_mode=desired_mode
        )
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_env_rollout_over_prob_given_fig_axs(self, fig, axs, desired_mode=None):
        trajectory = self.mode_opt.env_rollout(self.start_state)
        return self.plot_trajectory_over_prob(
            fig,
            axs,
            trajectory,
            desired_mode=desired_mode,
            label="env",
        )
        # cbars = []
        # if isinstance(axs, np.ndarray):
        #     for ax in axs.flat:
        #         cbars.append(
        #             self.plot_trajectory_over_prob(
        #                 fig,
        #                 ax,
        #                 trajectory,
        #                 desired_mode=desired_mode,
        #                 label="env",
        #             )
        #         )
        # else:
        #     cbars.append(
        #         self.plot_trajectory_over_prob(
        #             fig, axs, trajectory, desired_mode=desired_mode, label="env"
        #         )
        #     )
        # return cbars

    def plot_env_rollout_over_gating_gps_given_fig_axs(
        self, fig, axs, desired_mode=None
    ):
        trajectory = self.mode_opt.env_rollout(self.start_state)
        return self.plot_trajectory_over_gating_gps(
            fig, axs, trajectory, desired_mode=desired_mode, label="env"
        )

    def plot_dynamics_rollout_over_prob_given_fig_axs(
        self, fig, axs, desired_mode=None
    ):
        state_means, state_vars = self.mode_opt.dynamics_rollout(
            self.start_state, start_state_var=None
        )
        return self.plot_trajectory_over_prob(
            fig,
            axs,
            trajectory=state_means,
            desired_mode=desired_mode,
            label="dynamics",
        )
        # cbars = []
        # if isinstance(axs, np.ndarray):
        #     for ax in axs.flat:
        #         cbars.append(
        #             self.plot_trajectory_over_prob(
        #                 fig,
        #                 ax,
        #                 trajectory=state_means,
        #                 desired_mode=desired_mode,
        #                 label="dynamics",
        #             )
        #         )
        # else:
        #     cbars.append(
        #         self.plot_trajectory_over_prob(
        #             fig,
        #             axs,
        #             trajectory=state_means,
        #             desired_mode=desired_mode,
        #             label="dynamics",
        #         )
        #     )
        # return cbars

    def plot_dynamics_rollout_over_gating_gps_given_fig_axs(
        self, fig, axs, desired_mode=None
    ):
        state_means, state_vars = self.mode_opt.dynamics_rollout(
            self.start_state, start_state_var=None
        )
        return self.plot_trajectory_over_gating_gps(
            fig,
            axs,
            trajectory=state_means,
            desired_mode=desired_mode,
            label="dynamics",
        )

    def plot_trajectory_over_gating_gps(
        self, fig, axs, trajectory, desired_mode=None, label=None
    ):
        # cbars = self.mogpe_plotter.plot_gating_gps(fig, axs)
        cbars = self.mogpe_plotter.plot_gating_gps_given_fig_axs(
            fig, axs, desired_mode=desired_mode
        )
        self.plot_trajectory_over_axs(fig, axs, trajectory, label=label)
        return cbars

    def plot_trajectory_over_prob(
        self, fig, axs, trajectory, desired_mode=None, label=None
    ):
        # cbars = self.mogpe_plotter.plot_gating_network(fig, axs)
        cbars = self.mogpe_plotter.plot_mixing_probs_given_fig_axs(
            fig, axs, desired_mode=desired_mode
        )
        self.plot_trajectory_over_axs(fig, axs, trajectory, label=label)
        return cbars

    def plot_trajectory_over_axs(self, fig, axs, trajectory, label=None):
        def plot_fn(ax):
            # ax.scatter(trajectory[:, 0], trajectory[:, 1], marker=".", color=color)
            # , label=labels["env"]
            # ax.plot(trajectory[:, 0], trajectory[:, 1], marker=".", color=color)
            if label is None:
                style = ""
                traj_label = ""
            else:
                style = styles[label]
                traj_label = labels[label]
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                style,
                label=traj_label,
                markersize=2.0,
            )
            ax.scatter(
                self.start_state[0, 0],
                self.start_state[0, 1],
                marker="x",
                color="k",
                s=8.0,
            )
            ax.scatter(
                self.target_state[0, 0],
                self.target_state[0, 1],
                color="k",
                marker="x",
                s=8.0,
            )
            ax.annotate(
                # "Start $\mathbf{x}_0$",
                "$\mathbf{x}_0$",
                (self.start_state[0, 0], self.start_state[0, 1]),
                horizontalalignment="left",
                verticalalignment="top",
            )
            ax.annotate(
                # "End $\mathbf{x}_f$",
                "$\mathbf{x}_f$",
                (self.target_state[0, 0], self.target_state[0, 1]),
                # horizontalalignment="left",
                # verticalalignment="top",
                horizontalalignment="left",
                verticalalignment="bottom",
            )

        if isinstance(axs, np.ndarray):
            for ax in axs.flat:
                plot_fn(ax)
            axs.flatten()[-1].legend()
        else:
            plot_fn(axs)
            axs.legend()

    def tf_monitor_task_group(self, log_dir, slow_tasks_period=500):
        # figsize = self.mogpe_plotter.figsize
        figsize = (6, 4)
        num_experts = self.mogpe_plotter.num_experts
        image_task_dynamics_traj = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_dynamics_rollout_over_gating_gps_given_fig_axs,
            name="dynamics_rollout_over_dynamics_gp",
            fig_kw={"figsize": (figsize[0], figsize[1] / 4)},
            # subplots_kw={"nrows": 1, "ncols": 2},
            # subplots_kw={"nrows": 4, "ncols": 2},
            subplots_kw={"nrows": num_experts, "ncols": 2},
        )
        image_task_env_traj = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_env_rollout_over_gating_gps_given_fig_axs,
            name="env_rollout_over_dynamics_gp",
            fig_kw={"figsize": (figsize[0], figsize[1] / 2)},
            # subplots_kw={"nrows": 2, "ncols": 2},
            # subplots_kw={"nrows": 1, "ncols": 2},
            subplots_kw={"nrows": num_experts, "ncols": 2},
        )
        image_task_dynamics_traj_prob = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_dynamics_rollout_over_prob_given_fig_axs,
            name="dynamics_rollout_over_prob",
            fig_kw={"figsize": (figsize[0], figsize[1] / 4)},
            # fig_kw={"figsize": (10, 4)},
            # subplots_kw={"nrows": 1, "ncols": 2},
            # subplots_kw={"nrows": 2, "ncols": 1},
            # subplots_kw={"nrows": num_experts, "ncols": 1},
            subplots_kw={"nrows": 1, "ncols": num_experts},
        )
        image_task_env_traj_prob = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_env_rollout_over_prob_given_fig_axs,
            name="env_rollout_over_prob",
            fig_kw={"figsize": (figsize[0], figsize[1] / 4)},
            # fig_kw={"figsize": (10, 4)},
            # subplots_kw={"nrows": 1, "ncols": 2},
            # subplots_kw={"nrows": 2, "ncols": 1},
            # subplots_kw={"nrows": num_experts, "ncols": 1},
            subplots_kw={"nrows": 1, "ncols": num_experts},
        )
        image_tasks = [
            image_task_dynamics_traj,
            image_task_env_traj,
            image_task_dynamics_traj_prob,
            image_task_env_traj_prob,
        ]
        return MonitorTaskGroup(image_tasks, period=slow_tasks_period)


def create_test_inputs(X, Y, num_test=400, factor=1.2, const=0.0):
    sqrtN = int(np.sqrt(num_test))
    x_min = tf.reduce_min(X[:, 0])
    y_min = tf.reduce_min(X[:, 1])
    x_max = tf.reduce_max(X[:, 0])
    y_max = tf.reduce_max(X[:, 1])
    x_max = 3.0
    y_max = 3.0
    xx = np.linspace(x_min * factor, x_max * factor, sqrtN)
    yy = np.linspace(y_min * factor, y_max * factor, sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    test_inputs = np.concatenate(
        [
            test_inputs,
            np.ones((test_inputs.shape[0], X.shape[1] - 2)) * const,
        ],
        -1,
    )
    return test_inputs


def init_ModeOpt_monitor(
    mode_opt: ModeOpt,
    log_dir: str,
    fast_tasks_period: int = 10,
    slow_tasks_period: int = 500,
):
    test_inputs = create_test_inputs(*mode_opt.dataset)
    # mogpe_plotter = Plotter2D(
    #     model=mode_opt.dynamics.mosvgpe,
    #     X=mode_opt.dataset[0],
    #     Y=mode_opt.dataset[1],
    #     test_inputs=test_inputs,
    # )
    mogpe_plotter = QuadcopterPlotter(
        model=mode_opt.dynamics.mosvgpe,
        X=mode_opt.dataset[0],
        Y=mode_opt.dataset[1],
        test_inputs=test_inputs,
    )
    mode_opt_plotter = ModeOptPlotter(mode_opt, mogpe_plotter)

    # training_loss_closure = mode_opt.trajectory_optimiser.build_training_loss(
    #     mode_opt.start_state
    # )

    image_tasks = mode_opt_plotter.tf_monitor_task_group(
        log_dir, slow_tasks_period=slow_tasks_period
    )
    slow_tasks = MonitorTaskGroup(image_tasks, period=slow_tasks_period)
    # elbo_task = ScalarToTensorBoard(log_dir, training_loss_closure, "negative_elbo")
    policy_task = ModelToTensorBoard(log_dir, mode_opt.policy)
    # fast_tasks = MonitorTaskGroup([policy_task, elbo_task], period=fast_tasks_period)
    fast_tasks = MonitorTaskGroup([policy_task], period=fast_tasks_period)
    return Monitor(fast_tasks, slow_tasks)
