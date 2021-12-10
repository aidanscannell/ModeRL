#!/usr/bin/env python3
import os
from matplotlib import patches

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from gpflow.monitor import (
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
    ImageToTensorBoard,
)
from mogpe.helpers import Plotter2D
from mogpe.helpers.quadcopter_plotter import (
    QuadcopterPlotter,
    init_axis_labels_and_ticks,
)
from mogpe.training.monitor import ImageWithCbarToTensorBoard

from modeopt.mode_opt import ModeOpt
from modeopt.rollouts import (
    rollout_policy_in_dynamics,
    rollout_controls_in_dynamics,
    rollout_controls_in_env,
)

plt.style.use("science")
# matplotlib.rcParams.update({"font.size": 10})
matplotlib.rcParams.update({"font.size": 7})

expert_colors = ["c", "m", "y"]
colors = {"initial": "c", "optimised": "m"}
linestyles = {"initial": "x", "optimised": "*"}
markers = {"initial": "x", "optimised": "*"}

styles = {"env": "c-.", "dynamics": "m-s"}
labels = {"env": "Environment", "dynamics": "Dynamics"}


def sub_dict(somedict, somekeys, default=None):
    return dict([(k, somedict.get(k, default)) for k in somekeys])


def plot_patch(ax, mean, var):
    ax.add_patch(
        patches.Ellipse(
            (mean[0], mean[1]),
            var[0] * 100000000,
            var[1] * 100000000,
            facecolor="none",
            edgecolor="b",
            linewidth=0.1,
            alpha=0.6,
        )
    )


class Trajectory:
    def __init__(self, mode_opt, control_means=None, control_vars=None):
        self.mode_opt = mode_opt
        self.num_modes = mode_opt.dynamics.mosvgpe.num_experts

        if control_means is None and control_vars is None:
            self.control_means, self.control_vars = self.mode_opt.policy()
        else:
            self.control_means = control_means
            self.control_vars = control_vars

        # Rollout policy in dynamics
        self.state_means, self.state_vars = rollout_controls_in_dynamics(
            dynamics=self.mode_opt.dynamics,
            start_state=self.mode_opt.start_state,
            control_means=self.control_means,
            control_vars=self.control_vars,
        )

        # Rollout policy in env
        # self.env_trajectory = self.mode_opt.env_rollout(self.mode_opt.start_state)
        self.env_trajectory = rollout_controls_in_env(
            env=self.mode_opt.env,
            start_state=self.mode_opt.start_state,
            controls=self.control_means,
        )

        # Predict mode probabilities over trajectory
        state_control_inputs = tf.concat(
            [self.state_means[1:, :], self.control_means], -1
        )
        self.mode_probs = self.mode_opt.dynamics.mosvgpe.predict_mixing_probs(
            state_control_inputs
        )
        mode_prob_uncertain = self.mode_opt.dynamics.predict_mode_probability(
            state_mean=self.state_means[1:, :],
            control_mean=self.control_means,
            state_var=self.state_vars[1:, :],
            control_var=self.control_vars * 0.0,
        )
        mode_prob_uncertain = tf.reshape(mode_prob_uncertain, (-1, 1))
        self.mode_prob_uncertain_broadcast = tf.broadcast_to(
            mode_prob_uncertain, (self.control_means.shape[0], self.num_modes)
        )

        # Predict gating function over trajectory
        self.h_mean, self.h_var = self.mode_opt.dynamics.gating_gp.predict_f(
            state_control_inputs
        )
        (
            self.h_mean_unc,
            self.h_var_unc,
        ) = self.mode_opt.dynamics.uncertain_predict_gating(
            self.state_means[1:, :],
            self.control_means,
            state_var=self.state_vars[1:, :],
            control_var=self.control_vars * 0.0,
        )


class ModeOptPlotter:
    def __init__(
        self,
        mode_opt: ModeOpt,
        # mogpe_plotter: Plotter2D,
        mogpe_plotter: QuadcopterPlotter,
        control_means_init: ModeOpt = None,
        control_vars_init: ModeOpt = None,
    ):
        self.mogpe_plotter = mogpe_plotter
        self.figsize = self.mogpe_plotter.figsize
        self.mode_opt = mode_opt
        self.desired_mode = self.mode_opt.desired_mode
        self.start_state = self.mode_opt.start_state
        self.target_state = self.mode_opt.target_state
        self.num_modes = mode_opt.dynamics.mosvgpe.num_experts
        self.times = np.arange(0, self.mode_opt.horizon)

        self.plot_gating_gps_flag = True
        self.plot_probs_flag = True

        self.trajectory_opt = Trajectory(mode_opt)
        self.trajectories = {
            # "init_env": self.trajectory_init.env_trajectory,
            # "init_dynamics": self.trajectory_init.state_means,
            "opt_env": self.trajectory_opt.env_trajectory,
            "opt_dynamics": self.trajectory_opt.state_means,
        }
        # self.trajectory_vars = {
        #     "init_env": None,
        #     "opt_env": None,
        #     "init_dynamics": self.trajectory_init.state_vars,
        #     "opt_dynamics": self.trajectory_opt.state_vars,
        # }
        self.trajectory_labels = {
            "init_env": "$\\bar{\mathbf{x}}_0$ Environment",
            "opt_env": "Environment",
            "init_dynamics": "$\\bar{\mathbf{x}}_0$ Dynamics",
            "opt_dynamics": "Dynamics $f_{" + str(self.desired_mode + 1) + "}$",
        }
        self.trajectory_colors = {
            "init_env": "c",
            "opt_env": "c",
            "init_dynamics": "m",
            "opt_dynamics": "m",
        }
        self.trajectory_linestyles = {
            "init_env": "-.",
            "opt_env": "-.",
            "init_dynamics": ":",
            "opt_dynamics": ":",
        }
        self.trajectory_markers = {
            "init_env": "^",
            "opt_env": "^",
            "init_dynamics": ".",
            "opt_dynamics": ".",
        }

        # Rollout initial policy in dynamics
        # if control_means_init is not None:
        #     self.trajectory_init = Trajectory(
        #         mode_opt,
        #         control_means=control_means_init,
        #         control_vars=control_vars_init,
        #     )
        #     self.trajectories["init_env"] = self.trajectory_init.env_trajectory
        #     self.trajectories["init_dynamics"] = self.trajectory_init.state_means

    def plot_model(self, save_dir):
        try:
            os.makedirs(save_dir)
        except:
            print("save_dir already exists")

        save_filename = os.path.join(save_dir, "trajectories_over_desired_prob.pdf")
        self.plot_trajectories_over_probs(
            trajectories=self.trajectories,
            desired_mode=self.desired_mode,
            save_filename=save_filename,
        )
        save_filename = os.path.join(
            save_dir, "trajectories_over_desired_gating_gp.pdf"
        )
        self.plot_trajectories_over_gating_gps(
            trajectories=self.trajectories,
            desired_mode=self.desired_mode,
            save_filename=save_filename,
        )
        # save_filename = os.path.join(
        #     save_dir, "dynamics_rollouts_over_desired_gating_gp.pdf"
        # )
        # self.plot_trajectories_over_gating_gps(
        #     trajectories=sub_dict(self.trajectories, ["init_dynamics", "opt_dynamics"]),
        #     desired_mode=self.desired_mode,
        #     save_filename=save_filename,
        # )

        # save_filename = os.path.join(save_dir, "mode_prob_vs_time.pdf")
        # self.plot_prob_vs_time(
        #     save_filename=save_filename,
        #     probs=self.mode_probs,
        #     desired_mode=self.desired_mode,
        #     ylabel="$q ( \\alpha_t =k \mid \mathbf{x}_t ) \\approx \Pr ( \\alpha_t =k \mid \mathbf{x}_t )$",
        # )
        # save_filename = os.path.join(save_dir, "mode_probs_vs_time.pdf")
        # self.plot_prob_vs_time(
        #     save_filename=save_filename,
        #     probs=self.mode_probs,
        #     ylabel="$q ( \\alpha_t =k \mid \mathbf{x}_t ) \\approx \Pr ( \\alpha_t =k \mid \mathbf{x}_t )$",
        # )
        # save_filename = os.path.join(save_dir, "uncertain_mode_prob_vs_time.pdf")
        # self.plot_prob_vs_time(
        #     save_filename=save_filename,
        #     probs=self.mode_prob_uncertain_broadcast,
        #     desired_mode=self.desired_mode,
        #     ylabel="$q ( \\alpha_t =k) \\approx \int \Pr ( \\alpha_t =k \mid \mathbf{x}_t ) q(\mathbf{x}_t) \\text{d}\mathbf{x}_t$",
        # )
        # save_filename = os.path.join(save_dir, "variance_vs_time.pdf")
        # self.plot_variance_vs_time(save_filename=save_filename)
        # # save_filename = os.path.join(save_dir, "gating_gp_vs_time.pdf")
        # # self.plot_gating_gp_vs_time(save_filename=save_filename)
        # save_filename = os.path.join(save_dir, "uncertain_gating_gp_vs_time.pdf")
        # self.plot_gating_gp_vs_time(save_filename=save_filename)

    def plot_jacobian_mean(self):
        from geoflow.plotting import plot_svgp_jacobian_mean

        svgp = self.mode_opt.dynamics.gating_gp
        plot_svgp_jacobian_mean(svgp)
        plt.show()

    def create_fig_axs_plot_desired_prob(self, desired_mode=True):
        if desired_mode:
            fig = plt.figure(figsize=(self.figsize[0] / 2, self.figsize[1] / 4))
            gs = fig.add_gridspec(1, 1, wspace=0.3)
            axs = gs.subplots(sharex=True, sharey=True)
            axs = init_axis_labels_and_ticks(axs)
            desired_mode = self.desired_mode
        else:
            fig, axs = self.mogpe_plotter.create_fig_axs_plot_mixing_probs()
            desired_mode = None
        self.mogpe_plotter.plot_mixing_probs_given_fig_axs(
            fig, axs, desired_mode=desired_mode
        )
        return fig, axs

    def create_fig_axs_plot_desired_gating_gps(self, desired_mode=True):
        if desired_mode:
            fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] / 4))
            gs = fig.add_gridspec(1, 2, wspace=0.3)
            axs = gs.subplots(sharex=True, sharey=True)
            axs = init_axis_labels_and_ticks(axs)
            desired_mode = self.desired_mode
        else:
            fig, axs = self.mogpe_plotter.create_fig_axs_plot_gating_gps()
            desired_mode = None
        self.mogpe_plotter.plot_gating_gps_given_fig_axs(
            fig, axs, desired_mode=desired_mode
        )
        return fig, axs

    def create_fig_axs_plot_vs_time(self):
        fig = plt.figure(figsize=(self.figsize[0] / 2, self.figsize[1] / 2))
        gs = fig.add_gridspec(1, 1, wspace=0.3)
        ax = gs.subplots(sharex=True, sharey=True)
        ax.set_xlabel("$t$")
        return fig, ax

    def plot_trajectories_over_probs(
        self, trajectories, desired_mode=True, save_filename=None
    ):
        fig, axs = self.create_fig_axs_plot_desired_prob(desired_mode=desired_mode)
        self.plot_trajectories_given_fig_axs(fig, axs, trajectories)
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_trajectories_over_gating_gps(
        self, trajectories, desired_mode=True, save_filename=None
    ):
        fig, axs = self.create_fig_axs_plot_desired_gating_gps(
            desired_mode=desired_mode
        )
        self.plot_trajectories_given_fig_axs(fig, axs, trajectories)
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_trajectories_over_probs_given_fig_axs(
        self, fig, axs, trajectories=None, desired_mode=None
    ):
        self.mogpe_plotter.plot_mixing_probs_given_fig_axs(
            fig, axs, desired_mode=desired_mode
        )
        self.plot_trajectories_given_fig_axs(fig, axs, trajectories)

    def plot_trajectories_over_gating_gps_given_fig_axs(
        self, fig, axs, trajectories=None, desired_mode=None
    ):
        self.mogpe_plotter.plot_gating_gps_given_fig_axs(
            fig,
            axs,
            desired_mode=desired_mode,
        )
        self.plot_trajectories_given_fig_axs(fig, axs, trajectories)

    def plot_trajectories_given_fig_axs(self, fig, axs, trajectories=None):
        if trajectories is None:
            trajectories = self.generate_trajectories()

        # del trajectories["opt_env"]

        def plot_fn(ax):
            for key in trajectories.keys():
                # ax.scatter(
                #     trajectories[key][:, 0],
                #     trajectories[key][:, 1],
                #     label=self.trajectory_labels[key],
                #     color=self.trajectory_colors[key],
                #     # linestyle=self.trajectory_linestyles[key],
                #     marker=self.trajectory_markers[key],
                # )
                ax.plot(
                    trajectories[key][:, 0],
                    trajectories[key][:, 1],
                    label=self.trajectory_labels[key],
                    color=self.trajectory_colors[key],
                    linestyle=self.trajectory_linestyles[key],
                    linewidth=0.3,
                    marker=self.trajectory_markers[key],
                )
                # if self.trajectory_vars[key] is not None:
                # for mean, var in zip(trajectories[key], self.trajectory_vars[key]):
                #     plot_patch(ax, mean, var)
            self.plot_start_end_pos_given_fig_ax(fig, ax)

        if isinstance(axs, np.ndarray):
            for ax in axs.flat:
                plot_fn(ax)
            axs.flatten()[-1].legend()
        else:
            plot_fn(axs)
            axs.legend()

    def generate_trajectories(self):
        env_trajectory = self.mode_opt.env_rollout(self.start_state)
        dynamics_trajectory, dynamics_trajectory_vars = self.mode_opt.dynamics_rollout(
            self.start_state
        )
        return {"opt_env": env_trajectory, "opt_dynamics": dynamics_trajectory}

    def plot_start_end_pos_given_fig_ax(self, fig, ax):
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
            (self.start_state[0, 0], self.start_state[0, 1]),
            horizontalalignment="left",
            verticalalignment="top",
        )
        ax.annotate(
            "$\mathbf{x}_f$",
            (self.target_state[0, 0], self.target_state[0, 1]),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    def plot_prob_vs_time(
        self, probs, save_filename=None, desired_mode=None, ylabel=""
    ):
        fig, axs = self.create_fig_axs_plot_vs_time()
        self.plot_prob_vs_time_given_fig_axs(
            fig, axs, probs=probs, desired_mode=desired_mode, ylabel=ylabel
        )
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_variance_vs_time(self, save_filename=None):
        fig, axs = self.create_fig_axs_plot_vs_time()
        self.plot_variance_vs_time_given_fig_axs(fig, axs)
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_gating_gp_vs_time(self, save_filename=None):
        fig, axs = self.create_fig_axs_plot_vs_time()
        self.plot_gating_gp_vs_time_given_fig_axs(fig, axs)
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_prob_vs_time_given_fig_axs(
        self, fig, ax, probs, desired_mode=None, ylabel=None
    ):
        if desired_mode is None:
            for k in range(self.num_modes):
                ax.plot(self.times, probs[:, k], label="k=" + str(k + 1))
        else:
            ax.plot(
                self.times,
                probs[:, desired_mode],
                label="k=" + str(desired_mode + 1),
            )
        ax.set_ylabel(ylabel)
        ax.legend()

    def plot_variance_vs_time_given_fig_axs(self, fig, ax, label=None):
        ax.plot(self.times, self.h_var)
        ax.set_ylabel("$\mathbb{V}[h_k(\mathbf{x}_t)]$")

    def plot_gating_gp_vs_time_given_fig_axs(self, fig, ax, label=None):
        print("self.h_mean")
        print(self.h_mean.shape)
        print(self.h_var.shape)
        # mean = self.h_mean[:, 0]
        # var = self.h_var[:, 0]
        mean = self.h_mean_unc[:, 0]
        var = self.h_var_unc[:, 0]
        var_label = "$\mathbb{V}[h_k(\mathbf{x}_t)]$"
        mean_label = "$\mathbb{E}[h_k(\mathbf{x}_t)]$"
        ax.plot(self.times, mean, label=mean_label)
        ax.fill_between(
            self.times,
            mean - 1.96 * np.sqrt(var),
            mean + 1.96 * np.sqrt(var),
            # color=color,
            alpha=0.2,
            label=var_label,
        )
        ax.legend()

    def tf_monitor_task_group(self, log_dir, slow_tasks_period=500):
        # figsize = self.mogpe_plotter.figsize
        figsize = (6, 4)
        num_experts = self.mogpe_plotter.num_experts
        image_task_gating_gps_traj = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_trajectories_over_gating_gps_given_fig_axs,
            name="trajectories_over_gating_gps",
            fig_kw={"figsize": (figsize[0], figsize[1] / 4)},
            subplots_kw={"nrows": num_experts, "ncols": 2},
        )
        image_task_probs_traj = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_trajectories_over_probs_given_fig_axs,
            name="trajectories_over_mode_probs",
            fig_kw={"figsize": (figsize[0], figsize[1] / 4)},
            subplots_kw={"nrows": 1, "ncols": num_experts},
        )
        image_tasks = [image_task_gating_gps_traj, image_task_probs_traj]
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
        static=True,
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
