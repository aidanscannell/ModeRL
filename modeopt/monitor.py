#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mogpe.helpers import Plotter2D
from gpflow.monitor import (
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
from mogpe.training.monitor import ImageWithCbarToTensorBoard
from modeopt.mode_opt import ModeOpt

plt.style.use("science")


class ModeOptPlotter:
    def __init__(
        self,
        mode_opt: ModeOpt,
        mogpe_plotter: Plotter2D,
    ):
        self.mogpe_plotter = mogpe_plotter
        self.mode_opt = mode_opt
        self.start_state = self.mode_opt.start_state
        self.target_state = self.mode_opt.target_state
        # self.gp_means, self.gp_vars = self.mogpe_plotter.predict_y(
        #     self.mogpe.plotter.test_inputs, full_cov=False
        # )
        # print("gp_means")
        # print(self.gp_means.shape)
        # print(self.gp_vars.shape)
        # self.h_means, self.h_vars = self.mode_opt.dynamics.gating_gp.predict_f(
        #     self.mogpe.plotter.test_inputs, full_cov=False
        # )
        # self.mode_prob = self.mode_opt.dynamics.gating_gp.predict_y(
        #     self.mogpe.plotter.test_inputs
        # )

    def plot_env_rollout_over_mode_prob(self, fig, axs):
        trajectory = self.mode_opt.env_rollout(self.start_state)
        return self.plot_trajectory_over_mode_prob(fig, axs, trajectory)

    def plot_env_rollout_over_gating_gps(self, fig, axs):
        trajectory = self.mode_opt.env_rollout(self.start_state)
        return self.plot_trajectory_over_gating_gps(fig, axs, trajectory)

    def plot_dynamics_rollout_over_mode_prob(self, fig, axs):
        state_means, state_vars = self.mode_opt.dynamics_rollout(
            self.start_state, start_state_var=None
        )
        return self.plot_trajectory_over_mode_prob(fig, axs, trajectory=state_means)

    def plot_dynamics_rollout_over_gating_gps(self, fig, axs):
        state_means, state_vars = self.mode_opt.dynamics_rollout(
            self.start_state, start_state_var=None
        )
        return self.plot_trajectory_over_gating_gps(fig, axs, trajectory=state_means)

    def plot_trajectory_over_gating_gps(self, fig, axs, trajectory):
        cbars = self.mogpe_plotter.plot_gating_gps(fig, axs)
        self.plot_trajectory_over_axs(fig, axs, trajectory)
        return cbars

    def plot_trajectory_over_mode_prob(self, fig, axs, trajectory):
        cbars = self.mogpe_plotter.plot_gating_network(fig, axs)
        self.plot_trajectory_over_axs(fig, axs, trajectory)
        return cbars

    def plot_trajectory_over_axs(self, fig, axs, trajectory):
        for ax in axs:
            ax.scatter(trajectory[:, 0], trajectory[:, 1], marker="x", color="k")
            # ax.scatter(trajectory[:, 0], trajectory[:, 1])
            ax.scatter(
                self.start_state[0, 0], self.start_state[0, 1], marker="x", color="k"
            )
            ax.scatter(
                self.target_state[0, 0], self.target_state[0, 1], color="k", marker="x"
            )
            ax.annotate(
                "Start $\mathbf{x}_0$",
                (self.start_state[0, 0], self.start_state[0, 1]),
                horizontalalignment="left",
                verticalalignment="top",
            )
            ax.annotate(
                "End $\mathbf{x}_f$",
                (self.target_state[0, 0], self.target_state[0, 1]),
                horizontalalignment="left",
                verticalalignment="top",
            )

    def tf_monitor_task_group(self, log_dir, slow_tasks_period=500):
        num_experts = self.mogpe_plotter.num_experts
        image_task_dynamics_traj = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_dynamics_rollout_over_gating_gps,
            name="dynamics_rollout_over_dynamics_gp",
            fig_kw={"figsize": (8, 2)},
            subplots_kw={"nrows": 1, "ncols": 2},
            # subplots_kw={"nrows": 4, "ncols": 2},
        )
        image_task_env_traj = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_env_rollout_over_gating_gps,
            name="env_rollout_over_dynamics_gp",
            fig_kw={"figsize": (8, 2)},
            # subplots_kw={"nrows": 2, "ncols": 2},
            subplots_kw={"nrows": 1, "ncols": 2},
        )
        image_task_dynamics_traj_prob = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_dynamics_rollout_over_mode_prob,
            name="dynamics_rollout_over_mode_prob",
            # fig_kw={"figsize": (10, 4)},
            # subplots_kw={"nrows": 1, "ncols": 2},
            # subplots_kw={"nrows": 2, "ncols": 1},
            subplots_kw={"nrows": num_experts, "ncols": 1},
        )
        image_task_env_traj_prob = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_env_rollout_over_mode_prob,
            name="env_rollout_over_mode_prob",
            # fig_kw={"figsize": (10, 4)},
            # subplots_kw={"nrows": 1, "ncols": 2},
            # subplots_kw={"nrows": 2, "ncols": 1},
            subplots_kw={"nrows": num_experts, "ncols": 1},
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
    mogpe_plotter = Plotter2D(
        model=mode_opt.dynamics.mosvgpe,
        X=mode_opt.dataset[0],
        Y=mode_opt.dataset[1],
        test_inputs=test_inputs,
    )
    mode_opt_plotter = ModeOptPlotter(mode_opt, mogpe_plotter)

    training_loss_closure = mode_opt.trajectory_optimiser.build_training_loss(
        mode_opt.start_state
    )

    image_tasks = mode_opt_plotter.tf_monitor_task_group(
        log_dir, slow_tasks_period=slow_tasks_period
    )
    slow_tasks = MonitorTaskGroup(image_tasks, period=slow_tasks_period)
    elbo_task = ScalarToTensorBoard(log_dir, training_loss_closure, "negative_elbo")
    policy_task = ModelToTensorBoard(log_dir, mode_opt.policy)
    fast_tasks = MonitorTaskGroup([policy_task, elbo_task], period=fast_tasks_period)
    return Monitor(fast_tasks, slow_tasks)
