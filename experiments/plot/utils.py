#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf
from moderl.controllers import ControllerInterface, ExplorativeController
from moderl.custom_types import InputData, State
from moderl.dynamics import ModeRLDynamics
from moderl.rollouts import rollout_trajectory_optimisation_controller_in_env


LABELS = {
    # "env": r"Environment $\bar{\mathbf{s}}_{\pi_{i}}^{\text{env}}$",
    # "env": r"Environment $\bar{\mathbf{s}}^{\tilde{f}}_{\pi_{i}}$",
    "env": r"Environment $\bar{\mathbf{s}}_{\pi_{i}}$",
    "dynamics": r"Dynamics $\bar{\mathbf{s}}^{f_{k^*}}_{\pi_{i}}$",
    # "dynamics": r"Dynamics $\bar{\mathbf{s}}_{\pi_{i}}^{\text{GP}}$",
}
COLORS = {"env": "c", "dynamics": "m"}
LINESTYLES = {"env": "-", "dynamics": "-"}
MARKERS = {"env": "*", "dynamics": "."}


def get_ExplorativeController_from_id(i, id, wandb_dir) -> ExplorativeController:
    for file in os.listdir(wandb_dir):
        if id in file:
            load_file = os.path.join(
                os.path.join(wandb_dir, file),
                "files/saved-models/controller-optimised-{}-config.json".format(i),
            )
            return ExplorativeController.load(load_file)


def create_test_inputs(num_test: int = 400):
    sqrtN = int(np.sqrt(num_test))
    xx = np.linspace(-4, 3, sqrtN)
    yy = np.linspace(-4, 4, sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    zeros = np.zeros((num_test, 2))
    test_inputs = np.concatenate([test_inputs, zeros], -1)
    return test_inputs


def plot_desired_mixing_prob(
    ax,
    dynamics: ModeRLDynamics,
    test_inputs: InputData,
):
    probs = dynamics.mosvgpe.gating_network.predict_mixing_probs(test_inputs)
    return plot_contf(ax, test_inputs, z=probs[:, dynamics.desired_mode])


def plot_gating_function_mean(ax, dynamics: ModeRLDynamics, test_inputs: InputData):
    mean, _ = dynamics.mosvgpe.gating_network.predict_h(test_inputs)
    # label = (
    #     "$\mathbb{E}[h_{"
    #     + str(dynamics.desired_mode + 1)
    #     + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
    #     # + str(iteration)
    #     + "}]$"
    # )
    return plot_contf(ax, test_inputs, z=mean[:, dynamics.desired_mode])


def plot_gating_function_variance(
    ax, dynamics: ModeRLDynamics, test_inputs: InputData, cmap
):
    _, var = dynamics.mosvgpe.gating_network.predict_h(test_inputs)
    # label = (
    #     "$\mathbb{V}[h_{"
    #     + str(dynamics.desired_mode + 1)
    #     + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
    #     # + str(iteration)
    #     + "}]$"
    # )
    return plot_contf(ax, test_inputs, z=var[:, dynamics.desired_mode], cmap=cmap)


def plot_mode_satisfaction_prob(
    ax, controller: ControllerInterface, test_inputs: InputData
):
    mixing_probs = controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
        test_inputs
    )
    ax.tricontour(
        test_inputs[:, 0],
        test_inputs[:, 1],
        mixing_probs[:, controller.dynamics.desired_mode].numpy(),
        [controller.mode_satisfaction_prob],
        colors=["k"],
    )
    # ax.clabel(CS, inline=True, fontsize=12)


def plot_contf(ax, test_inputs, z, levels=None, cmap=None):
    try:
        contf = ax.tricontourf(
            test_inputs[:, 0],
            test_inputs[:, 1],
            z,
            # 100,
            levels=levels,
            cmap=cmap,
        )
    except ValueError:
        # TODO check this works
        contf = ax.tricontourf(
            test_inputs[:, 0],
            test_inputs[:, 1],
            np.ones(z.shape),
            # 100,
            levels=levels,
            cmap=cmap,
        )
    return contf


def plot_env(ax, env, test_inputs: InputData):
    test_states = test_inputs[:, 0:2]
    mode_probs = []
    for test_state in test_states:
        pixel = env.state_to_pixel(test_state)
        mode_probs.append(env.gating_bitmap[pixel[0], pixel[1]])
    mode_probs = tf.stack(mode_probs, 0)
    return ax.tricontour(
        test_states[:, 0],
        test_states[:, 1],
        mode_probs.numpy(),
        # [0.9],
        [0.5],
        colors=["red"],
        # colors=["gray"],
        # colors=["orange"],
        # colors=["blue"],
        linestyles="dashed",
        alpha=1.0,
        # alpha=0.5,
        zorder=50,
    )


def plot_start_end_pos(ax, start_state, target_state):
    bbox = dict(boxstyle="round,pad=0.1", fc="thistle", alpha=1.0)
    if len(start_state.shape) == 1:
        start_state = start_state[tf.newaxis, :]
    if len(target_state.shape) == 1:
        target_state = target_state[tf.newaxis, :]
    ax.annotate(
        # r"$\mathbf{s}_0$",
        "Start",
        (start_state[0, 0], start_state[0, 1] - 0.3),
        horizontalalignment="right",
        verticalalignment="top",
        bbox=bbox,
    )
    ax.annotate(
        "Target",
        (target_state[0, 0] + 0.15, target_state[0, 1]),
        horizontalalignment="left",
        verticalalignment="bottom",
        bbox=bbox,
    )
    # ax.annotate(
    #     r"$\mathbf{s}_0$",
    #     (start_state[0, 0] + 0.15, start_state[0, 1]),
    #     horizontalalignment="left",
    #     verticalalignment="top",
    #     bbox=bbox,
    # )
    # ax.annotate(
    #     r"$\mathbf{s}_f$",
    #     (target_state[0, 0] + 0.15, target_state[0, 1]),
    #     horizontalalignment="left",
    #     verticalalignment="bottom",
    #     bbox=bbox,
    # )
    # ax.scatter(start_state[0, 0], start_state[0, 1], marker="x", color="k", s=8.0)
    # ax.scatter(target_state[0, 0], target_state[0, 1], color="k", marker="x", s=8.0)
    # ax.scatter(start_state[0, 0], start_state[0, 1], marker="x", color="k", s=8.0)
    ax.scatter(start_state[0, 0], start_state[0, 1], marker="o", color="k", s=40.0)
    ax.scatter(target_state[0, 0], target_state[0, 1], marker="*", color="k", s=250.0)


def plot_trajectories(ax, env, controller: ControllerInterface, target_state: State):
    env_traj = rollout_trajectory_optimisation_controller_in_env(
        env=env, start_state=controller.start_state, controller=controller
    )
    dynamics_traj = controller.rollout_in_dynamics().mean()

    for traj, key in zip([env_traj, dynamics_traj], ["env", "dynamics"]):
        # for traj, key in zip([dynamics_traj], ["dynamics"]):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            label=LABELS[key],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=0.3,
            marker=MARKERS[key],
        )
    plot_start_end_pos(
        ax, start_state=controller.start_state, target_state=target_state
    )


def plot_env_cmap(ax, env, test_inputs: InputData, aspect_ratio: float = 0.75):
    extent = (
        env.observation_spec().minimum[0],
        env.observation_spec().maximum[0],
        env.observation_spec().minimum[1],
        env.observation_spec().maximum[1],
    )
    # gating_bitmap = np.round(env.gating_bitmap) * 0.7
    gating_bitmap = np.round(env.gating_bitmap) * 0.8
    padding = np.ones((100, 100)) * 0.8
    # gating_bitmap = np.round(env.gating_bitmap) * 0.7
    # padding = np.ones((100, 100)) * 0.7
    alpha = 1
    vmin = -0.1
    ax.imshow(
        gating_bitmap,
        extent=extent,
        vmin=vmin,
        vmax=1.0,
        aspect=aspect_ratio,
        alpha=alpha,
    )
    extent = (
        np.min(test_inputs[:, 0]),
        env.observation_spec().minimum[0],
        np.min(test_inputs[:, 1]),
        np.max(test_inputs[:, 1]),
    )
    ax.imshow(
        padding, extent=extent, vmin=vmin, vmax=1.0, aspect=aspect_ratio, alpha=alpha
    )
    extent = (
        np.min(test_inputs[:, 0]),
        np.max(test_inputs[:, 0]),
        env.observation_spec().maximum[1] * 0.9,
        np.max(test_inputs[:, 1]),
    )
    ax.imshow(
        padding, extent=extent, vmin=vmin, vmax=1.0, aspect=aspect_ratio, alpha=alpha
    )
    extent = (
        np.min(test_inputs[:, 0]),
        np.max(test_inputs[:, 0]),
        np.min(test_inputs[:, 1]),
        env.observation_spec().minimum[1],
    )
    ax.imshow(
        padding, extent=extent, vmin=vmin, vmax=1.0, aspect=aspect_ratio, alpha=alpha
    )


# def plot_data_and_traj_over_desired_mixing_prob(
#     ax,
#     dynamics: ModeRLDynamics,
#     controller: ExplorativeController,
#     test_inputs: InputData,
# ):
#     plot_desired_mixing_prob(ax, dynamics=dynamics, test_inputs=test_inputs)
#     plot_data_over_ax(ax, x=dynamics.dataset[0][:, 0], y=dynamics.dataset[0][:, 1])
#     plot_mode_satisfaction_prob(
#         ax,
#         dynamics=dynamics,
#         test_inputs=test_inputs,
#         mode_satisfaction_prob=controller.mode_satisfaction_prob,
#     )


def plot_data_over_ax(ax, X):
    ax.scatter(
        X[:, 0],
        X[:, 1],
        marker="x",
        color="k",
        linewidth=0.5,
        alpha=0.2,
        label="Observations",
    )
