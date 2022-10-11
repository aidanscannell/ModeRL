#!/usr/bin/env python3
import os

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from moderl.controllers import ControllerInterface, ExplorativeController
from moderl.custom_types import InputData, State
from moderl.dynamics import ModeRLDynamics
from moderl.rollouts import rollout_trajectory_optimisation_controller_in_env


LABELS = {"env": "Environment", "dynamics": "Dynamics"}
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


def plot_gating_function_variance(ax, dynamics: ModeRLDynamics, test_inputs: InputData):
    _, var = dynamics.mosvgpe.gating_network.predict_h(test_inputs)
    # label = (
    #     "$\mathbb{V}[h_{"
    #     + str(dynamics.desired_mode + 1)
    #     + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
    #     # + str(iteration)
    #     + "}]$"
    # )
    return plot_contf(ax, test_inputs, z=var[:, dynamics.desired_mode])


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


def plot_contf(ax, test_inputs, z, levels=None):
    try:
        contf = ax.tricontourf(
            test_inputs[:, 0],
            test_inputs[:, 1],
            z,
            # 100,
            levels=levels,
        )
    except ValueError:
        # TODO check this works
        contf = ax.tricontourf(
            test_inputs[:, 0],
            test_inputs[:, 1],
            np.ones(z.shape),
            # 100,
            levels=levels,
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
        [0.5],
        colors=["b"],
        linestyles="dashed",
        zorder=50,
    )


def plot_start_end_pos(ax, start_state, target_state):
    bbox = dict(boxstyle="round,pad=0.1", fc="thistle", alpha=1.0)
    if len(start_state.shape) == 1:
        start_state = start_state[tf.newaxis, :]
    if len(target_state.shape) == 1:
        target_state = target_state[tf.newaxis, :]
    ax.annotate(
        r"$\mathbf{s}_0$",
        (start_state[0, 0] + 0.15, start_state[0, 1]),
        horizontalalignment="left",
        verticalalignment="top",
        bbox=bbox,
    )
    ax.annotate(
        r"$\mathbf{s}_f$",
        (target_state[0, 0] + 0.15, target_state[0, 1]),
        horizontalalignment="left",
        verticalalignment="bottom",
        bbox=bbox,
    )
    ax.scatter(start_state[0, 0], start_state[0, 1], marker="x", color="k", s=8.0)
    ax.scatter(
        target_state[0, 0],
        target_state[0, 1],
        color="k",
        marker="x",
        s=8.0,
    )


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


# def plot_data_over_ax(ax, x, y):
#     ax.scatter(
#         x,
#         y,
#         marker="x",
#         color="b",
#         linewidth=0.5,
#         alpha=0.5,
#         label="Observations",
#     )


# def plot_env_cmap(ax, env, test_inputs: InputData):
#     test_states = test_inputs[:, 0:2]
#     mode_probs = []
#     for test_state in test_states:
#         pixel = env.state_to_pixel(test_state)
#         mode_probs.append(env.gating_bitmap[pixel[0], pixel[1]])
#     mode_probs = tf.stack(mode_probs, 0)
#     print("mode_probs")
#     print(mode_probs)
#     print(tf.reduce_min(mode_probs))
#     print(tf.reduce_max(mode_probs))
#     ax.tricontour(test_states[:, 0], test_states[:, 1], mode_probs.numpy())
