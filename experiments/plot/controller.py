#!/usr/bin/env python3
import tikzplotlib
import os
from functools import partial
from typing import Callable, List

import gpflow as gpf
import hydra
import keras
import matplotlib
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import palettable
import simenvs
import tensorflow as tf
from matplotlib import patches
from moderl.controllers import explorative_controller
from moderl.controllers.controller import ControllerInterface
from moderl.controllers.explorative_controller import ExplorativeController
from moderl.dynamics import ModeRLDynamics
from moderl.dynamics.dynamics import ModeRLDynamics
from mosvgpe.custom_types import InputData, Dataset
from mosvgpe.mixture_of_experts import MixtureOfSVGPExperts
from mpl_toolkits.axes_grid1 import make_axes_locatable

import wandb

plt.style.use("seaborn-paper")
CMAP = palettable.scientific.sequential.Bilbao_15.mpl_colormap
from experiments.plot.utils import plot_contf, PlottingCallback
from moderl.rollouts import (
    rollout_trajectory_optimisation_controller_in_env,
    rollout_ExplorativeController_in_ModeRLDynamics,
)

LABELS = {"env": "Environment", "dynamics": "Dynamics"}
COLORS = {"env": "c", "dynamics": "m"}
LINESTYLES = {"env": "-", "dynamics": "-"}
MARKERS = {"env": "*", "dynamics": "."}


def plot_trajectories(
    ax, env, dynamics: ModeRLDynamics, controller: ControllerInterface
):
    env_traj = rollout_trajectory_optimisation_controller_in_env(
        env=env, start_state=controller.start_state, controller=controller
    )
    dynamics_traj = rollout_ExplorativeController_in_ModeRLDynamics(
        dynamics=dynamics, controller=controller, start_state=controller.start_state
    )

    for traj, key in zip([env_traj, dynamics_traj], ["env", "dynamics"]):
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
        ax, start_state=controller.start_state, target_state=controller.target_state
    )


def plot_start_end_pos(ax, start_state, target_state):
    # def plot_start_end_pos(ax, start_state, target_state, bbox=False):
    # if bbox:
    #     bbox = dict(boxstyle="round,pad=0.1", fc="thistle", alpha=1.0)
    # else:
    #     bbox = None
    bbox = dict(boxstyle="round,pad=0.1", fc="thistle", alpha=1.0)
    if len(start_state.shape) == 1:
        start_state = start_state[tf.newaxis, :]
    if len(target_state.shape) == 1:
        target_state = target_state[tf.newaxis, :]
    ax.annotate(
        "$\mathbf{s}_0$",
        (start_state[0, 0] + 0.1, start_state[0, 1]),
        horizontalalignment="left",
        verticalalignment="top",
        bbox=bbox,
    )
    ax.annotate(
        "$\mathbf{s}_f$",
        (target_state[0, 0] - 0.1, target_state[0, 1]),
        horizontalalignment="right",
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


def plot_gating_network_gps(dynamics: ModeRLDynamics, test_inputs: InputData):
    fig = plt.figure()
    gs = fig.add_gridspec(dynamics.mosvgpe.num_experts, 2)
    axs = gs.subplots()
    mean, var = dynamics.mosvgpe.gating_network.predict_h(test_inputs)
    for k in range(dynamics.mosvgpe.num_experts):
        label = (
            "$\mathbb{E}[h_{"
            + str(dynamics.desired_mode + 1)
            + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
            # + str(iteration)
            + "}]$"
        )
        plot_contf(axs[k, 0], test_inputs, z=mean[:, k])
        # plot_contf(axs[0], test_inputs, z=mean[:, dynamics.desired_mode], label=label)
        label = (
            "$\mathbb{V}[h_{"
            + str(dynamics.desired_mode + 1)
            + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
            # + str(iteration)
            + "}]$"
        )
        plot_contf(axs[k, 1], test_inputs, z=var[:, k])
        # plot_contf(axs[1], test_inputs, z=var[:, dynamics.desired_mode], label=label)
    # axs[-1].legend()
    return fig


def plot_mixing_probs(dynamics: ModeRLDynamics, test_inputs: InputData):
    fig = plt.figure()
    gs = fig.add_gridspec(1, dynamics.mosvgpe.num_experts)
    axs = gs.subplots()
    probs = dynamics.mosvgpe.gating_network.predict_mixing_probs(test_inputs)
    for k in range(dynamics.mosvgpe.num_experts):
        plot_contf(axs[k], test_inputs, z=probs[:, k])
    return fig


def build_plotting_callbacks(
    dynamics: ModeRLDynamics, logging_epoch_freq: int = 100, num_test: int = 100
) -> List[PlottingCallback]:
    test_inputs = create_test_inputs(num_test=num_test)

    callbacks = [
        PlottingCallback(
            partial(
                plot_gating_network_gps, dynamics=dynamics, test_inputs=test_inputs
            ),
            logging_epoch_freq=logging_epoch_freq,
            name="Gating function posterior",
        ),
        PlottingCallback(
            partial(plot_mixing_probs, dynamics=dynamics, test_inputs=test_inputs),
            logging_epoch_freq=logging_epoch_freq,
            name="Mixing Probs",
        ),
    ]
    return callbacks


def create_test_inputs(num_test: int = 400):
    sqrtN = int(np.sqrt(num_test))
    xx = np.linspace(-3, 3, sqrtN)
    yy = np.linspace(-3, 3, sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    zeros = np.zeros((num_test, 2))
    test_inputs = np.concatenate([test_inputs, zeros], -1)
    return test_inputs


if __name__ == "__main__":
    save_dirs = {
        "before": "./wandb/run-20220929_123312-2x3tjd8w/files/saved-models/dynamics-before-training-config.json",
        "after": "./wandb/run-20220929_123312-2x3tjd8w/files/saved-models/dynamics-after-training-on-dataset-0-config.json",
    }

    test_inputs = create_test_inputs(100)
    print("test_inputs")
    print(test_inputs.shape)
    print(test_inputs)

    for key in save_dirs.keys():
        dynamics = ModeRLDynamics.load(save_dirs[key])
        explorative_controller = ExplorativeController.load(save_dirs[key])

        plot_gating_networks_gp(dynamics, test_inputs)  # pyright: ignore
        save_name = "./images/gating_network_gp_" + key
        plt.savefig(save_name + ".pdf", transparent=True)
        # tikzplotlib.clean_figure()
        tikzplotlib.save(save_name + ".tex")

        plot_desired_mixing_probs(dynamics, test_inputs)  # pyright: ignore
        save_name = "./images/desired_mixing_prob_" + key
        plt.savefig(save_name + ".pdf", transparent=True)
        # tikzplotlib.clean_figure()
        tikzplotlib.save(save_name + ".tex")
