#!/usr/bin/env python3
from typing import Callable, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import palettable
import simenvs
import tensorflow as tf
import tikzplotlib
import wandb
from experiments.plot.dynamics import plot_gating_network_gps
from experiments.plot.utils import (
    PlottingCallback,
    create_test_inputs,
    plot_contf,
    plot_desired_mixing_prob,
)
from matplotlib import patches
from moderl.controllers import ControllerInterface, ExplorativeController
from moderl.custom_types import InputData, State
from moderl.dynamics import ModeRLDynamics
from moderl.dynamics.dynamics import ModeRLDynamics
from moderl.rollouts import rollout_trajectory_optimisation_controller_in_env


LABELS = {"env": "Environment", "dynamics": "Dynamics"}
COLORS = {"env": "c", "dynamics": "m"}
LINESTYLES = {"env": "-", "dynamics": "-"}
MARKERS = {"env": "*", "dynamics": "."}


PlotFn = Callable[[], matplotlib.figure.Figure]


class WandBImageCallbackScipy:
    def __init__(
        self, plot_fn: PlotFn, logging_epoch_freq: int = 10, name: Optional[str] = ""
    ):
        self.plot_fn = plot_fn
        self.logging_epoch_freq = logging_epoch_freq
        self.name = name

    def __call__(self, step, variables, value):
        if step % self.logging_epoch_freq == 0:
            fig = self.plot_fn()
            wandb.log({self.name: wandb.Image(fig)})


def plot_trajectories_over_desired_mixing_prob(
    env, controller: ControllerInterface, test_inputs: InputData, target_state: State
):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    probs = controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(test_inputs)
    plot_contf(ax, test_inputs, z=probs[:, controller.dynamics.desired_mode])
    plot_trajectories(ax, env, controller=controller, target_state=target_state)
    return fig


def plot_trajectories_over_desired_gating_gp(
    env, controller: ControllerInterface, test_inputs: InputData, target_state: State
):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)
    axs = gs.subplots()
    mean, var = controller.dynamics.mosvgpe.gating_network.predict_h(test_inputs)
    plot_contf(axs[0], test_inputs, z=mean[:, controller.dynamics.desired_mode])
    plot_contf(axs[1], test_inputs, z=var[:, controller.dynamics.desired_mode])

    for ax in axs:
        plot_trajectories(ax, env, controller=controller, target_state=target_state)
    return fig


def plot_trajectories(ax, env, controller: ControllerInterface, target_state: State):
    env_traj = rollout_trajectory_optimisation_controller_in_env(
        env=env, start_state=controller.start_state, controller=controller
    )
    dynamics_traj = controller.rollout_in_dynamics().mean()

    # for traj, key in zip([env_traj, dynamics_traj], ["env", "dynamics"]):
    for traj, key in zip([dynamics_traj], ["dynamics"]):
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


def build_controller_plotting_callback(
    env,
    controller: ControllerInterface,
    target_state: State,
    logging_epoch_freq: int = 10,
    num_test: int = 100,
) -> Callable:
    test_inputs = create_test_inputs(num_test=num_test)

    def plotting_callback(step, variables, value):
        if step % logging_epoch_freq == 0:
            fig = plot_trajectories_over_desired_gating_gp(
                env=env,
                controller=controller,
                test_inputs=test_inputs,
                target_state=target_state,
            )
            wandb.log({"Trajectories over desired gating GP": wandb.Image(fig)})
            fig = plot_trajectories_over_desired_mixing_prob(
                env=env,
                controller=controller,
                test_inputs=test_inputs,
                target_state=target_state,
            )
            wandb.log({"Trajectories over desired mixing prob": wandb.Image(fig)})

    return plotting_callback


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
