#!/usr/bin/env python3
from collections import OrderedDict
from typing import Callable

import matplotlib.pyplot as plt
import wandb
from experiments.plot.utils import (
    create_test_inputs,
    plot_contf,
    plot_mode_satisfaction_prob,
    plot_trajectories,
)
from moderl.controllers import ControllerInterface
from moderl.custom_types import InputData, State


def plot_trajectories_over_desired_mixing_prob(
    env, controller: ControllerInterface, test_inputs: InputData, target_state: State
):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    probs = controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(test_inputs)
    plot_contf(ax, test_inputs, z=probs[:, controller.dynamics.desired_mode])
    plot_trajectories(ax, env, controller=controller, target_state=target_state)
    plot_env(ax, env=env, test_inputs=test_inputs)
    plot_mode_satisfaction_prob(ax, controller=controller, test_inputs=test_inputs)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.tight_layout()
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 0.05),
        loc="upper center",
        bbox_transform=fig.transFigure,
        ncol=len(by_label),
    )
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
        plot_trajectories(ax, env=env, controller=controller, target_state=target_state)
        plot_env(ax, env=env, test_inputs=test_inputs)
        plot_mode_satisfaction_prob(ax, controller=controller, test_inputs=test_inputs)
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.tight_layout()
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 0.05),
        loc="upper center",
        bbox_transform=fig.transFigure,
        ncol=len(by_label),
    )
    return fig


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
