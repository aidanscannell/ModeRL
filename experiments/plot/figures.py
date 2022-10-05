#!/usr/bin/env python3
import os
from collections import OrderedDict
from typing import Callable, List

import hydra
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import palettable
import simenvs
import tensorflow as tf
import tikzplotlib
from experiments.plot.controller import plot_env, plot_trajectories
from experiments.plot.utils import (
    create_test_inputs,
    plot_contf,
    plot_mode_satisfaction_prob,
)
from matplotlib import patches
from moderl.controllers import ExplorativeController
from moderl.controllers.explorative_controller import ExplorativeController
from moderl.custom_types import InputData, State
from moderl.dynamics import ModeRLDynamics
from moderl.dynamics.dynamics import ModeRLDynamics
from mpl_toolkits.axes_grid1 import make_axes_locatable

import wandb

plt.style.use("seaborn-paper")
CMAP = palettable.scientific.sequential.Bilbao_15.mpl_colormap

LABELS = {"env": "Environment", "dynamics": "Dynamics"}
COLORS = {"env": "c", "dynamics": "m"}
LINESTYLES = {"env": "-", "dynamics": "-"}
MARKERS = {"env": "*", "dynamics": "."}

LINESTYLES = ["-", ".", "..", "-."]


def figure_1(env, load_dir: str, iterations: List[int] = [0, 1, 2]):
    test_inputs = create_test_inputs(num_test=40000)
    test_states = test_inputs[:, 0:2]
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    plot_env(ax, env, test_inputs=test_inputs)

    # contf = plot_contf(
    #     ax, test_inputs, z=probs[:, dynamics.desired_mode], levels=levels
    # )

    for i in iterations:
        load_file = os.path.join(
            load_dir, "controller-optimised-{}-config.json".format(i)
        )
        explorative_controller = ExplorativeController.load(load_file)
        # load_file = os.path.join(
        #     load_dir, "dynamics-after-training-on-dataset-{}-config.json".format(i)
        # )
        # dynamics = ModeRLDynamics.load(load_file)
        probs = (
            explorative_controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                test_inputs
            )[:, explorative_controller.dynamics.desired_mode]
        )
        # plot_contf(ax, test_inputs, z=probs[:, dynamics.desired_mode])
        CS = ax.tricontour(
            test_states[:, 0],
            test_states[:, 1],
            probs.numpy(),
            [explorative_controller.mode_satisfaction_prob],
            label="i={}".format(i),
            linestyle=LINESTYLES[i],
        )
        # clabel = ax.clabel(CS, inline=True, fontsize=8, fmt={0.5: "i=" + str(i)})
        # clabel = ax.clabel(CS, fmt={0.5: "i=" + str(i)})
        # clabel[0].set_bbox(dict(boxstyle="round,pad=0.1", fc="white", alpha=1.0))
    ax.set_title(
        "$S_{k^*}^{i} = \{\mathbf{s} \in \mathcal{S} \mid \\alpha(\mathbf{s})=k^{*} \}$"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # ax.legend()
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
    # fig.tight_layout()
    return fig


def figure_3(
    env, load_dir: str, target_state: State, iterations: List[int] = [0, 1, 2]
):
    test_inputs = create_test_inputs(num_test=40000)
    # assert len(iterations) == 3
    fig = plt.figure()
    gs = fig.add_gridspec(1, 3, wspace=0.0, hspace=0.01)
    axs = gs.subplots(sharey="row")
    # axs = gs.subplots(sharex="col", sharey="row")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    axs[1].label_outer()
    axs[2].label_outer()
    for ax in axs.flat:
        # ax.label_outer()
        ax.set_xlabel("$x$")
        plot_env(ax, env, test_inputs=test_inputs)

    levels = np.linspace(0, 1, 11)
    for idx, i in enumerate(iterations):
        load_file = os.path.join(
            load_dir, "controller-optimised-{}-config.json".format(i)
        )
        explorative_controller = ExplorativeController.load(load_file)
        probs = (
            explorative_controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                test_inputs
            )
        )
        contf = plot_contf(
            axs[idx],
            test_inputs,
            z=probs[:, explorative_controller.dynamics.desired_mode],
            levels=levels,
        )
        plot_mode_satisfaction_prob(
            axs[idx],
            dynamics=explorative_controller.dynamics,
            test_inputs=test_inputs,
            mode_satisfaction_prob=explorative_controller.mode_satisfaction_prob,
        )
        plot_trajectories(
            axs[idx], env, controller=explorative_controller, target_state=target_state
        )
        axs[idx].set_title("i=" + str(i))
    axs[0].set_ylabel("$y$")
    axs[1].set_yticks([])
    axs[2].set_yticks([])
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)

    cbar.set_label("$\Pr(\\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:i})$")
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


if __name__ == "__main__":
    run_dir = "../wandb/run-20221005_001811-3olk04ju"
    load_dir = os.path.join(run_dir, "files/saved-models")
    # iterations = [1, 2, 3]
    iterations = [1, 2]

    cfg = omegaconf.OmegaConf.load(os.path.join(run_dir, "files/config.yaml"))
    # print(cfg)
    env = simenvs.make(cfg.env.value.name)
    print(cfg.target_state.value.value)
    print(type(cfg.target_state))
    target_state = np.array(cfg.target_state.value.value)

    fig = figure_1(env=env, load_dir=load_dir, iterations=iterations)
    save_name = "../images/figure-1"
    plt.savefig(save_name + ".pdf", transparent=True)
    # tikzplotlib.clean_figure()
    tikzplotlib.save(
        save_name
        + ".tex"
        # save_name + ".tex", axis_width="\\figurewidth", axis_height="\\figureheight"
    )

    fig = figure_3(
        env=env, load_dir=load_dir, target_state=target_state, iterations=iterations
    )
    save_name = "../images/figure-3"
    plt.savefig(save_name + ".pdf", transparent=True)
    # tikzplotlib.clean_figure()
    tikzplotlib.save(
        save_name
        + ".tex"
        # save_name + ".tex", axis_width="\\figurewidth", axis_height="\\figureheight"
    )
