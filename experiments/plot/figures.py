#!/usr/bin/env python3
from collections import OrderedDict
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from experiments.plot.utils import (
    create_test_inputs,
    get_ExplorativeController_from_id,
    plot_contf,
    plot_env,
    plot_env_cmap,
    plot_mode_satisfaction_prob,
    plot_start_end_pos,
    plot_trajectories,
)
from moderl.custom_types import State
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_sinlge_run(
    env,
    run_id: str,
    wandb_dir: str,
    target_state: State,
    iteration: int = 0,
    title: Optional[str] = "",
):
    test_inputs = create_test_inputs(num_test=40000)
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()

    levels = np.linspace(0, 1, 11)

    env_contf = plot_env(ax, env, test_inputs=test_inputs)
    explorative_controller = get_ExplorativeController_from_id(
        iteration, id=run_id, wandb_dir=wandb_dir
    )
    probs = explorative_controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
        test_inputs
    )
    contf = plot_contf(
        ax,
        test_inputs,
        z=probs[:, explorative_controller.dynamics.desired_mode],
        levels=levels,
    )
    plot_mode_satisfaction_prob(
        ax, controller=explorative_controller, test_inputs=test_inputs
    )
    plot_trajectories(
        ax, env, controller=explorative_controller, target_state=target_state
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right",
        size="5%",
        pad=0.05,
    )
    cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)

    cbar.set_label(
        r"$\Pr(\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:" + str(iteration) + "})$"
    )
    # ax.legend(loc="upper left")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([1], [1], color="b", linestyle="dashed"))
    labels.append("Mode boundary")
    handles.append(mpl.lines.Line2D([1], [1], color="k"))
    labels.append(r"$\delta$-mode constraint")
    by_label = OrderedDict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    ax.legend(by_label.values(), by_label.keys(), loc="lower left")
    fig.tight_layout()
    return fig


def plot_four_iterations_in_row(
    env,
    run_id: str,
    wandb_dir: str,
    target_state: State,
    iterations: List[int] = [0, 1, 2, 3],
    title: Optional[str] = "",
):
    test_inputs = create_test_inputs(num_test=40000)
    fig, axs = plt.subplots(ncols=4, figsize=(7, 2.2), sharey="row")
    fig.subplots_adjust(bottom=0.4, wspace=0.0)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.set_xlabel("$x$")
        plot_env(ax, env, test_inputs=test_inputs)

    levels = np.linspace(0, 1, 11)
    for idx, i in enumerate(iterations):
        explorative_controller = get_ExplorativeController_from_id(
            i=i, id=run_id, wandb_dir=wandb_dir
        )
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
            axs[idx], controller=explorative_controller, test_inputs=test_inputs
        )
        plot_trajectories(
            axs[idx], env, controller=explorative_controller, target_state=target_state
        )
        axs[idx].set_title("$i=" + str(i) + "$")
    axs[0].set_ylabel("$y$")
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)

    cbar.set_label(r"$\Pr(\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:i})$")
    handles, labels = axs[idx].get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([1], [1], color="b", linestyle="dashed"))
    labels.append("Mode boundary")
    handles.append(mpl.lines.Line2D([1], [1], color="k"))
    labels.append(r"$\delta$-mode constraint")
    by_label = OrderedDict(zip(labels, handles))
    axs[2].legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.1, -0.35),
        fancybox=False,
        shadow=False,
        ncol=len(by_label),
    )
    fig.tight_layout()
    return fig


def plot_constraint_expanding_figure(
    env,
    run_id: str,
    wandb_dir: str,
    target_state: State,
    iterations: List[int] = [0, 1, 2],
):
    test_inputs = create_test_inputs(num_test=40000)
    test_states = test_inputs[:, 0:2]
    # fig = plt.figure(figsize=(5, 4))
    # fig = plt.figure(figsize=(4, 3.6))
    # fig = plt.figure(figsize=(5, 4))
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    plot_env_cmap(ax, env, test_inputs=test_inputs, aspect_ratio=0.6)
    # plot_env(ax, env, test_inputs=test_inputs)

    # COLORS = ["r", "g", "b", "y"]
    # COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    iterations.reverse()
    # COLORS = ["#6e5f6e", "#8e3f8e", "#ad20ad", "#c508c5"]  # ,"#cd00cd","#800080"]
    COLORS = ["#efbbff", "#d896ff", "#be29ec", "#800080", "#660066"]
    COLORS.reverse()
    for idx, i in enumerate(iterations):
        explorative_controller = get_ExplorativeController_from_id(
            i=i, id=run_id, wandb_dir=wandb_dir
        )

        probs = (
            explorative_controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                test_inputs
            )[:, explorative_controller.dynamics.desired_mode]
        )
        ax.tricontour(
            test_states[:, 0],
            test_states[:, 1],
            probs.numpy(),
            [explorative_controller.mode_satisfaction_prob],
            colors=COLORS[idx],
        )
        # r"$\Pr(\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:" + str(i) + "}>0.8$"

    plot_start_end_pos(
        ax, start_state=explorative_controller.start_state, target_state=target_state
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    iterations.reverse()
    COLORS.reverse()
    for idx, i in enumerate(iterations):
        handles.append(mpl.lines.Line2D([1], [1], color=COLORS[idx]))
        labels.append(r"$i=" + str(i) + "$")

    cmap = mpl.cm.get_cmap()
    c0 = cmap(0.1)
    c1 = cmap(0.9)
    handles.append(mpl.patches.Patch(color=c1))
    labels.append("Mode 1")
    handles.append(mpl.patches.Patch(color=c0))
    labels.append("Mode 2")
    # handles.append(mpl.lines.Line2D([1], [1], color="b", linestyle="dashed"))
    # labels.append("Mode boundary")

    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower left")
    fig.tight_layout()
    return fig
