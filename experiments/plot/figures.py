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
    plot_mode_satisfaction_prob,
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
    ax.set_title(title)
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
    labels.append("Mode constraint")
    by_label = OrderedDict(zip(labels, handles))
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")
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
    # fig = plt.figure(figsize=(7, 2.2))
    # fig = plt.figure(figsize=(7, 2))
    # gs = fig.add_gridspec(1, 4, wspace=0.0)
    # axs = gs.subplots(sharey="row")

    fig, axs = plt.subplots(ncols=4, figsize=(7, 2.2), sharey="row")
    fig.subplots_adjust(bottom=0.4, wspace=0.0)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.set_xlabel("$x$")
        plot_env(ax, env, test_inputs=test_inputs)

    levels = np.linspace(0, 1, 11)
    for idx, i in enumerate(iterations):
        # load_file = os.path.join(
        #     load_dir, "controller-optimised-{}-config.json".format(i)
        # )
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
    # handles, labels = axs[idx].get_legend_handles_labels()
    handles, labels = axs[idx].get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([1], [1], color="b", linestyle="dashed"))
    labels.append("Mode boundary")
    handles.append(mpl.lines.Line2D([1], [1], color="k"))
    labels.append("Mode constraint")
    by_label = OrderedDict(zip(labels, handles))
    # axs[0].legend(by_label.values(), by_label.keys(), loc="upper left")
    axs[2].legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.2, -0.35),
        fancybox=False,
        shadow=False,
        ncol=len(by_label),
    )
    fig.tight_layout()
    return fig
