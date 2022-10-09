#!/usr/bin/env python3
import os
from collections import OrderedDict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import palettable
import simenvs
import tensorflow as tf
import tikzplotlib
from experiments.plot.controller import (
    plot_env,
    plot_env_cmap,
    plot_start_end_pos,
    plot_trajectories,
)
from experiments.plot.utils import (
    create_test_inputs,
    plot_contf,
    plot_mode_satisfaction_prob,
)
from moderl.controllers import ExplorativeController
from moderl.custom_types import State
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.style.use("seaborn-paper")
CMAP = palettable.scientific.sequential.Bilbao_15.mpl_colormap

LABELS = {"env": "Environment", "dynamics": "Dynamics"}
COLORS = {"env": "c", "dynamics": "m"}
LINESTYLES = {"env": "-", "dynamics": "-"}
MARKERS = {"env": "*", "dynamics": "."}

LINESTYLES = ["-", ".", "..", "-."]

params = {
    # 'axes.labelsize': 30,
    # 'font.size': 30,
    # 'legend.fontsize': 20,
    # 'xtick.labelsize': 30,
    # 'ytick.labelsize': 30,
    "text.usetex": True,
}
plt.rcParams.update(params)


def figure_1(
    env, load_dir: str, target_state: State, iterations: List[int] = [0, 1, 2]
):
    test_inputs = create_test_inputs(num_test=40000)
    test_states = test_inputs[:, 0:2]
    # fig = plt.figure()
    fig = plt.figure(figsize=(3, 2.7))
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    plot_env(ax, env, test_inputs=test_inputs)
    plot_env_cmap(ax, env, test_inputs=test_inputs, cmap=CMAP)

    # contf = plot_contf(
    #     ax, test_inputs, z=probs[:, dynamics.desired_mode], levels=levels
    # )

    for idx, i in enumerate(iterations):
        load_file = os.path.join(
            load_dir, "controller-optimised-{}-config.json".format(i)
        )
        explorative_controller = ExplorativeController.load(load_file)

        probs = (
            explorative_controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                test_inputs
            )[:, explorative_controller.dynamics.desired_mode]
        )
        CS = ax.tricontour(
            test_states[:, 0],
            test_states[:, 1],
            probs.numpy(),
            [explorative_controller.mode_satisfaction_prob],
        )
        clabel = ax.clabel(
            CS,
            inline=True,
            fontsize=8,
            fmt={explorative_controller.mode_satisfaction_prob: "i=" + str(i)},
        )
        clabel[0].set_bbox(dict(boxstyle="round,pad=0.1", fc="white", alpha=1.0))
        # clabel = ax.clabel(CS, fmt={0.5: "i=" + str(i)})
        # clabel[0].set_bbox(dict(boxstyle="round,pad=0.1", fc="white", alpha=1.0))

    plot_start_end_pos(
        ax, start_state=explorative_controller.start_state, target_state=target_state
    )

    # ax.set_title(
    #    "$S_{k^*}^{i} = \{\\mathbf{s} \\in \\mathcal{S} \\mid \\Pr \\left(\\alpha=k^{*}
    #     \\mid \\mathbf{s}, \\mathcal{D}_{0:i} \\geq 1-\\delta \\right) \}$"
    # )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # ax.legend()
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    fig.tight_layout()
    # fig.legend(
    #     by_label.values(),
    #     by_label.keys(),
    #     bbox_to_anchor=(0.5, 0.05),
    #     loc="upper center",
    #     bbox_transform=fig.transFigure,
    #     ncol=len(by_label),
    # )
    # fig.tight_layout()
    return fig


def figure_3(
    env, load_dir: str, target_state: State, iterations: List[int] = [0, 1, 2]
):
    test_inputs = create_test_inputs(num_test=40000)
    # fig = plt.figure(figsize=(6, 2))
    # gs = fig.add_gridspec(1, 3, wspace=0.0, hspace=0.01)
    fig = plt.figure(figsize=(7, 2))
    gs = fig.add_gridspec(1, 4, wspace=0.0)
    axs = gs.subplots(sharey="row")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # axs[1].label_outer()
    # axs[2].label_outer()
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
        axs[idx].set_title("$i=" + str(i) + "$")
    axs[0].set_ylabel("$y$")
    # axs[1].set_yticks([])
    # axs[2].set_yticks([])
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)

    cbar.set_label(r"$\Pr(\\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:i})$")
    handles, labels = axs[idx].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    axs[0].legend(
        by_label.values(),
        by_label.keys(),
        # bbox_to_anchor=(0.5, 0.05),
        loc="upper left",
        # bbox_transform=fig.transFigure,
        # ncol=len(by_label),
    )
    # fig.legend(
    #     by_label.values(),
    #     by_label.keys(),
    #     # bbox_to_anchor=(0.5, 0.05),
    #     loc="lower center",
    #     bbox_transform=fig.transFigure,
    #     ncol=len(by_label),
    # )
    fig.tight_layout()
    return fig


def figure_3_separately(
    env, load_dir: str, target_state: State, iterations: List[int] = [0, 1, 2]
):
    test_inputs = create_test_inputs(num_test=40000)
    levels = np.linspace(0, 1, 11)
    figs = []

    for idx, i in enumerate(iterations):
        fig = plt.figure()
        gs = fig.add_gridspec(1, 1)
        ax = gs.subplots()

        # ax.set_xlabel("$x$")
        plot_env(ax, env, test_inputs=test_inputs)
        load_file = os.path.join(
            load_dir, "controller-optimised-{}-config.json".format(i)
        )
        explorative_controller = ExplorativeController.load(load_file)
        probs = (
            explorative_controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                test_inputs
            )
        )
        plot_contf(
            ax,
            test_inputs,
            z=probs[:, explorative_controller.dynamics.desired_mode],
            levels=levels,
        )
        plot_mode_satisfaction_prob(
            ax,
            dynamics=explorative_controller.dynamics,
            test_inputs=test_inputs,
            mode_satisfaction_prob=explorative_controller.mode_satisfaction_prob,
        )
        plot_trajectories(
            ax, env, controller=explorative_controller, target_state=target_state
        )
        # ax.set_title("i=" + str(i))
        figs.append(fig)

    # axs[0].set_ylabel("$y$")
    # axs[1].set_yticks([])
    # axs[2].set_yticks([])
    # divider = make_axes_locatable(axs[-1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)

    # cbar.set_label("$\Pr(\\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:i})$")
    # handles, labels = figs[-1].gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # # figs.tight_layout()
    # fig.legend(
    #     by_label.values(),
    #     by_label.keys(),
    #     bbox_to_anchor=(0.5, 0.05),
    #     loc="upper center",
    #     bbox_transform=fig.transFigure,
    #     ncol=len(by_label),
    # )
    return figs


def figure_4(
    env, load_dir: str, target_state: State, iterations: List[int] = [0, 1, 2]
):
    test_inputs = create_test_inputs(num_test=40000)
    fig = plt.figure(figsize=(3, 2.7))
    gs = fig.add_gridspec(1, 1)
    axs = gs.subplots()

    axs[0].set_xlabel("$i$")
    # ax.set_ylabel("$\\sum_{t=0}^{T} r(s_t, a_t)$")
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
        axs[idx].set_title("$i=" + str(i) + "$")
    axs[0].set_ylabel("$y$")
    # axs[1].set_yticks([])
    # axs[2].set_yticks([])
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)

    cbar.set_label(r"$\Pr(\\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:i})$")
    handles, labels = fig.gca().get_legend_handles_labels()
    print("handles")
    print(handles)
    print(labels)
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
    tf.keras.utils.set_random_seed(42)
    # run_dir = "../wandb/run-20221005_122823-1isab5ta"
    run_dir = "../wandb/run-20221007_123617-32cvz8j6"
    load_dir = os.path.join(run_dir, "files/saved-models")
    # iterations = [1, 2, 3]
    iterations = [1, 2]
    iterations = [1, 3, 6, 15]

    cfg = omegaconf.OmegaConf.load(os.path.join(run_dir, "files/config.yaml"))
    # print(cfg)
    env = simenvs.make(cfg.env.value.name)
    print(cfg.target_state.value.value)
    print(type(cfg.target_state))
    target_state = np.array(cfg.target_state.value.value)

    # fig = figure_1(
    #     env=env, load_dir=load_dir, target_state=target_state, iterations=iterations
    # )
    # save_name = "../images/figure-1"
    # plt.savefig(save_name + ".pdf", transparent=True)
    # # tikzplotlib.clean_figure()
    # tikzplotlib.save(
    #     save_name
    #     + ".tex"
    #     # save_name + ".tex", axis_width="\\figurewidth", axis_height="\\figureheight"
    # )

    # figs = figure_3_separately(
    #     env=env, load_dir=load_dir, target_state=target_state, iterations=iterations
    # )
    # for i, fig in enumerate(figs):
    #     save_name = "../images/figure-3-" + str(i)
    #     plt.savefig(save_name + ".pdf", transparent=True)
    #     # tikzplotlib.clean_figure()
    #     tikzplotlib.save(
    #         save_name
    #         + ".tex"
    #         # save_name + ".tex",
    #     )

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
