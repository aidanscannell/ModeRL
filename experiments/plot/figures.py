#!/usr/bin/env python3
from collections import OrderedDict
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import palettable
import tensorflow as tf
import tensorflow_probability as tfp
import wandb
from experiments.plot.utils import (
    create_test_inputs,
    get_ExplorativeController_from_id,
    plot_contf,
    plot_data_over_ax,
    plot_env,
    plot_env_cmap,
    plot_gating_function_variance,
    plot_mode_satisfaction_prob,
    plot_start_end_pos,
    plot_trajectories,
)
from moderl.custom_types import State
from mpl_toolkits.axes_grid1 import make_axes_locatable


tfd = tfp.distributions


def plot_sinlge_run_gating_function_variance(
    env,
    run_id: str,
    wandb_dir: str,
    target_state: State,
    iteration: int = 0,
    title: Optional[str] = "",
    legend: bool = False,
    cbar: bool = False,
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
    h_mean, h_var = explorative_controller.dynamics.mosvgpe.gating_network.gp.predict_f(
        test_inputs
    )
    contf = plot_contf(
        ax,
        test_inputs,
        z=h_var[:, explorative_controller.dynamics.desired_mode],
        levels=levels,
        cmap="coolwarm",
    )
    plot_mode_satisfaction_prob(
        ax, controller=explorative_controller, test_inputs=test_inputs
    )
    plot_trajectories(
        ax, env, controller=explorative_controller, target_state=target_state
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.set_title(title)
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)

        cbar.set_label(
            r"$\mathbb{V}[h_{k^*}(\mathbf{s}) \mid \mathbf{s}, \mathcal{D}_{0:"
            + str(iteration)
            + "})$"
        )

    if legend:
        by_label = custom_labels(ax)
        ax.legend(by_label.values(), by_label.keys(), loc="lower left")
    fig.tight_layout()
    return fig


def plot_sinlge_run_mode_prob(
    env,
    run_id: str,
    wandb_dir: str,
    target_state: State,
    iteration: int = 0,
    title: Optional[str] = "",
    legend: bool = False,
    cbar: bool = False,
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
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.set_title(title)
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)

        cbar.set_label(
            r"$\Pr(\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:" + str(iteration) + "})$"
        )

    if legend:
        by_label = custom_labels(ax)
        ax.legend(by_label.values(), by_label.keys(), loc="lower left")
    fig.tight_layout()
    return fig


def plot_greedy_results(
    env,
    run_id: str,
    wandb_dir: str,
    target_state: State,
    iteration: int = 0,
    title: Optional[str] = "",
    legend: bool = False,
    cbar: bool = False,
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
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.set_title(title)
    if cbar:
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

    if legend:
        by_label = custom_labels(ax)
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
    # fig, axs = plt.subplots(ncols=4, figsize=(7, 2.2), sharey="row")
    fig, axs = plt.subplots(ncols=4, figsize=(8.5, 3), sharey="row")
    # fig, axs = plt.subplots(ncols=4, figsize=(8.5, 3), sharey="row")
    fig.subplots_adjust(bottom=0.4, wspace=0.0)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.set_xlabel("$x$")
        plot_env(ax, env, test_inputs=test_inputs)
        ax.set_xlim(np.min(test_inputs[:, 0]), np.max(test_inputs[:, 0]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

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

    by_label = custom_labels(axs[2])
    axs[2].legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.1, -0.17),
        fancybox=False,
        shadow=False,
        ncol=len(by_label),
    )
    # fig.tight_layout()
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
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    plot_env_cmap(ax, env, test_inputs=test_inputs, aspect_ratio=0.6)
    # plot_env(ax, env, test_inputs=test_inputs)
    # COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    iterations.reverse()
    # COLORS = ["#6e5f6e", "#8e3f8e", "#ad20ad", "#c508c5"]  # ,"#cd00cd","#800080"]
    # COLORS = ["#efbbff", "#d896ff", "#be29ec", "#800080", "#660066"]
    # COLORS = ["#efbbff", "#d896ff", "#be29ec", "#800080", "#660066"]
    # COLORS = ["#0000ff", "#0000b1", "#000076", "#00003b"]
    # COLORS = ["#0000ff", "#efbbff" "#0000b1", "#000076", "#00003b"]
    # COLORS = ["#ffc0cb", "#70d0f0", "#787078", "#261150"]
    COLORS = ["#ffc0cb", "#70d0f0", "#484348", "#261150"]

    # COLORS.reverse()
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
            # colors=cmap(idx),
            colors=COLORS[idx],
        )
        # r"$\Pr(\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:" + str(i) + "}>0.8$"

    plot_env(ax, env, test_inputs=test_inputs)
    plot_start_end_pos(
        ax, start_state=explorative_controller.start_state, target_state=target_state
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

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
    # fig.tight_layout()
    return fig


from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes


def plot_uncertainty_comparison(
    env,
    saved_runs: omegaconf.DictConfig,
    wandb_dir: str,
    target_state: State,
    iterations: List[int] = [0, 1, 2, 3],
):
    api = wandb.Api()
    run = api.run(saved_runs.joint_gating.id)
    test_inputs = create_test_inputs(num_test=40000)
    # fig, axs = plt.subplots(ncols=2, figsize=(5.1, 2.2), sharey="row")
    # fig, axs = plt.subplots(ncols=2, figsize=(6, 2.8), sharey="row")

    fig, axs = plt.subplots(ncols=2, figsize=(6, 3), sharey="row")
    fig.subplots_adjust(bottom=0.4, wspace=0.4)

    fig, axs = plt.subplots(ncols=3, figsize=(10, 3), sharey="row")
    fig.subplots_adjust(bottom=0.4, wspace=0.4)

    cmap = palettable.scientific.sequential.Bilbao_15.mpl_colormap
    cmap = "coolwarm"
    i = 20
    i_prob = 60
    levels = np.linspace(0, 1, 11)

    # plot bernoulli entropy
    # run_id = saved_runs.bernoulli.id.split("/")[-1]
    run_id = saved_runs.joint_gating.id.split("/")[-1]
    explorative_controller = get_ExplorativeController_from_id(
        i=i, id=run_id, wandb_dir=wandb_dir
    )
    probs = explorative_controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
        test_inputs
    )
    mode_indicator_variable = tfd.Bernoulli(
        probs=probs[:, explorative_controller.dynamics.desired_mode]
    )
    bernoulli_entropy = mode_indicator_variable.entropy()
    contf = plot_contf(
        ax=axs[0], test_inputs=test_inputs, z=bernoulli_entropy, cmap=cmap
    )
    plot_mode_satisfaction_prob(
        axs[0], controller=explorative_controller, test_inputs=test_inputs
    )
    plot_data_over_ax(ax=axs[0], X=explorative_controller.dynamics.dataset[0][:-15, :])

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)
    cbar.set_label(
        r"$\mathcal{H}[\alpha \mid \mathbf{s}, \mathcal{D}_{0:" + str(i) + "}]$"
    )

    # Plot gating function entropy
    (
        h_means,
        h_vars,
    ) = explorative_controller.dynamics.mosvgpe.gating_network.gp.predict_f(test_inputs)
    print("tf.reduce_min(h_vars)")
    print(tf.reduce_min(h_vars))
    print(tf.reduce_max(h_vars))
    # h_dist = tfd.Normal(h_means[:, 0], h_vars[:, 0])
    h_dist = tfd.Normal(h_means[:, 0], tf.math.sqrt(h_vars[:, 0]))
    print(h_dist)
    gating_entropy = h_dist.entropy()
    print("tf.reduce_min(gating_entropy)")
    print(tf.reduce_min(gating_entropy))
    print(tf.reduce_max(gating_entropy))
    contf = plot_contf(ax=axs[1], test_inputs=test_inputs, z=gating_entropy, cmap=cmap)
    plot_mode_satisfaction_prob(
        axs[1], controller=explorative_controller, test_inputs=test_inputs
    )
    plot_data_over_ax(ax=axs[1], X=explorative_controller.dynamics.dataset[0][:-15, :])

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)
    cbar.set_label(
        r"$\mathcal{H}[h_{k^*}(\mathbf{s}) \mid \mathbf{s}, \mathcal{D}_{0:"
        + str(i)
        + "}]$"
    )

    def plot_zoomed_in(ax, z):
        axins = zoomed_inset_axes(ax, zoom=1.8, loc="upper left")
        axins.tick_params(labelleft=False, labelbottom=False)
        plot_mode_satisfaction_prob(
            axins, controller=explorative_controller, test_inputs=test_inputs
        )
        plot_contf(ax=axins, test_inputs=test_inputs, z=z, cmap=cmap)
        plot_data_over_ax(
            ax=axins, X=explorative_controller.dynamics.dataset[0][:-15, :]
        )
        plot_env(axins, env, test_inputs=test_inputs)
        axins.set_xlim(-0.8, 1.2)
        axins.set_ylim(-1.2, 1)
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
        # mark_inset(axs[0], axins, loc1=-2, loc2=2, fc="none", ec="0.5")

    plot_zoomed_in(axs[0], z=bernoulli_entropy)
    plot_zoomed_in(axs[1], z=gating_entropy)

    # Plot mode prob for Bernoulli experiment
    i = 60
    run_id = saved_runs.bernoulli.id.split("/")[-1]
    explorative_controller = get_ExplorativeController_from_id(
        i=i, id=run_id, wandb_dir=wandb_dir
    )
    probs = explorative_controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
        test_inputs
    )
    contf = plot_contf(
        axs[-1],
        test_inputs,
        z=probs[:, explorative_controller.dynamics.desired_mode],
        levels=levels,
    )
    plot_mode_satisfaction_prob(
        axs[-1], controller=explorative_controller, test_inputs=test_inputs
    )
    plot_trajectories(
        axs[-1], env, controller=explorative_controller, target_state=target_state
    )
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)
    cbar.set_label(r"$\Pr(\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:" + str(i) + "})$")

    for ax in axs.flat:
        ax.set_xlim(np.min(test_inputs[:, 0]), np.max(test_inputs[:, 0]))
        ax.set_xlabel("$x$")
        plot_env(ax, env, test_inputs=test_inputs)
        plot_start_end_pos(
            ax,
            start_state=explorative_controller.start_state,
            target_state=target_state,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    axs[0].set_ylabel("$y$")

    axs[0].set_title("Aleatoric uncertainty")
    axs[1].set_title("Epistemic uncertainty")
    axs[2].set_title("Aleatoric strategy fails")

    by_label_custom = custom_labels(axs[0])
    handles, labels = axs[-1].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    by_label.update(by_label_custom)

    axs[1].legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.45, -0.15),
        # bbox_to_anchor=(-0.08, -0.15),
        fancybox=False,
        shadow=False,
        ncol=len(by_label),
    )

    # fig.tight_layout()
    return fig


def custom_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([1], [1], color="b", alpha=0.5, linestyle="dashed"))
    labels.append("Mode boundary")
    handles.append(mpl.lines.Line2D([1], [1], color="k"))
    labels.append(r"$\delta$-mode constraint")
    by_label = OrderedDict(zip(labels, handles))
    return by_label
