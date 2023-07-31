#!/usr/bin/env python3
import math
from collections import OrderedDict
from typing import List, Optional

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import palettable
import tensorflow as tf
import tensorflow_probability as tfp
import wandb
from experiments.plot.utils import (  # plot_gating_function_variance,
    create_test_inputs,
    get_ExplorativeController_from_id,
    plot_contf,
    plot_data_over_ax,
    plot_env,
    plot_env_cmap,
    plot_mode_satisfaction_prob,
    plot_start_end_pos,
    plot_trajectories,
)
from moderl.custom_types import State
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes


tfd = tfp.distributions


def plot_constraint_expanding_figure(
    env,
    run_id: str,
    wandb_dir: str,
    target_state: State,
    iterations: List[int] = [0, 1, 2],
):
    """Figure 1"""
    test_inputs = create_test_inputs(num_test=40000)
    test_states = test_inputs[:, 0:2]
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    plot_env_cmap(ax, env, test_inputs=test_inputs, aspect_ratio=0.6)
    iterations.reverse()
    LINESTYLES = ["solid", "dashed", "dashdot", "dotted"]

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
            colors="k",
            linestyles=LINESTYLES[idx],
        )

    # plot_env(ax, env, test_inputs=test_inputs)
    plot_start_end_pos(
        ax, start_state=explorative_controller.start_state, target_state=target_state
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    iterations.reverse()
    LINESTYLES.reverse()
    for idx, i in enumerate(iterations):
        handles.append(mpl.lines.Line2D([1], [1], linestyle=LINESTYLES[idx], color="k"))
        # handles.append(mpl.lines.Line2D([1], [1], color=COLORS[idx]))
        labels.append(r"$i=" + str(i) + "$")

    cmap = mpl.cm.get_cmap()
    c0 = cmap(0.15)
    c1 = cmap(0.8)
    handles.append(mpl.patches.Patch(color=c1))
    labels.append("Mode 1")
    handles.append(mpl.patches.Patch(color=c0))
    labels.append("Mode 2")

    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower left")
    # fig.tight_layout()
    return fig


def plot_four_iterations_in_row(
    env,
    run_id: str,
    wandb_dir: str,
    target_state: State,
    iterations: List[int] = [0, 1, 2, 3],
    title: Optional[str] = "",
):
    """Figure 2"""
    test_inputs = create_test_inputs(num_test=40000)
    # fig, axs = plt.subplots(ncols=4, figsize=(7, 2.2), sharey="row")
    fig, axs = plt.subplots(ncols=4, figsize=(8.5, 3), sharey="row")
    # fig, axs = plt.subplots(ncols=4, figsize=(8.5, 3), sharey="row")
    fig.subplots_adjust(bottom=0.4, wspace=0.0)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        # ax.set_xlabel("$x$")
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
    # axs[0].set_ylabel("$y$")
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)

    cbar.set_label(r"$\Pr(\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:i})$")

    by_label = custom_labels(axs[2])
    axs[2].legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.1, -0.07),
        fancybox=False,
        shadow=False,
        ncol=len(by_label),
    )
    # fig.tight_layout()
    return fig


def plot_greedy_and_myopic_comparison_figure(
    env,
    saved_runs: omegaconf.DictConfig,
    wandb_dir: str,
    target_state: State,
    iteration: int = 0,
    title: Optional[str] = "",
    legend: bool = False,
    cbar: bool = False,
):
    """Figure 4"""

    fig = plt.figure(figsize=(9, 2))
    # create a 1-row 3-column container as the left container
    gs_left = gridspec.GridSpec(1, 2)
    # fig, axs = plt.subplots(ncols=4, figsize=(9.5, 3), sharey="row")
    # fig.subplots_adjust(bottom=0.4, wspace=0.0)
    # create a 1-row 1-column grid as the right container
    gs_right = gridspec.GridSpec(1, 2)

    # add plots to the nested structure
    ax1 = fig.add_subplot(gs_left[0, 0])
    ax2 = fig.add_subplot(gs_left[0, 1])
    ax3 = fig.add_subplot(gs_right[0, 0])
    ax4 = fig.add_subplot(gs_right[0, 1])

    # now the plots are on top of each other, we'll have to adjust their edges so that they won't overlap
    # gs_left.update(right=0.65)
    # gs_right.update(left=0.7)
    gs_left.update(right=0.468)
    gs_right.update(left=0.532)

    # get rid of the horizontal spacing on both gridspecs
    gs_left.update(wspace=0)
    gs_right.update(wspace=0)

    test_inputs = create_test_inputs(num_test=40000)
    levels = np.linspace(0, 1, 11)

    def plot_prob(ax, run_id, iteration):
        explorative_controller = get_ExplorativeController_from_id(
            iteration, id=run_id, wandb_dir=wandb_dir
        )
        probs = (
            explorative_controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                test_inputs
            )
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
        return contf

    # iteration = 60
    iteration = saved_runs.greedy_no_constraint.iteration
    run_id = saved_runs.greedy_no_constraint.id.split("/")[-1]
    contf = plot_prob(ax1, run_id=run_id, iteration=iteration)
    run_id = saved_runs.greedy_with_constraint.id.split("/")[-1]
    contf = plot_prob(ax2, run_id=run_id, iteration=iteration)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)
    cbar.set_label(
        r"$\Pr(\alpha=k^*\mid \mathbf{s}, \mathcal{D}_{0:" + str(iteration) + "})$"
    )

    def plot_variance(ax, run_id, iteration):
        explorative_controller = get_ExplorativeController_from_id(
            iteration, id=run_id, wandb_dir=wandb_dir
        )
        (
            h_mean,
            h_var,
        ) = explorative_controller.dynamics.mosvgpe.gating_network.gp.predict_f(
            test_inputs
        )
        contf = plot_contf(
            ax,
            test_inputs,
            z=h_var[:, explorative_controller.dynamics.desired_mode],
            # levels=levels,
            # cmap="coolwarm",
            cmap=palettable.scientific.sequential.Bilbao_15.mpl_colormap,
        )
        plot_mode_satisfaction_prob(
            ax, controller=explorative_controller, test_inputs=test_inputs
        )
        plot_trajectories(
            ax, env, controller=explorative_controller, target_state=target_state
        )
        return contf

    # iteration = 2
    iteration = saved_runs.myopic_ablation.iteration
    run_id = saved_runs.moderl.id.split("/")[-1]
    contf = plot_variance(ax3, run_id=run_id, iteration=iteration)
    # run_id = saved_runs.independent_gating.id.split("/")[-1]
    # run_id = saved_runs.independent_gating_greedy_and_myopic_comparison.id.split("/")[
    run_id = saved_runs.myopic_ablation.id.split("/")[-1]
    contf = plot_variance(ax4, run_id=run_id, iteration=iteration)
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)
    cbar.set_label(
        r"$\mathbb{V}[h_{k^*}(\mathbf{s}) \mid \mathbf{s}, \mathcal{D}_{0:"
        + str(iteration)
        + "}]$"
    )

    for ax in [ax1, ax2, ax3, ax4]:
        plot_env(ax, env, test_inputs=test_inputs)
        # ax.set_xlabel("$x$")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # ax1.set_ylabel("$y$")
    ax1.set_title("Greedy no constraint")
    ax2.set_title("Greedy")
    ax3.set_title("Non-myopic exploration")
    ax4.set_title("Myopic exploration")

    by_label = custom_labels(ax1)
    ax3.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        # bbox_to_anchor=(0.25, -0.15),
        bbox_to_anchor=(0.25, -0.07),
        # bbox_to_anchor=(-0.08, -0.15),
        fancybox=False,
        shadow=False,
        ncol=len(by_label),
    )
    fig.tight_layout()
    return fig


def plot_uncertainty_comparison(
    env,
    saved_runs: omegaconf.DictConfig,
    wandb_dir: str,
    target_state: State,
    iterations: List[int] = [0, 1, 2, 3],
):
    """Figure 3"""
    api = wandb.Api()
    run = api.run(saved_runs.moderl.id)
    test_inputs = create_test_inputs(num_test=40000)
    # fig, axs = plt.subplots(ncols=2, figsize=(5.1, 2.2), sharey="row")
    # fig, axs = plt.subplots(ncols=2, figsize=(6, 2.8), sharey="row")

    fig, axs = plt.subplots(ncols=2, figsize=(6, 3), sharey="row")
    fig.subplots_adjust(bottom=0.4, wspace=0.4)

    fig, axs = plt.subplots(ncols=3, figsize=(10, 3), sharey="row")
    fig.subplots_adjust(bottom=0.4, wspace=0.4)

    cmap = palettable.scientific.sequential.Bilbao_15.mpl_colormap
    # cmap = "coolwarm"
    # i = 20
    i = saved_runs.aleatoric_unc_ablation.iteration
    i_prob = 60
    levels = np.linspace(0, 1, 11)

    # plot bernoulli entropy
    # run_id = saved_runs.bernoulli.id.split("/")[-1]
    run_id = saved_runs.moderl.id.split("/")[-1]
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
    # h_dist = tfd.Normal(h_means[:, 0], h_vars[:, 0])
    h_dist = tfd.Normal(h_means[:, 0], tf.math.sqrt(h_vars[:, 0]))
    gating_entropy = h_dist.entropy()
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
        # axins.set_xlim(-0.7, 1.1)
        axins.set_xlim(0.2, 1.8)
        axins.set_ylim(-1.4, 0.6)
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.2")
        # mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
        # mark_inset(axs[0], axins, loc1=-2, loc2=2, fc="none", ec="0.5")

    plot_zoomed_in(axs[0], z=bernoulli_entropy)
    plot_zoomed_in(axs[1], z=gating_entropy)

    # Plot mode prob for Bernoulli experiment
    # i = 37
    i = saved_runs.aleatoric_unc_ablation.iteration
    run_id = saved_runs.aleatoric_unc_ablation.id.split("/")[-1]
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
        # ax.set_xlabel("$x$")
        plot_env(ax, env, test_inputs=test_inputs)
        plot_start_end_pos(
            ax,
            start_state=explorative_controller.start_state,
            target_state=target_state,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # axs[0].set_ylabel("$y$")

    axs[0].set_title("Aleatoric uncertainty")
    axs[1].set_title("Epistemic uncertainty")
    axs[2].set_title("Aleatoric strategy failing")

    by_label_custom = custom_labels(axs[0])
    handles, labels = axs[-1].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    by_label.update(by_label_custom)

    axs[1].legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.45, -0.07),
        # bbox_to_anchor=(-0.08, -0.15),
        fancybox=False,
        shadow=False,
        ncol=len(by_label),
    )

    # fig.tight_layout()
    return fig


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
        # levels=levels,
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


def plot_constraint_levels_figure(api, saved_runs, window_width=5):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()

    fig_2 = plt.figure()
    gs_2 = fig_2.add_gridspec(1, 1)
    ax_2 = gs_2.subplots()

    fig_3 = plt.figure()
    gs_3 = fig_3.add_gridspec(1, 1)
    ax_3 = gs_3.subplots()

    returns_schedule, num_episodes_with_violations_schedule = [], []
    num_violations_schedule = []
    for seed in saved_runs.constraint_schedule.seeds:
        run = api.run(seed.id)
        history = run.scan_history(keys=["Num episodes with constraint violations"])
        num_episodes_with_violations = []
        num_violations = []
        for row in history:
            if not math.isnan(row["Num episodes with constraint violations"]):
                num_episodes_with_violations.append(
                    row["Num episodes with constraint violations"]
                )
                try:
                    num_violations.append(
                        num_episodes_with_violations[-1]
                        - num_episodes_with_violations[-2]
                    )
                except:
                    num_violations.append(num_episodes_with_violations[-1])

        history = run.scan_history(keys=["Extrinsic return"])
        returns = []
        for row in history:
            if not math.isnan(row["Extrinsic return"]):
                returns.append(row["Extrinsic return"])
        returns_schedule.append(returns)
        num_episodes_with_violations_schedule.append(num_episodes_with_violations)
        num_violations_schedule.append(num_violations)

    for level in saved_runs.constraint_levels:
        returns_all, num_episodes_with_violations_all = [], []
        num_violations_all = []
        print("delta={}".format(level.delta))
        for seed in level.ids:
            print("seed: {}".format(seed))
            run = api.run(seed)
            history = run.scan_history(keys=["Num episodes with constraint violations"])
            num_episodes_with_violations = []
            num_violations = []
            for row in history:
                if not math.isnan(row["Num episodes with constraint violations"]):
                    num_episodes_with_violations.append(
                        row["Num episodes with constraint violations"]
                    )
                    try:
                        num_violations.append(
                            num_episodes_with_violations[-1]
                            - num_episodes_with_violations[-2]
                        )
                    except:
                        num_violations.append(num_episodes_with_violations[-1])

            history = run.scan_history(keys=["Extrinsic return"])
            returns = []
            for row in history:
                if not math.isnan(row["Extrinsic return"]):
                    returns.append(row["Extrinsic return"])

            num_episodes_with_violations_all.append(num_episodes_with_violations)
            num_violations_all.append(num_violations)
            returns_all.append(returns)

        def plot(ax, values, label=""):
            min_len = len(values[0])
            for val in values:
                cumsum_vec = np.cumsum(np.insert(val, 0, 0))
                ma_vec = (
                    cumsum_vec[window_width:] - cumsum_vec[:-window_width]
                ) / window_width
                if len(ma_vec) < min_len:
                    min_len = len(ma_vec)

            values_same_length = []
            for val in values:
                cumsum_vec = np.cumsum(np.insert(val, 0, 0))
                ma_vec = (
                    cumsum_vec[window_width:] - cumsum_vec[:-window_width]
                ) / window_width
                values_same_length.append(ma_vec[0:min_len])
                # values_same_length.append(val[0:min_len])

            num_episodes = len(values_same_length[0])
            episodes = np.arange(0, num_episodes)

            values_same_length = np.stack(values_same_length, 0)
            values_mean = np.mean(values_same_length, 0)
            values_var = np.var(values_same_length, 0)
            ax.plot(episodes, values_mean, label=label)
            ax.fill_between(
                episodes,
                values_mean - 1.96 * np.sqrt(values_var),
                values_mean + 1.96 * np.sqrt(values_var),
                alpha=0.2,
            )

        plot(ax=ax, values=returns_all, label="$\delta={}$".format(level.delta))
        plot(
            ax=ax_2,
            values=num_episodes_with_violations_all,
            label="$\delta={}$".format(level.delta),
        )
        plot(
            ax=ax_3,
            values=num_violations_all,
            label="$\delta={}$".format(level.delta),
        )
    plot(
        ax=ax,
        values=returns_schedule,
        label=r"$\delta^{s}_{0}="
        + str(saved_runs.constraint_schedule.delta_start)
        + "$",
    )
    plot(
        ax=ax_2,
        values=num_episodes_with_violations_schedule,
        label=r"$\delta^{s}_{0}="
        + str(saved_runs.constraint_schedule.delta_start)
        + "$",
    )
    plot(
        ax=ax_3,
        values=num_violations_schedule,
        label=r"$\delta^{s}_{0}="
        + str(saved_runs.constraint_schedule.delta_start)
        + "$",
    )

    # plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=False)
    ax.set_xlabel(r"Episode $i$")
    ax.set_xlim(0, 150)
    ax.set_ylim(-3000, 100)
    ax.set_ylabel(
        r"Episode return",
        # r"Episode return $\sum_{t=0}^{t} r(\mathbf{s}_{t}, \mathbf{a}_{t})$"
    )
    ax.legend()
    fig.tight_layout(pad=0.5)

    # fig_2.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=False)
    ax_2.set_xlabel(r"Episode $i$")
    ax_2.set_xlim(0, 150)
    ax_2.set_ylim(0, 200)
    # ax_2.set_xlim(0, 150)
    ax_2.set_ylabel(r"Accumulated constraint violations $N_{\alpha}$")
    ax_2.legend()
    fig_2.tight_layout(pad=0.5)

    # fig_2.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=False)
    ax_3.set_xlabel(r"Episode $i$")
    ax_3.set_xlim(0, 150)
    # ax_3.set_ylim(0, 200)
    ax_3.set_ylabel(r"Average constraint violations")
    # ax_3.legend()
    fig_3.tight_layout(pad=0.5)

    return fig, fig_2, fig_3


def custom_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([1], [1], color="r", alpha=1, linestyle="dashed"))
    labels.append("Mode boundary")
    handles.append(mpl.lines.Line2D([1], [1], color="k"))
    labels.append(r"$\delta$-mode constraint")
    by_label = OrderedDict(zip(labels, handles))
    return by_label
