#!/usr/bin/env python3
import distutils
import logging
import os
from collections import OrderedDict
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from gpflow import default_float
from modeopt.metrics import (
    approximate_riemannian_energy,
    gating_function_variance,
    mode_probability,
    state_variance,
)
from modeopt.mode_opt import ModeOpt
from modeopt.plotting import ModeOptContourPlotter
from modeopt.utils import combine_state_controls_to_input
from velocity_controlled_point_mass.mode_opt_riemannian_energy_traj_opt import (
    create_test_inputs,
)


def plot_trajectories_over_desired_mixing_prob(iteration):
    fig = plt.figure(
        figsize=(
            plotter.mosvgpe_plotter.figsize[0] / 2,
            plotter.mosvgpe_plotter.figsize[1],
        )
    )
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots(sharex=True, sharey=True)
    mixing_probs = mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs(
        test_inputs
    )
    if iteration == 0:
        label = (
            "$\Pr(\\alpha="
            + str(mode_optimiser.desired_mode + 1)
            + " \mid \mathbf{x}, \mathcal{D}_{0})$"
        )
    else:
        label = (
            "$\Pr(\\alpha="
            + str(mode_optimiser.desired_mode + 1)
            + " \mid \mathbf{x}, \mathcal{D}_{0:"
            + str(iteration)
            + "})$"
        )
    plotter.mosvgpe_plotter.plot_contf(
        ax, mixing_probs[:, mode_optimiser.desired_mode], label=label
    )
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    plotter.plot_env_no_obs_start_end_given_fig(fig)
    plotter.plot_trajectories_given_fig(fig, plot_satisfaction_prob=True)
    #     fig.get_gca().get_legend().remove()
    fig.legend_ = None
    fig.tight_layout()
    return fig


def plot_trajectories_over_gating_function_variance(iteration):
    fig = plt.figure(
        figsize=(
            plotter.mosvgpe_plotter.figsize[0] / 2,
            plotter.mosvgpe_plotter.figsize[1],
        )
    )
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots(sharex=True, sharey=True)
    h_means, h_vars = mode_optimiser.dynamics.mosvgpe.gating_network.predict_h(
        test_inputs
    )
    if iteration == 0:
        label = (
            "$\mathbb{V}[h_{"
            + str(mode_optimiser.desired_mode + 1)
            + "}(\mathbf{x}) \mid \mathcal{D}_{0}]$"
        )
    else:
        label = (
            "$\mathbb{V}[h_{"
            + str(mode_optimiser.desired_mode + 1)
            + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
            + str(iteration)
            + "}]$"
        )

    plotter.mosvgpe_plotter.plot_contf(
        ax,
        h_vars[:, mode_optimiser.desired_mode],
        label=label,
    )
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    #     plotter.plot_env_no_obs_start_end_given_fig(fig)
    plotter.plot_trajectories_given_fig(fig, plot_satisfaction_prob=True)

    #     fig.get_legend().remove()
    fig.tight_layout()
    return fig


def plot_data_over_desired_mixing_prob(iteration, epoch):
    fig = plt.figure(
        figsize=(
            plotter.mosvgpe_plotter.figsize[0] / 2,
            plotter.mosvgpe_plotter.figsize[1],
        )
    )
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots(sharex=True, sharey=True)
    mixing_probs = mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs(
        test_inputs
    )
    if iteration == 0:
        label = (
            "$\mathcal{D}_{0} \\text{ over } \Pr(\\alpha="
            + str(mode_optimiser.desired_mode + 1)
            + " \mid \mathbf{x}, \mathcal{D}_{0})$"
        )
    else:
        if epoch == 0:
            if iteration - 1 == 0:
                label = (
                    "$\mathcal{D}_{0:"
                    + str(iteration)
                    + "} \\text{ over } \Pr(\\alpha="
                    + str(mode_optimiser.desired_mode + 1)
                    + " \mid \mathbf{x}, \mathcal{D}_{0})$"
                )
            else:
                label = (
                    "$\mathcal{D}_{0:"
                    + str(iteration)
                    + "} \\text{ over } \Pr(\\alpha="
                    + str(mode_optimiser.desired_mode + 1)
                    + " \mid \mathbf{x}, \mathcal{D}_{0:"
                    + str(iteration - 1)
                    + "})$"
                )
        else:
            label = (
                "$\mathcal{D}_{0:"
                + str(iteration)
                + "} \\text{ over } \Pr(\\alpha="
                + str(mode_optimiser.desired_mode + 1)
                + " \mid \mathbf{x}, \mathcal{D}_{0:"
                + str(iteration)
                + "})$"
            )
    plotter.mosvgpe_plotter.plot_contf(
        ax, mixing_probs[:, mode_optimiser.desired_mode], label=label
    )
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    plotter.plot_env_no_obs_start_end_given_fig(fig)
    plotter.plot_data_given_fig(fig)
    #     ax.get_legend().remove()
    fig.tight_layout()
    return fig


def plot_data_over_gating_function_variance(iteration, epoch):
    fig = plt.figure(
        figsize=(
            plotter.mosvgpe_plotter.figsize[0] / 2,
            plotter.mosvgpe_plotter.figsize[1],
        )
    )
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots(sharex=True, sharey=True)
    h_means, h_vars = mode_optimiser.dynamics.mosvgpe.gating_network.predict_h(
        test_inputs
    )
    if iteration == 0:
        label = (
            "$\mathcal{D}_{0} \\text{ over } \mathbb{V}[h_{"
            + str(mode_optimiser.desired_mode + 1)
            + "}(\mathbf{x}) \mid \mathcal{D}_{0}]$"
        )
    else:
        if epoch == 0:
            if iteration - 1 == 0:
                label = (
                    "$\mathcal{D}_{0:"
                    + str(iteration)
                    + "} \\text{ over } \mathbb{V}[h_{"
                    + str(mode_optimiser.desired_mode + 1)
                    + "}(\mathbf{x}) \mid \mathcal{D}_{0}]$"
                )
            else:
                label = (
                    "$\mathcal{D}_{0:"
                    + str(iteration)
                    + "} \\text{ over } \mathbb{V}[h_{"
                    + str(mode_optimiser.desired_mode + 1)
                    + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
                    + str(iteration - 1)
                    + "}]$"
                )
        else:
            label = (
                "$\mathcal{D}_{0:"
                + str(iteration)
                + "} \\text{ over } \mathbb{V}[h_{"
                + str(mode_optimiser.desired_mode + 1)
                + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
                + str(iteration)
                + "}]$"
            )

    plotter.mosvgpe_plotter.plot_contf(
        ax, h_vars[:, mode_optimiser.desired_mode], label=label
    )
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    plotter.plot_data_given_fig(fig)
    fig.tight_layout()
    return fig


def plot_data_over_gating_network_and_save(step: int, epoch: int, save: bool = True):
    plot_data_over_desired_mixing_prob(iteration=step, epoch=epoch)
    if save:
        save_filename = os.path.join(
            image_save_dir,
            "data_over_desired_mixing_probs_step_{}_epoch_{}.pdf".format(step, epoch),
        )
        plt.savefig(save_filename, transparent=True)

    plot_data_over_gating_function_variance(iteration=step, epoch=epoch)
    if save:
        save_filename = os.path.join(
            image_save_dir,
            "data_over_gating_variance_step_{}_epoch_{}.pdf".format(step, epoch),
        )
        plt.savefig(save_filename, transparent=True)


def plot_trajectories_over_gating_network_and_save(step: int, save: bool = True):
    plot_trajectories_over_desired_mixing_prob(iteration=step - 1)
    if save:
        save_filename = os.path.join(
            image_save_dir,
            "trajectories_over_desired_mixing_probs_step_{}.pdf".format(step),
        )
        plt.savefig(save_filename, transparent=True)

    #     plotter.plot_trajectories_over_desired_gating_network_gp(plot_satisfaction_prob=True)
    plot_trajectories_over_gating_function_variance(iteration=step - 1)
    if save:
        save_filename = os.path.join(
            image_save_dir, "trajectories_over_gating_variance_step_{}.pdf".format(step)
        )
        plt.savefig(save_filename, transparent=True)


if __name__ == "__main__":
    path_to_experiment = "./velocity_controlled_point_mass/notebooks/experiments/mvn-full-cov/explorative-prob-0.8/2022-04-18-12-32/ckpts"
    path_to_saved_experiment = "./velocity_controlled_point_mass/saved_experiments/mvn-full-cov/explorative-prob-0.8/ckpts"
    distutils.dir_util.copy_tree(path_to_experiment, path_to_saved_experiment)

    image_save_dir = "./velocity_controlled_point_mass/notebooks/reports/figures/mvn-full-cov/explorative-prob-0.8"
    os.makedirs(image_save_dir, exist_ok=True)

    test_inputs = create_test_inputs(
        x_min=[-3, -3], x_max=[3, 3], input_dim=4, num_test=8100
    )

    mode_optimiser = ModeOpt.load(path_to_saved_experiment)
    mode_chance_constraints = build_mode_chance_constraints_scipy(
        mode_optimiser.dynamics,
        mode_optimiser.start_state,
        horizon=15,
        control_dim=2,
        lower_bound=0.8,
    )
    explorative_controller = TrajectoryOptimisationController(
        max_iterations=max_iterations,
        initial_solution=initial_solution_in_mode,
        objective_fn=explorative_objective,
        keep_last_solution=keep_last_solution,
        constraints=[mode_chance_constraints],
        method=method,
    )
    mode_optimiser.explorative_controller = explorative_controller

    print(mode_optimiser.explorative_controller)

    num_iterations = 5
    for iteration in range(num_iterations):
        # experiment_i = os.path.join(
        #     path_to_saved_experiment,
        # )
        # mode_optimiser = ModeOpt.load(experiment_i)
        plotter = ModeOptContourPlotter(
            mode_optimiser=mode_optimiser,
            test_inputs=test_inputs,
            static_trajectories=True,
        )

        plot_data_over_gating_network_and_save(step=iteration, epoch=0, save=True)
        plot_data_over_gating_network_and_save(step=iteration, epoch="last", save=True)
