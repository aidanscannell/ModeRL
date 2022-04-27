#!/usr/bin/env python3
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from modeopt.metrics import (
    approximate_riemannian_energy,
    gating_function_variance,
    mode_probability,
    state_variance,
)
from modeopt.mode_opt import ModeOpt
from modeopt.plotting import ModeOptContourPlotter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from velocity_controlled_point_mass.mode_opt_riemannian_energy_traj_opt import (
    create_test_inputs,
)

de_linestyle = "-"
ig_linestyle = "--"
cai_linestyle = "-."
LINESTYLES = {
    "scenario_5/riemannian-energy": de_linestyle,
    "scenario_5/riemannian-energy-low": de_linestyle,
    "scenario_5/riemannian-energy-low-2": de_linestyle,
    "scenario_5/riemannian-energy-high": de_linestyle,
    "scenario_5/geodesic-collocation": ig_linestyle,
    "scenario_5/geodesic-collocation-mid-point": ig_linestyle,
    "scenario_5/geodesic-collocation-low": ig_linestyle,
    "scenario_5/geodesic-collocation-high": ig_linestyle,
    "scenario_5/control-as-inference": cai_linestyle,
    "scenario_5/control-as-inference-deterministic": cai_linestyle,
    "scenario_7/riemannian-energy": de_linestyle,
    "scenario_7/riemannian-energy-low": de_linestyle,
    "scenario_7/riemannian-energy-high": de_linestyle,
    "scenario_7/geodesic-collocation": ig_linestyle,
    "scenario_7/geodesic-collocation-low": ig_linestyle,
    "scenario_7/geodesic-collocation-high": ig_linestyle,
    "scenario_7/control-as-inference": cai_linestyle,
    "scenario_7/control-as-inference-deterministic": cai_linestyle,
}
color_low = "blue"
color_medium = "green"
color_high = "yellow"
color_high = "darkorange"
COLORS = {
    "scenario_5/riemannian-energy": color_medium,
    "scenario_5/riemannian-energy-low": "brown",
    "scenario_5/riemannian-energy-low-2": color_low,
    "scenario_5/riemannian-energy-high": "red",
    "scenario_5/geodesic-collocation": color_medium,
    "scenario_5/geodesic-collocation-mid-point": "purple",
    "scenario_5/geodesic-collocation-low": color_low,
    "scenario_5/geodesic-collocation-high": "red",
    "scenario_5/control-as-inference": "black",
    "scenario_5/control-as-inference-deterministic": "pink",
    "scenario_7/riemannian-energy": color_medium,
    "scenario_7/riemannian-energy-low": color_low,
    "scenario_7/riemannian-energy-high": color_high,
    "scenario_7/geodesic-collocation": color_medium,
    "scenario_7/geodesic-collocation-low": color_low,
    "scenario_7/geodesic-collocation-high": color_high,
    "scenario_7/control-as-inference": "black",
    "scenario_7/control-as-inference-deterministic": "pink",
}
LABELS = {
    "scenario_5/riemannian-energy": "DRE $\lambda=1.0$",
    "scenario_5/riemannian-energy-low": "DRE $\lambda=0.01$",
    "scenario_5/riemannian-energy-low-2": "DRE $\lambda=0.5$",
    "scenario_5/riemannian-energy-high": "DRE $\lambda=5.0$",
    "scenario_5/geodesic-collocation": "IG $\lambda=1.0$",
    "scenario_5/geodesic-collocation-mid-point": "IG $\lambda=1.0$ (mid point)",
    "scenario_5/geodesic-collocation-low": "IG $\lambda=0.5$",
    "scenario_5/geodesic-collocation-high": "IG $\lambda=5.0$",
    "scenario_7/riemannian-energy": "DRE $\lambda=1.0$",
    "scenario_7/riemannian-energy-low": "DRE $\lambda=0.5$",
    "scenario_7/riemannian-energy-high": "DRE $\lambda=20.0$",
    "scenario_7/geodesic-collocation": "IG $\lambda=1.0$",
    "scenario_7/geodesic-collocation-low": "IG $\lambda=0.5$",
    "scenario_7/geodesic-collocation-high": "IG $\lambda=20.0$",
    "scenario_5/control-as-inference": "CaI (gauss)",
    "scenario_5/control-as-inference-deterministic": "CaI (dirac)",
    "scenario_7/control-as-inference": "CaI (gauss)",
    "scenario_7/control-as-inference-deterministic": "CaI (dirac)",
}

energy_marker = "*"
geodesic_marker = "d"
inference_marker = "."
# COLORS = {"env": "c", "dynamics": "m", "collocation": "y"}
# LINESTYLES = {"env": "-.", "dynamics": ":", "collocation": "-"}
# MARKERS = {"env": "*", "dynamics": ".", "collocation": "d"}
# MARKERS = {"env": "", "dynamics": ".", "collocation": "d"}
LINESTYLES = {"env": "--", "dynamics": "-", "collocation": ":"}
low_linestyle = ""
# LINESTYLES = {
#     "scenario_5/riemannian-energy": de_linestyle,
#     "scenario_5/riemannian-energy-low": de_linestyle,
#     "scenario_5/riemannian-energy-low-2": de_linestyle,
#     "scenario_5/riemannian-energy-high": de_linestyle,
#     "scenario_5/geodesic-collocation": ig_linestyle,
#     "scenario_5/geodesic-collocation-mid-point": ig_linestyle,
#     "scenario_5/geodesic-collocation-low": ig_linestyle,
#     "scenario_5/geodesic-collocation-high": ig_linestyle,
#     "scenario_5/control-as-inference": cai_linestyle,
#     "scenario_5/control-as-inference-deterministic": cai_linestyle,
#     "scenario_7/riemannian-energy": de_linestyle,
#     "scenario_7/riemannian-energy-low": de_linestyle,
#     "scenario_7/riemannian-energy-high": de_linestyle,
#     "scenario_7/geodesic-collocation": ig_linestyle,
#     "scenario_7/geodesic-collocation-low": ig_linestyle,
#     "scenario_7/geodesic-collocation-high": ig_linestyle,
#     "scenario_7/control-as-inference": cai_linestyle,
#     "scenario_7/control-as-inference-deterministic": cai_linestyle,
# }

mode_opt_ckpt_dirs = {
    5: "./velocity_controlled_point_mass/saved_experiments/scenario_5/trajectory_optimisation/riemannian_energy/2022.03.16/130358",
    # 7: "./velocity_controlled_point_mass/saved_experiments/scenario_7/trajectory_optimisation/riemannian_energy/2022.03.08/151633",
    7: "./velocity_controlled_point_mass/saved_experiments/scenario_7/trajectory_optimisation/control_as_inference/2022.03.14/161004",
}
if __name__ == "__main__":
    scenario = 5
    # scenario = 7
    scenarios = [5, 7]
    test_inputs = create_test_inputs(
        x_min=[-3, -3],
        x_max=[3, 3],
        input_dim=4,
        num_test=8100
        # num_test=100
        # num_test=10000
        # num_test=40000
    )

    def add_cbar_to_ax(ax, contf, label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(contf, use_gridspec=True, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cbar.set_label(label)

    def plot_grid():
        figsize = (6.2, 6.7)
        # figsize = (12.8, 10.2)
        # figsize: Tuple[float, float] = (6.4, 3.4),
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2)
        axs = gs.subplots(sharex=True, sharey=True)

        mixing_probs = (
            mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                test_inputs
            )
        )
        h_means, h_vars = mode_optimiser.dynamics.mosvgpe.gating_network.predict_h(
            test_inputs
        )
        for ax in axs[:, 0]:
            # contf = plot_contf(ax, h_means[:, mode_optimiser.desired_mode])
            contf = plot_contf(ax, mixing_probs[:, mode_optimiser.desired_mode])
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(contf, use_gridspec=True, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cbar.set_label(
            "$\Pr(\\alpha="
            + str(mode_optimiser.desired_mode + 1)
            + " \mid \mathbf{x})$"
        )
        for ax in axs[:, 1]:
            contf = plot_contf(ax, h_vars[:, mode_optimiser.desired_mode])
        divider = make_axes_locatable(axs[0, 1])
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(contf, use_gridspec=True, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cbar.set_label(
            "$\mathbb{V}[h_{"
            + str(mode_optimiser.desired_mode + 1)
            + "} (\mathbf{x})]$"
        )

        for ax in axs.flat:
            mode_optimiser_plotter.plot_start_end_pos_given_ax(ax, bbox=True)
            mode_optimiser_plotter.plot_env_given_ax(ax)
        for ax in axs[-1, :]:
            ax.set_xlabel("$x$")
        axs[0, 0].set_ylabel("$y \quad (\\text{IG})$")
        axs[1, 0].set_ylabel("$y \quad (\\text{DRE})$")
        axs[2, 0].set_ylabel("$y \quad (\\text{CaI})$")
        return fig, axs

    def plot_contf(ax, z):
        contf = ax.tricontourf(
            test_inputs[:, 0],
            test_inputs[:, 1],
            z,
            levels=10,
            cmap=mode_optimiser_plotter.mosvgpe_plotter.cmap,
        )
        return contf

    mode_optimisers, mode_optimiser_plotters, all_trajectories = [], [], []
    for scenario in scenarios:
        # save_filename = (
        #     "./velocity_controlled_point_mass/reports/figures/all_trajectories_over_gating_funcion_scenario_"
        #     + str(scenario)
        #     + ".pdf"
        # )
        save_filename = (
            "./velocity_controlled_point_mass/reports/figures/all_trajectories_over_prob_and_variance_scenario_"
            + str(scenario)
            + ".pdf"
        )
        npz_traj_dir = (
            "./velocity_controlled_point_mass/reports/saved_trajectories/scenario_"
            + str(scenario)
        )
        mode_opt_ckpt_dir = mode_opt_ckpt_dirs[scenario]
        mode_optimiser = ModeOpt.load(os.path.join(mode_opt_ckpt_dir, "ckpts"))
        mode_optimiser_plotter = ModeOptContourPlotter(
            mode_optimiser=mode_optimiser,
            test_inputs=test_inputs,
            static_trajectories=True,
        )
        mode_opt_ckpt_dir = mode_opt_ckpt_dirs[scenario]
        mode_optimisers.append(mode_optimiser)
        mode_optimiser_plotters.append(mode_optimiser_plotter)

        load_dir = (
            "./velocity_controlled_point_mass/reports/saved_trajectories/scenario_"
            + str(scenario)
        )

        fig, axs = plot_grid()

        trajectories = {}
        for filename in os.listdir(npz_traj_dir):

            # trajectories.update({filename: np.load(os.path.join(load_dir, filename))})
            trajectories = np.load(os.path.join(load_dir, filename))
            if "geodesic" in filename:
                for ax in axs[0, :].flat:
                    key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                    ax.plot(
                        # ax.scatter(
                        trajectories["dynamics"][:, 0],
                        trajectories["dynamics"][:, 1],
                        label=LABELS[key],
                        color=COLORS[key],
                        # color=COLORS["dynamics"],
                        # linestyle=LINESTYLES[key],
                        linestyle=LINESTYLES["dynamics"],
                        linewidth=0.3,
                        marker=geodesic_marker,
                        # marker=MARKERS[key],
                    )
            if "riemannian" in filename:
                for ax in axs[1, :].flat:
                    key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                    ax.plot(
                        # ax.scatter(
                        trajectories["dynamics"][:, 0],
                        trajectories["dynamics"][:, 1],
                        label=LABELS[key],
                        color=COLORS[key],
                        # color=COLORS["dynamics"],
                        # linestyle=LINESTYLES[key],
                        linestyle=LINESTYLES["dynamics"],
                        linewidth=0.3,
                        marker=energy_marker,
                        # marker=MARKERS[key],
                    )
            if "inference" in filename:
                for ax in axs[-1, :].flat:
                    key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                    # ax.scatter(
                    ax.plot(
                        trajectories["dynamics"][:, 0],
                        trajectories["dynamics"][:, 1],
                        label=LABELS[key],
                        color=COLORS[key],
                        # color=COLORS["dynamics"],
                        # linestyle=LINESTYLES[key],
                        linestyle=LINESTYLES["dynamics"],
                        linewidth=0.3,
                        marker=inference_marker,
                        # marker=MARKERS[key],
                    )

        handles, labels = [], []
        for ax in axs.flat:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0.5, -0.1),
            loc="lower center",
            bbox_transform=fig.transFigure,
            # ncol=len(by_label),
            # ncol=int(len(by_label) / 3),
            ncol=3,
        )
        print(len(by_label))
        fig.tight_layout()
        plt.savefig(save_filename, transparent=True)

        def plot_collocation_grid():
            figsize = (6.7, 3.1)
            # figsize = (12.8, 10.2)
            # figsize: Tuple[float, float] = (6.4, 3.4),
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(1, 2)
            axs = gs.subplots(sharex=True, sharey=True)

            mixing_probs = (
                mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                    test_inputs
                )
            )
            h_means, h_vars = mode_optimiser.dynamics.mosvgpe.gating_network.predict_h(
                test_inputs
            )
            for ax in axs.flat:
                contf = plot_contf(ax, mixing_probs[:, mode_optimiser.desired_mode])
                mode_optimiser_plotter.plot_start_end_pos_given_ax(ax, bbox=True)
                mode_optimiser_plotter.plot_env_given_ax(ax)

            label = (
                "$\Pr(\\alpha="
                + str(mode_optimiser.desired_mode + 1)
                + " \mid \mathbf{x})$"
            )
            axs[0].set_title(
                "Collocation trajectory ($\\text{IG}_{\\text{collocation}}$) $\\bar{\mathbf{z}}$"
            )
            axs[1].set_title("Dynamics trajectory (IG) $\\bar{\mathbf{x}}$")
            # add_cbar_to_ax(axs[0], contf=contf, label=label)
            # label = (
            #     "Collocation trajectory over $\Pr(\\alpha="
            #     + str(mode_optimiser.desired_mode + 1)
            #     + " \mid \mathbf{x})$"
            # )
            # add_cbar_to_ax(axs[1], contf=contf, label=label)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(contf, use_gridspec=True, cax=cax)
            cbar.set_label(label)

            for ax in axs.flat:
                ax.set_xlabel("$x$")
            axs[0].set_ylabel("$y$")
            return fig, axs

        ##################
        # Plot collocation
        ##################
        fig, axs = plot_collocation_grid()
        trajectories = {}
        for filename in os.listdir(npz_traj_dir):

            # trajectories.update({filename: np.load(os.path.join(load_dir, filename))})
            trajectories = np.load(os.path.join(load_dir, filename))
            if "geodesic" in filename:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                axs[1].plot(
                    # ax.scatter(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    color=COLORS[key],
                    # color=COLORS["dynamics"],
                    # linestyle=LINESTYLES[key],
                    linestyle=LINESTYLES["dynamics"],
                    linewidth=0.3,
                    marker=geodesic_marker,
                    # marker=MARKERS[key],
                )
                label = (
                    "$\\text{IG}_{\\text{collocation}}$ " + LABELS[key].split("IG ")[1]
                )
                axs[0].plot(
                    trajectories["collocation"][:, 0],
                    trajectories["collocation"][:, 1],
                    label=label,
                    # label=LABELS[key] + " \\text{collocation}",
                    # label=LABELS[key],
                    color=COLORS[key],
                    # color=COLORS["dynamics"],
                    # linestyle=LINESTYLES[key],
                    linestyle="-",
                    # linestyle=LINESTYLES["collocation"],
                    linewidth=0.6,
                    # marker=geodesic_marker,
                    marker="s",
                    # marker=MARKERS[key],
                )

        handles, labels = [], []
        for ax in axs.flat:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0.5, -0.28),
            loc="lower center",
            bbox_transform=fig.transFigure,
            # ncol=len(by_label),
            # ncol=int(len(by_label) / 3),
            ncol=2,
        )
        print(len(by_label))
        fig.tight_layout()
        save_filename = (
            "./velocity_controlled_point_mass/reports/figures/collocation_trajectories_over_desired_prob_scenario_"
            + str(scenario)
            + ".pdf"
        )
        plt.savefig(save_filename, transparent=True)

    ##################
    # Plot both envs prob
    ##################
    save_filename = "./velocity_controlled_point_mass/reports/figures/all_trajectories_over_prob_both_scenarios.pdf"

    def plot_prob_grid():
        figsize = (6.2, 6.7)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2)
        axs = gs.subplots(sharex=True, sharey=True)

        mixing_probs = mode_optimisers[
            0
        ].dynamics.mosvgpe.gating_network.predict_mixing_probs(test_inputs)
        label = (
            "Environment 2 $\Pr(\\alpha="
            + str(mode_optimisers[0].desired_mode + 1)
            + " \mid \mathbf{x})$"
        )
        for ax in axs[:, 1]:
            contf = plot_contf(ax, mixing_probs[:, mode_optimisers[0].desired_mode])
        add_cbar_to_ax(axs[0, 1], contf, label)
        label = (
            "Environment 1 $\Pr(\\alpha="
            + str(mode_optimisers[1].desired_mode + 1)
            + " \mid \mathbf{x})$"
        )
        mixing_probs = mode_optimisers[
            1
        ].dynamics.mosvgpe.gating_network.predict_mixing_probs(test_inputs)
        for ax in axs[:, 0]:
            contf = plot_contf(ax, mixing_probs[:, mode_optimisers[1].desired_mode])
        add_cbar_to_ax(axs[0, 0], contf, label)

        for ax in axs[:, 1].flat:
            mode_optimiser_plotters[0].plot_start_end_pos_given_ax(ax, bbox=True)
            mode_optimiser_plotters[0].plot_env_given_ax(ax)
        for ax in axs[:, 0].flat:
            mode_optimiser_plotters[1].plot_start_end_pos_given_ax(ax, bbox=True)
            mode_optimiser_plotters[1].plot_env_given_ax(ax)
        for ax in axs[-1, :]:
            ax.set_xlabel("$x$")
        axs[0, 0].set_ylabel("$y \quad (\\text{IG})$")
        axs[1, 0].set_ylabel("$y \quad (\\text{DRE})$")
        axs[2, 0].set_ylabel("$y \quad (\\text{CaI})$")
        return fig, axs

    fig, axs = plot_prob_grid()
    for scenario, col in zip(scenarios, [1, 0]):
        npz_traj_dir = (
            "./velocity_controlled_point_mass/reports/saved_trajectories/scenario_"
            + str(scenario)
        )
        for filename in os.listdir(npz_traj_dir):
            trajectories = np.load(os.path.join(npz_traj_dir, filename))
            if "geodesic" in filename:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                axs[0, col].plot(
                    # ax.scatter(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    color=COLORS[key],
                    # color=COLORS["dynamics"],
                    # linestyle=LINESTYLES[key],
                    linestyle=LINESTYLES["dynamics"],
                    linewidth=0.3,
                    marker=geodesic_marker,
                    # marker=MARKERS[key],
                )
            if "riemannian" in filename:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                axs[1, col].plot(
                    # ax.scatter(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    color=COLORS[key],
                    # color=COLORS["dynamics"],
                    # linestyle=LINESTYLES[key],
                    linestyle=LINESTYLES["dynamics"],
                    linewidth=0.3,
                    marker=energy_marker,
                    # marker=MARKERS[key],
                )
            if "inference" in filename:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                # ax.scatter(
                axs[-1, col].plot(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    color=COLORS[key],
                    # color=COLORS["dynamics"],
                    # linestyle=LINESTYLES[key],
                    linestyle=LINESTYLES["dynamics"],
                    linewidth=0.3,
                    marker=inference_marker,
                    # marker=MARKERS[key],
                )

    handles, labels = [], []
    for ax in axs.flat:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, -0.15),
        loc="lower center",
        bbox_transform=fig.transFigure,
        # ncol=len(by_label),
        # ncol=int(len(by_label) / 3),
        ncol=3,
    )
    print(len(by_label))
    fig.tight_layout()
    plt.savefig(save_filename, transparent=True)

    def plot_variance_grid():
        figsize = (6.2, 6.7)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2)
        axs = gs.subplots(sharex=True, sharey=True)

        h_means, h_vars = mode_optimisers[0].dynamics.mosvgpe.gating_network.predict_h(
            test_inputs
        )
        label = (
            "Environment 2 $\mathbb{V}[h_{"
            + str(mode_optimisers[0].desired_mode + 1)
            + "} (\mathbf{x})]$"
        )
        for ax in axs[:, 1]:
            contf = plot_contf(ax, h_vars[:, mode_optimisers[0].desired_mode])
        add_cbar_to_ax(axs[0, 1], contf, label)
        label = (
            "Environment 1 $\mathbb{V}[h_{"
            + str(mode_optimisers[1].desired_mode + 1)
            + "} (\mathbf{x})]$"
        )
        h_means, h_vars = mode_optimisers[1].dynamics.mosvgpe.gating_network.predict_h(
            test_inputs
        )
        for ax in axs[:, 0]:
            contf = plot_contf(ax, h_vars[:, mode_optimisers[1].desired_mode])
        add_cbar_to_ax(axs[0, 0], contf, label)

        for ax in axs[:, 1].flat:
            mode_optimiser_plotters[0].plot_start_end_pos_given_ax(ax, bbox=True)
            mode_optimiser_plotters[0].plot_env_given_ax(ax)
        for ax in axs[:, 0].flat:
            mode_optimiser_plotters[1].plot_start_end_pos_given_ax(ax, bbox=True)
            mode_optimiser_plotters[1].plot_env_given_ax(ax)
        for ax in axs[-1, :]:
            ax.set_xlabel("$x$")
        axs[0, 0].set_ylabel("$y \quad (\\text{IG})$")
        axs[1, 0].set_ylabel("$y \quad (\\text{DRE})$")
        axs[2, 0].set_ylabel("$y \quad (\\text{CaI})$")
        return fig, axs

    fig, axs = plot_variance_grid()
    for scenario, col in zip(scenarios, [1, 0]):
        npz_traj_dir = (
            "./velocity_controlled_point_mass/reports/saved_trajectories/scenario_"
            + str(scenario)
        )
        for filename in os.listdir(npz_traj_dir):
            trajectories = np.load(os.path.join(npz_traj_dir, filename))
            if "geodesic" in filename:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                axs[0, col].plot(
                    # ax.scatter(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    color=COLORS[key],
                    # color=COLORS["dynamics"],
                    # linestyle=LINESTYLES[key],
                    linestyle=LINESTYLES["dynamics"],
                    linewidth=0.3,
                    marker=geodesic_marker,
                    # marker=MARKERS[key],
                )
            if "riemannian" in filename:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                axs[1, col].plot(
                    # ax.scatter(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    color=COLORS[key],
                    # color=COLORS["dynamics"],
                    # linestyle=LINESTYLES[key],
                    linestyle=LINESTYLES["dynamics"],
                    linewidth=0.3,
                    marker=energy_marker,
                    # marker=MARKERS[key],
                )
            if "inference" in filename:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                # ax.scatter(
                axs[-1, col].plot(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    color=COLORS[key],
                    # color=COLORS["dynamics"],
                    # linestyle=LINESTYLES[key],
                    linestyle=LINESTYLES["dynamics"],
                    linewidth=0.3,
                    marker=inference_marker,
                    # marker=MARKERS[key],
                )

    handles, labels = [], []
    for ax in axs.flat:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, -0.15),
        loc="lower center",
        bbox_transform=fig.transFigure,
        # ncol=len(by_label),
        # ncol=int(len(by_label) / 3),
        ncol=3,
    )
    print(len(by_label))
    fig.tight_layout()
    save_filename = "./velocity_controlled_point_mass/reports/figures/all_trajectories_over_gating_variance_both_scenarios.pdf"
    plt.savefig(save_filename, transparent=True)

    def plot_mean_grid():
        figsize = (6.2, 6.7)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2)
        axs = gs.subplots(sharex=True, sharey=True)

        h_means, h_vars = mode_optimisers[0].dynamics.mosvgpe.gating_network.predict_h(
            test_inputs
        )
        label = (
            "Environment 2 $\mathbb{E}[h_{"
            + str(mode_optimisers[0].desired_mode + 1)
            + "} (\mathbf{x})]$"
        )
        for ax in axs[:, 1]:
            contf = plot_contf(ax, h_means[:, mode_optimisers[0].desired_mode])
        add_cbar_to_ax(axs[0, 1], contf, label)
        label = (
            "Environment 1 $\mathbb{E}[h_{"
            + str(mode_optimisers[1].desired_mode + 1)
            + "} (\mathbf{x})]$"
        )
        h_means, h_vars = mode_optimisers[1].dynamics.mosvgpe.gating_network.predict_h(
            test_inputs
        )
        for ax in axs[:, 0]:
            contf = plot_contf(ax, h_means[:, mode_optimisers[1].desired_mode])
        add_cbar_to_ax(axs[0, 0], contf, label)

        for ax in axs[:, 1].flat:
            mode_optimiser_plotters[0].plot_start_end_pos_given_ax(ax, bbox=True)
            mode_optimiser_plotters[0].plot_env_given_ax(ax)
        for ax in axs[:, 0].flat:
            mode_optimiser_plotters[1].plot_start_end_pos_given_ax(ax, bbox=True)
            mode_optimiser_plotters[1].plot_env_given_ax(ax)
        for ax in axs[-1, :]:
            ax.set_xlabel("$x$")
        axs[0, 0].set_ylabel("$y \quad (\\text{IG})$")
        axs[1, 0].set_ylabel("$y \quad (\\text{DRE})$")
        axs[2, 0].set_ylabel("$y \quad (\\text{CaI})$")
        return fig, axs

    fig, axs = plot_mean_grid()
    for scenario, col in zip(scenarios, [1, 0]):
        npz_traj_dir = (
            "./velocity_controlled_point_mass/reports/saved_trajectories/scenario_"
            + str(scenario)
        )
        for filename in os.listdir(npz_traj_dir):
            trajectories = np.load(os.path.join(npz_traj_dir, filename))
            if "geodesic" in filename:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                axs[0, col].plot(
                    # ax.scatter(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    color=COLORS[key],
                    # color=COLORS["dynamics"],
                    # linestyle=LINESTYLES[key],
                    linestyle=LINESTYLES["dynamics"],
                    linewidth=0.3,
                    marker=geodesic_marker,
                    # marker=MARKERS[key],
                )
            if "riemannian" in filename:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                axs[1, col].plot(
                    # ax.scatter(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    color=COLORS[key],
                    # color=COLORS["dynamics"],
                    # linestyle=LINESTYLES[key],
                    linestyle=LINESTYLES["dynamics"],
                    linewidth=0.3,
                    marker=energy_marker,
                    # marker=MARKERS[key],
                )
            if "inference" in filename:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                # ax.scatter(
                axs[-1, col].plot(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    color=COLORS[key],
                    # color=COLORS["dynamics"],
                    # linestyle=LINESTYLES[key],
                    linestyle=LINESTYLES["dynamics"],
                    linewidth=0.3,
                    marker=inference_marker,
                    # marker=MARKERS[key],
                )

    handles, labels = [], []
    for ax in axs.flat:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, -0.15),
        loc="lower center",
        bbox_transform=fig.transFigure,
        # ncol=len(by_label),
        # ncol=int(len(by_label) / 3),
        ncol=3,
    )
    print(len(by_label))
    fig.tight_layout()
    save_filename = "./velocity_controlled_point_mass/reports/figures/all_trajectories_over_gating_mean_both_scenarios.pdf"
    plt.savefig(save_filename, transparent=True)
