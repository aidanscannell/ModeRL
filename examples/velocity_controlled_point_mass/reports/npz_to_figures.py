#!/usr/bin/env python3
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from modeopt.plotting import ModeOptContourPlotter
import os

import numpy as np
from modeopt.metrics import (
    approximate_riemannian_energy,
    gating_function_variance,
    mode_probability,
    state_variance,
)
from modeopt.mode_opt import ModeOpt
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
COLORS = {
    "scenario_5/riemannian-energy": "blue",
    "scenario_5/riemannian-energy-low": "gray",
    "scenario_5/riemannian-energy-low-2": "green",
    "scenario_5/riemannian-energy-high": "orange",
    "scenario_5/geodesic-collocation": "red",
    "scenario_5/geodesic-collocation-mid-point": "purple",
    "scenario_5/geodesic-collocation-low": "brown",
    "scenario_5/geodesic-collocation-high": "pink",
    "scenario_5/control-as-inference": "olive",
    "scenario_5/control-as-inference-deterministic": "cyan",
    "scenario_7/riemannian-energy": "blue",
    "scenario_7/riemannian-energy-low": "gray",
    "scenario_7/riemannian-energy-high": "orange",
    "scenario_7/geodesic-collocation": "red",
    "scenario_7/geodesic-collocation-low": "brown",
    "scenario_7/geodesic-collocation-high": "pink",
    "scenario_7/control-as-inference": "olive",
    "scenario_7/control-as-inference-deterministic": "cyan",
}
LABELS = {
    "scenario_5/riemannian-energy": "DE $\lambda=1.0$",
    "scenario_5/riemannian-energy-low": "DE $\lambda=0.01$",
    "scenario_5/riemannian-energy-low-2": "DE $\lambda=0.5$",
    "scenario_5/riemannian-energy-high": "DE $\lambda=5.0$",
    "scenario_5/geodesic-collocation": "IG $\lambda=1.0$",
    "scenario_5/geodesic-collocation-mid-point": "IG $\lambda=1.0$ (mid point)",
    "scenario_5/geodesic-collocation-low": "IG $\lambda=0.5$",
    "scenario_5/geodesic-collocation-high": "IG $\lambda=5.0$",
    "scenario_7/riemannian-energy": "DE $\lambda=1.0$",
    "scenario_7/riemannian-energy-low": "DE $\lambda=1.0$",
    "scenario_7/riemannian-energy-high": "DE $\lambda=1.0$",
    "scenario_7/geodesic-collocation": "IG $\lambda=1.0$",
    "scenario_7/geodesic-collocation-low": "IG $\lambda=1.0$",
    "scenario_7/geodesic-collocation-high": "IG $\lambda=1.0$",
    "scenario_5/control-as-inference": "CaI (gauss)",
    "scenario_5/control-as-inference-deterministic": "CaI (dirac)",
    "scenario_7/control-as-inference": "CaI (gauss)",
    "scenario_7/control-as-inference-deterministic": "CaI (dirac)",
}

COLORS = {"env": "c", "dynamics": "m", "collocation": "y"}
# LINESTYLES = {"env": "-.", "dynamics": ":", "collocation": "-"}
MARKERS = {"env": "*", "dynamics": ".", "collocation": "d"}
low_linestyle = ""
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

mode_opt_ckpt_dirs = {
    5: "./velocity_controlled_point_mass/saved_experiments/scenario_5/trajectory_optimisation/riemannian_energy/2022.03.16/130358",
    7: "./velocity_controlled_point_mass/saved_experiments/scenario_7/trajectory_optimisation/riemannian_energy/2022.03.08/151633",
}
if __name__ == "__main__":
    scenario = 5
    scenario = 7
    save_filename = (
        "./velocity_controlled_point_mass/reports/figures/scenario_"
        + str(scenario)
        + ".pdf"
    )
    npz_traj_dir = (
        "./velocity_controlled_point_mass/reports/saved_trajectories/scenario_"
        + str(scenario)
    )
    mode_opt_ckpt_dir = mode_opt_ckpt_dirs[scenario]
    mode_optimiser = ModeOpt.load(os.path.join(mode_opt_ckpt_dir, "ckpts"))
    test_inputs = create_test_inputs(
        x_min=[-3, -3],
        x_max=[3, 3],
        input_dim=4,
        num_test=8100
        # num_test=100
        # num_test=10000
        # num_test=40000
    )
    mode_optimiser_plotter = ModeOptContourPlotter(
        mode_optimiser=mode_optimiser, test_inputs=test_inputs, static_trajectories=True
    )

    load_dir = (
        "./velocity_controlled_point_mass/reports/saved_trajectories/scenario_"
        + str(scenario)
    )

    def plot_grid():
        figsize = (12.8, 10.2)
        # figsize: Tuple[float, float] = (6.4, 3.4),
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2)
        axs = gs.subplots(sharex=True, sharey=True)

        mixing_probs = (
            mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                test_inputs
            )
        )
        for ax in axs[:, 0]:
            contf = plot_contf(ax, mixing_probs[:, mode_optimiser.desired_mode])
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(contf, use_gridspec=True, cax=cax, orientation="horizontal")
        cbar.set_label(
            "$\Pr(\\alpha="
            + str(mode_optimiser.desired_mode + 1)
            + " \mid \mathbf{x})$"
        )
        h_means, h_vars = mode_optimiser.dynamics.mosvgpe.gating_network.predict_h(
            test_inputs
        )
        # for ax in axs[1, :]:
        #     contf = plot_contf(ax, h_means[:, mode_optimiser.desired_mode])
        # divider = make_axes_locatable(axs[1, -1])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = plt.colorbar(contf, use_gridspec=True, cax=cax)
        # cbar.set_label(
        #     "$\mathbb{E}[h_{"
        #     + str(mode_optimiser.desired_mode + 1)
        #     + "} (\mathbf{x})]$"
        # )
        for ax in axs[:, 1]:
            contf = plot_contf(ax, h_vars[:, mode_optimiser.desired_mode])
        divider = make_axes_locatable(axs[0, 1])
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(contf, use_gridspec=True, cax=cax, orientation="horizontal")
        cbar.set_label(
            "$\mathbb{V}[h_{"
            + str(mode_optimiser.desired_mode + 1)
            + "} (\mathbf{x})]$"
        )

        for ax in axs.flat:
            mode_optimiser_plotter.plot_start_end_pos_given_ax(ax)
            mode_optimiser_plotter.plot_env_given_ax(ax)
        for ax in axs[-1, :]:
            ax.set_xlabel("$x$")
        for ax in axs[:, 0]:
            ax.set_ylabel("$y$")
        fig.tight_layout()
        axs[0, 0].set_title("Indirect Optimal Control")
        axs[1, 0].set_title("Direct Optimal Control")
        axs[2, 0].set_title("Control as Inference")
        # axs[k].set_title("$\Pr(\\alpha=" + str(k + 1) + " \mid \mathbf{x})$")
        # [ax.set_ylabel("") for ax in axs[1:].flat]
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

    fig, axs = plot_grid()

    trajectories = {}
    for filename in os.listdir(npz_traj_dir):

        # trajectories.update({filename: np.load(os.path.join(load_dir, filename))})
        trajectories = np.load(os.path.join(load_dir, filename))
        if "geodesic" in filename:
            for ax in axs[0, :].flat:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                ax.plot(
                    trajectories["collocation"][:, 0],
                    trajectories["collocation"][:, 1],
                    label=LABELS[key],
                    color=COLORS["collocation"],
                    # color=COLORS[key],
                    linestyle=LINESTYLES[key],
                    linewidth=0.3,
                    marker=MARKERS["collocation"],
                    # marker=MARKERS[key],
                )
                ax.plot(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    # color=COLORS[key],
                    color=COLORS["dynamics"],
                    linestyle=LINESTYLES[key],
                    linewidth=0.3,
                    marker=MARKERS["dynamics"],
                    # marker=MARKERS[key],
                )
                ax.plot(
                    trajectories["env"][:, 0],
                    trajectories["env"][:, 1],
                    label=LABELS[key],
                    color=COLORS["env"],
                    # color=COLORS[key],
                    linestyle=LINESTYLES[key],
                    linewidth=0.3,
                    marker=MARKERS["env"],
                    # marker=MARKERS[key],
                )
        if "riemannian" in filename:
            for ax in axs[1, :].flat:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                ax.plot(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    # color=COLORS[key],
                    color=COLORS["dynamics"],
                    linestyle=LINESTYLES[key],
                    linewidth=0.3,
                    marker=MARKERS["dynamics"],
                    # marker=MARKERS[key],
                )
                ax.plot(
                    trajectories["env"][:, 0],
                    trajectories["env"][:, 1],
                    label=LABELS[key],
                    # color=COLORS[key],
                    color=COLORS["env"],
                    linestyle=LINESTYLES[key],
                    linewidth=0.3,
                    marker=MARKERS["env"],
                    # marker=MARKERS[key],
                )
        if "inference" in filename:
            for ax in axs[-1, :].flat:
                key = "scenario_" + str(scenario) + "/" + filename.split(".npz")[0]
                ax.plot(
                    trajectories["dynamics"][:, 0],
                    trajectories["dynamics"][:, 1],
                    label=LABELS[key],
                    # color=COLORS[key],
                    color=COLORS["dynamics"],
                    linestyle=LINESTYLES[key],
                    linewidth=0.3,
                    marker=MARKERS["dynamics"],
                    # marker=MARKERS[key],
                )
                ax.plot(
                    trajectories["env"][:, 0],
                    trajectories["env"][:, 1],
                    label=LABELS[key],
                    # color=COLORS[key],
                    color=COLORS["env"],
                    linestyle=LINESTYLES[key],
                    linewidth=0.3,
                    marker=MARKERS["env"],
                    # marker=MARKERS[key],
                )

        # print(trajectories[filename]["env"])
        # print(trajectories[filename]["dynamics"])
    # print(trajectories)

    plt.savefig(save_filename, transparent=True)
    # plt.show()
