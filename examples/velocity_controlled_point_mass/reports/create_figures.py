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

logging.basicConfig(filename="metrics.log", level=logging.INFO)
de_linestyle = "-"
ig_linestyle = "--"
cai_linestyle = "-."
linestyles_dict = {
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
colors_dict = {
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
labels_dict = {
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


def plot_mode_opt_mode_controller(mode_optimiser: ModeOpt, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
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
    # mode_optimiser_plotter.plot_model()
    mode_optimiser_plotter.plot_trajectories_over_gating_network_gps()
    plt.savefig(
        os.path.join(save_dir, "trajectories_over_gating_gps.pdf"), transparent=True
    )
    mode_optimiser_plotter.plot_trajectories_over_desired_gating_network_gp()
    plt.savefig(
        os.path.join(save_dir, "trajectories_over_desired_gating_gp.pdf"),
        transparent=True,
    )
    mode_optimiser_plotter.plot_trajectories_over_mixing_probs()
    plt.savefig(
        os.path.join(save_dir, "trajectories_over_mixing_probs.pdf"), transparent=True
    )
    mode_optimiser_plotter.plot_trajectories_over_desired_mixing_prob()
    plt.savefig(
        os.path.join(save_dir, "trajectories_over_desired_mixing_prob.pdf"),
        transparent=True,
    )
    # mode_optimiser_plotter.plot_trajectories_over_metric_trace()
    # plt.savefig(
    #     os.path.join(save_dir, "trajectories_over_metric_trace.pdf"), transparent=True
    # )
    # plt.show()
    return mode_optimiser_plotter.trajectories


def calculate_metrics(
    mode_optimiser: ModeOpt, sum: bool = True, name: str = ""
) -> None:
    prob = mode_probability(mode_optimiser, sum=sum)
    prob_no_state_unc = mode_probability(
        mode_optimiser, marginalise_state=False, sum=sum
    )
    prob_no_gating_unc = mode_probability(
        mode_optimiser, marginalise_gating_func=False, sum=sum
    )
    prob_no_gating_or_state_unc = mode_probability(
        mode_optimiser, marginalise_state=False, marginalise_gating_func=False, sum=sum
    )
    state_var = state_variance(mode_optimiser, sum=sum)
    gating_variance = gating_function_variance(mode_optimiser, sum=sum)
    gating_variance_no_state_unc = gating_function_variance(
        mode_optimiser, marginalise_state=False, sum=sum
    )
    riemannian_energy = approximate_riemannian_energy(mode_optimiser, sum=sum)

    with open(
        "./velocity_controlled_point_mass/reports/performance_metrics.txt", "a"
    ) as f:
        f.write(name + "\n")
        f.write("Desired prob: {}\n".format(prob))
        f.write("Desired prob, no state variance: {}\n".format(prob_no_state_unc))
        f.write("Desired prob, no gating variance: {}\n".format(prob_no_gating_unc))
        f.write(
            "Desired prob, no state or  gating variance: {}\n".format(
                prob_no_gating_or_state_unc
            )
        )
        f.write("State variance: {}\n".format(state_var))
        f.write("Gating function variance: {}\n".format(gating_variance))
        f.write(
            "Gating function variance, no state variance: {}\n".format(
                gating_variance_no_state_unc
            )
        )
        f.write("Riemannian energy: {}\n\n".format(riemannian_energy))


if __name__ == "__main__":
    path_to_saved_experiments = "./velocity_controlled_point_mass/saved_experiments"
    path_to_experiments = "./velocity_controlled_point_mass/experiments"

    experiment_dict = {
        "riemannian-energy": "scenario_7/trajectory_optimisation/riemannian_energy/2022.03.07/151334",
        "riemannian-energy-low": "scenario_7/trajectory_optimisation/riemannian_energy_low/2022.03.07/165349",
        "riemannian-energy-high": "scenario_7/trajectory_optimisation/riemannian_energy_high/2022.03.07/165924",
        # "geodesic-collocation": "scenario_7/trajectory_optimisation/geodesic_collocation/2022.03.04/164916",
        # "geodesic-collocation-low": "scenario_7/trajectory_optimisation/geodesic_collocation/2022.03.04/164916",
        "geodesic-collocation-high": "scenario_7/trajectory_optimisation/geodesic_collocation_high/2022.03.07/160728",
    }
    experiment_dict = {
        # "scenario_5/riemannian-energy": "scenario_5/trajectory_optimisation/riemannian_energy/2022.03.09/121822",
        # "scenario_5/riemannian-energy-low": "scenario_5/trajectory_optimisation/riemannian_energy_low/2022.03.09/121550",
        # "scenario_5/riemannian-energy-high": "scenario_5/trajectory_optimisation/riemannian_energy_high/2022.03.09/122350",
        # "scenario_5/riemannian-energy": "scenario_5/trajectory_optimisation/riemannian_energy/2022.03.15/123140",
        # "scenario_5/riemannian-energy-low": "scenario_5/trajectory_optimisation/riemannian_energy_low/2022.03.15/122734",
        # "scenario_5/riemannian-energy-low-2": "scenario_5/trajectory_optimisation/riemannian_energy_low_2/2022.03.15/123607",
        # "scenario_5/riemannian-energy-high": "scenario_5/trajectory_optimisation/riemannian_energy_high/2022.03.15/122943",
        "scenario_5/riemannian-energy": "scenario_5/trajectory_optimisation/riemannian_energy/2022.03.16/130358",
        "scenario_5/riemannian-energy-low": "scenario_5/trajectory_optimisation/riemannian_energy_low/2022.03.16/131428",
        "scenario_5/riemannian-energy-low-2": "scenario_5/trajectory_optimisation/riemannian_energy_low_2/2022.03.16/131442",
        "scenario_5/riemannian-energy-high": "scenario_5/trajectory_optimisation/riemannian_energy_high/2022.03.16/130011",
        "scenario_5/geodesic-collocation": "scenario_5/trajectory_optimisation/geodesic_collocation/2022.03.18/123518",
        "scenario_5/geodesic-collocation-mid-point": "scenario_5/trajectory_optimisation/geodesic_collocation/2022.03.18/125548",
        "scenario_5/geodesic-collocation-low": "scenario_5/trajectory_optimisation/geodesic_collocation_low/2022.03.18/154418",
        "scenario_5/geodesic-collocation-high": "scenario_5/trajectory_optimisation/geodesic_collocation_high/2022.03.18/121855",
        "scenario_7/riemannian-energy": "scenario_7/trajectory_optimisation/riemannian_energy/2022.03.08/151633",
        # "scenario_7/riemannian-energy-low": "scenario_7/trajectory_optimisation/riemannian_energy_low/2022.03.08/153724",
        "scenario_7/riemannian-energy-low": "scenario_7/trajectory_optimisation/riemannian_energy_low/2022.03.08/195703",
        # "scenario_7/riemannian-energy-high": "scenario_7/trajectory_optimisation/riemannian_energy_high/2022.03.08/151851",
        "scenario_7/riemannian-energy-high": "scenario_7/trajectory_optimisation/riemannian_energy_high/2022.03.09/114346",
        "scenario_7/geodesic-collocation": "scenario_7/trajectory_optimisation/geodesic_collocation/2022.03.08/174538",
        "scenario_7/geodesic-collocation-low": "scenario_7/trajectory_optimisation/geodesic_collocation_low/2022.03.08/171012",
        "scenario_7/geodesic-collocation-high": "scenario_7/trajectory_optimisation/geodesic_collocation_high/2022.03.09/113857",
        # "scenario_5/control-as-inference": "scenario_5/trajectory_optimisation/control_as_inference/2022.03.14/154427",
        # "scenario_5/control-as-inference-deterministic": "scenario_5/trajectory_optimisation/control_as_inference_deterministic/2022.03.14/154836",
        "scenario_5/control-as-inference": "scenario_5/trajectory_optimisation/control_as_inference/2022.03.16/133955",
        "scenario_5/control-as-inference-deterministic": "scenario_5/trajectory_optimisation/control_as_inference_deterministic/2022.03.16/133940",
        "scenario_7/control-as-inference": "scenario_7/trajectory_optimisation/control_as_inference/2022.03.14/161004",
        "scenario_7/control-as-inference-deterministic": "scenario_7/trajectory_optimisation/control_as_inference_deterministic/2022.03.14/160121",
    }
    experiment_dict = {
        "scenario_5/riemannian-energy-low": "scenario_5/trajectory_optimisation/riemannian_energy_low/2022.03.16/131428",
        "scenario_5/riemannian-energy-low-2": "scenario_5/trajectory_optimisation/riemannian_energy_low_2/2022.03.16/131442",
        "scenario_5/riemannian-energy": "scenario_5/trajectory_optimisation/riemannian_energy/2022.03.16/130358",
        "scenario_5/riemannian-energy-high": "scenario_5/trajectory_optimisation/riemannian_energy_high/2022.03.16/130011",
        "scenario_5/geodesic-collocation-low": "scenario_5/trajectory_optimisation/geodesic_collocation_low/2022.03.18/154418",
        "scenario_5/geodesic-collocation": "scenario_5/trajectory_optimisation/geodesic_collocation/2022.03.18/123518",
        "scenario_5/geodesic-collocation-mid-point": "scenario_5/trajectory_optimisation/geodesic_collocation/2022.03.18/125548",
        "scenario_5/geodesic-collocation-high": "scenario_5/trajectory_optimisation/geodesic_collocation_high/2022.03.18/121855",
        "scenario_5/control-as-inference": "scenario_5/trajectory_optimisation/control_as_inference/2022.03.16/133955",
        "scenario_5/control-as-inference-deterministic": "scenario_5/trajectory_optimisation/control_as_inference_deterministic/2022.03.16/133940",
    }

    trajectories = {}
    for key in experiment_dict.keys():
        save_dir = os.path.join(
            "./velocity_controlled_point_mass/reports/figures/trajectory_optimisation",
            key,
        )
        logging.info(save_dir)
        ckpt_dir = os.path.join(path_to_experiments, experiment_dict[key])
        saved_exp_dir = os.path.join(path_to_saved_experiments, experiment_dict[key])
        distutils.dir_util.copy_tree(ckpt_dir, saved_exp_dir)

        mode_optimiser = ModeOpt.load(os.path.join(saved_exp_dir, "ckpts"))
        trajectories.update(
            {
                key: plot_mode_opt_mode_controller(
                    mode_optimiser=mode_optimiser, save_dir=save_dir
                )
            },
        )
        calculate_metrics(mode_optimiser=mode_optimiser, name=key)
        # plot_mode_opt_mode_controller(mode_opt_ckpt_dir=ckpt_dir, save_dir=save_dir)
        # calculate_metrics_given_ckpt_dir(ckpt_dir)

    print("after")
    print(trajectories)

    def plot_prob_vs_time(trajectories):
        figsize = (6.4, 3.4)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_gridspec(1, 1).subplots()
        for key in trajectories.keys():
            env_probs = (
                mode_optimiser.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                    trajectories[key]["dynamics"]
                )
            )
            times = np.arange(env_probs.shape[0])
            ax.plot(
                times,
                env_probs[:, mode_optimiser.desired_mode],
                label=labels_dict[key],
                color=colors_dict[key],
                linestyle=linestyles_dict[key],
            )
        ax.set_ylim(bottom=0.5)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel(
            "$\Pr(\\alpha_t="
            + str(mode_optimiser.desired_mode + 1)
            + " \mid h_{"
            + str(mode_optimiser.desired_mode + 1)
            + "}(\mathbf{x}_{0:t}), \mathbf{x}_{0:t},\mathbf{u}_{0:t}, \\alpha_{0:t-1}="
            + str(mode_optimiser.desired_mode + 1)
            + ")$"
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0.5, 0.05),
            loc="upper center",
            bbox_transform=fig.transFigure,
            # ncol=len(by_label),
            ncol=int(len(by_label) / 3),
        )
        fig.tight_layout()

    def plot_state_variance_vs_time(trajectories):
        figsize = (6.4, 3.4)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_gridspec(1, 1).subplots()
        for key in trajectories.keys():
            # state_mean, state_var = trajectories[key]["dynamics"]
            traj = trajectories[key]["dynamics"]
            print("traj")
            print(traj)
            print("state_mean")
            print(state_mean.shape)
            print(state_var.shape)
            times = np.arange(state_var.shape[0])
            print(times.shape)
            ax.plot(
                times,
                state_var,
                label=labels_dict[key],
                # color=colors_dict[key],
            )
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("$\mathbb{V}[\mathbf{x}_{t}]$")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0.5, 0.05),
            loc="upper center",
            bbox_transform=fig.transFigure,
            ncol=int(len(by_label) / 3),
        )
        fig.tight_layout()

    def plot_gating_variance_vs_time(trajectories):
        figsize = (6.4, 3.4)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_gridspec(1, 1).subplots()
        for key in trajectories.keys():
            state_mean = trajectories[key]["dynamics"]
            input_mean, _ = combine_state_controls_to_input(
                state_mean=state_mean,
                control_mean=tf.zeros(state_mean.shape, dtype=default_float()),
            )
            (
                _,
                gating_var,
            ) = mode_optimiser.dynamics.mosvgpe.gating_network.predict_h(input_mean)
            gating_var = gating_var[:, mode_optimiser.desired_mode]
            print("gating_mean")
            # print(gating_mean.shape)
            print(gating_var.shape)
            times = np.arange(gating_var.shape[0])
            print(times.shape)
            ax.plot(
                times,
                gating_var,
                label=labels_dict[key],
                color=colors_dict[key],
                linestyle=linestyles_dict[key],
            )
        ax.set_xlabel("Time $t$")
        ax.set_ylabel(
            "$\mathbb{V}[h_{"
            + str(mode_optimiser.desired_mode + 1)
            + "}(\mathbf{x}_{t})]$"
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(0.5, 0.05),
            loc="upper center",
            bbox_transform=fig.transFigure,
            # ncol=len(by_label),
            ncol=int(len(by_label) / 3),
        )
        fig.tight_layout()

    trajectories_scenario_5 = {}
    trajectories_scenario_7 = {}
    for key in trajectories.keys():
        if "scenario_5" in key:
            trajectories_scenario_5.update({key: trajectories[key]})
        # if "scenario_7" in key:
        #     trajectories_scenario_7.update({key: trajectories[key]})
    print("trajectories_scenario_5")
    print(trajectories_scenario_5)
    # print("trajectories_scenario_7")
    # print(trajectories_scenario_7)

    plot_prob_vs_time(trajectories_scenario_5)

    save_dir = "./velocity_controlled_point_mass/reports/figures/trajectory_optimisation/scenario_5"
    plt.savefig(os.path.join(save_dir, "desired_prob_vs_time.pdf"), transparent=True)
    # plot_state_variance_vs_time(trajectories_scenario_5)
    # plt.savefig(os.path.join(save_dir, "state_variance_vs_time.pdf"), transparent=True)
    plot_gating_variance_vs_time(trajectories_scenario_5)
    plt.savefig(os.path.join(save_dir, "gating_variance_vs_time.pdf"), transparent=True)

    # plot_prob_vs_time(trajectories_scenario_7)
    # plt.savefig(
    #     os.path.join("./scenario_7/desired_mixing_prob_vs_time.pdf"), transparent=True
    # )
