#!/usr/bin/env python3
import os
import numpy as np

import matplotlib.pyplot as plt
from modeopt.mode_opt import ModeOpt
from modeopt.plotting import ModeOptContourPlotter
from omegaconf import OmegaConf
from velocity_controlled_point_mass.data.utils import load_vcpm_dataset

# from velocity_controlled_point_mass.mode_opt_riemannian_energy_traj_opt import (
#     create_test_inputs,
# )


def get_experiment_cfg(path_to_experiment):
    config_path = os.path.join(path_to_experiment, ".hydra/config.yaml")
    cfg = OmegaConf.load(config_path)
    return cfg


def get_dynamics_cfg(path_to_experiment):
    cfg = get_experiment_cfg(path_to_experiment)
    dynamics_ckpt_dir = cfg.dynamics.ckpt_dir.replace("../", "")
    dynamics_cfg = get_experiment_cfg(
        os.path.join(path_to_experiments, dynamics_ckpt_dir.split("/ckpts")[0])
    )
    return dynamics_cfg


def get_dataset_and_trim_coords(path_to_experiment):
    dynamics_cfg = get_dynamics_cfg(path_to_experiment)
    dataset_filename = os.path.join(
        "./velocity_controlled_point_mass",
        dynamics_cfg.dataset.filename.replace("../", ""),
    )
    dataset, _ = load_vcpm_dataset(
        filename=dataset_filename,
        trim_coords=dynamics_cfg.dataset.trim_coords,
    )
    dynamics_cfg = OmegaConf.to_object(dynamics_cfg)
    trim_coords = list(dynamics_cfg["dataset"]["trim_coords"])
    return dataset, trim_coords


def create_test_inputs(x_min=[-3, -3], x_max=[3, 3], input_dim=4, num_test: int = 1600):
    sqrtN = int(np.sqrt(num_test))
    # xx = np.linspace(x_min[0] * factor, x_max[0] * factor, sqrtN)
    xx = np.linspace(x_min[0], x_max[0], sqrtN)
    yy = np.linspace(x_min[1], x_max[1], sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    if input_dim > 2:
        zeros = np.zeros((num_test, input_dim - 2))
        test_inputs = np.concatenate([test_inputs, zeros], -1)
    return test_inputs


if __name__ == "__main__":
    path_to_saved_experiments = "./velocity_controlled_point_mass/saved_experiments"
    path_to_experiments = "./velocity_controlled_point_mass/experiments"
    # path_to_experiments = "../experiments"

    # ckpt_dir = "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/baseline_traj_opt/12-15-150625"
    experiment_dict = {
        # "scenario_7": "scenario_7/trajectory_optimisation/riemannian_energy/2022.03.07/151334"
        "scenario_7": "scenario_7/trajectory_optimisation/riemannian_energy/2022.03.08/151633",
        # "scenario_5": "scenario_5/trajectory_optimisation/riemannian_energy/2022.03.09/121822",
        "scenario_5": "scenario_5/trajectory_optimisation/riemannian_energy/2022.03.16/130358",
    }
    x_min_dict = {"scenario_5": [-3.6, -3.6], "scenario_7": [-3.6, -3.6]}
    x_max_dict = {"scenario_5": [3.6, 3], "scenario_7": [3, 3.6]}

    for key in experiment_dict.keys():
        path_to_experiment = os.path.join(path_to_experiments, experiment_dict[key])
        dataset, trim_coords = get_dataset_and_trim_coords(path_to_experiment)

        save_dir = os.path.join(
            "./velocity_controlled_point_mass/reports/figures/env", key
        )
        os.makedirs(save_dir, exist_ok=True)

        ckpt_dir = os.path.join(path_to_experiments, experiment_dict[key])
        mode_optimiser = ModeOpt.load(os.path.join(ckpt_dir, "ckpts"))
        mode_optimiser.dataset = dataset

        # Create plotter
        test_inputs = create_test_inputs(
            x_min=x_min_dict[key], x_max=x_max_dict[key], input_dim=4, num_test=8100
        )
        mode_optimiser_plotter = ModeOptContourPlotter(
            mode_optimiser=mode_optimiser, test_inputs=test_inputs
        )

        mode_optimiser_plotter.plot_env()
        plt.savefig(
            os.path.join(save_dir, "env_with_dataset_start_end_pos.pdf"),
            transparent=True,
        )

        fig = mode_optimiser_plotter.mosvgpe_plotter.plot_single_gating_network_gp(
            desired_mode=mode_optimiser.desired_mode
        )
        plt.savefig(
            os.path.join(save_dir, "mosvgpe/desired_gating_gp.pdf"), transparent=True
        )
        mode_optimiser_plotter.plot_env_no_obs_start_end_given_fig(fig, trim_coords)
        plt.savefig(
            os.path.join(save_dir, "mosvgpe/desired_gating_gp_no_obs.pdf"),
            transparent=True,
        )

        fig = mode_optimiser_plotter.mosvgpe_plotter.plot_mixing_probs()
        plt.savefig(
            os.path.join(save_dir, "mosvgpe/mixing_probs.pdf"), transparent=True
        )
        mode_optimiser_plotter.plot_env_no_obs_start_end_given_fig(fig, trim_coords)
        plt.savefig(
            os.path.join(save_dir, "mosvgpe/mixing_probs_no_obs.pdf"), transparent=True
        )
