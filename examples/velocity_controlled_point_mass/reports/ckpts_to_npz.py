#!/usr/bin/env python3
import shutil
import os

import numpy as np

# from modeopt.metrics import (
#     approximate_riemannian_energy,
#     gating_function_variance,
#     mode_probability,
#     state_variance,
# )
from modeopt.mode_opt import ModeOpt
from modeopt.trajectories import GeodesicTrajectory

# from velocity_controlled_point_mass.mode_opt_riemannian_energy_traj_opt import (
#     create_test_inputs,
# )


def generate_trajectories(mode_optimiser):
    dynamics_trajectory = mode_optimiser.dynamics_rollout()[0]
    trajectories = {"dynamics": dynamics_trajectory}
    if mode_optimiser.env is not None:
        trajectories.update({"env": mode_optimiser.env_rollout()})
    if isinstance(
        mode_optimiser.mode_controller.previous_solution,
        GeodesicTrajectory,
    ):
        trajectories.update(
            {"collocation": mode_optimiser.mode_controller.previous_solution.states}
        )
    return trajectories


if __name__ == "__main__":
    path_to_saved_experiments = "./velocity_controlled_point_mass/saved_experiments"
    path_to_experiments = "./velocity_controlled_point_mass/experiments"

    # experiment_dict = {
    #     # "scenario_5/riemannian-energy": "scenario_5/trajectory_optimisation/riemannian_energy/2022.03.09/121822",
    #     # "scenario_5/riemannian-energy-low": "scenario_5/trajectory_optimisation/riemannian_energy_low/2022.03.09/121550",
    #     # "scenario_5/riemannian-energy-high": "scenario_5/trajectory_optimisation/riemannian_energy_high/2022.03.09/122350",
    #     # "scenario_5/riemannian-energy": "scenario_5/trajectory_optimisation/riemannian_energy/2022.03.15/123140",
    #     # "scenario_5/riemannian-energy-low": "scenario_5/trajectory_optimisation/riemannian_energy_low/2022.03.15/122734",
    #     # "scenario_5/riemannian-energy-low-2": "scenario_5/trajectory_optimisation/riemannian_energy_low_2/2022.03.15/123607",
    #     # "scenario_5/riemannian-energy-high": "scenario_5/trajectory_optimisation/riemannian_energy_high/2022.03.15/122943",
    #     "scenario_5/riemannian-energy": "scenario_5/trajectory_optimisation/riemannian_energy/2022.03.16/130358",
    #     "scenario_5/riemannian-energy-low": "scenario_5/trajectory_optimisation/riemannian_energy_low/2022.03.16/131428",
    #     "scenario_5/riemannian-energy-low-2": "scenario_5/trajectory_optimisation/riemannian_energy_low_2/2022.03.16/131442",
    #     "scenario_5/riemannian-energy-high": "scenario_5/trajectory_optimisation/riemannian_energy_high/2022.03.16/130011",
    #     "scenario_5/geodesic-collocation": "scenario_5/trajectory_optimisation/geodesic_collocation/2022.03.18/123518",
    #     "scenario_5/geodesic-collocation-mid-point": "scenario_5/trajectory_optimisation/geodesic_collocation/2022.03.18/125548",
    #     "scenario_5/geodesic-collocation-low": "scenario_5/trajectory_optimisation/geodesic_collocation_low/2022.03.18/154418",
    #     "scenario_5/geodesic-collocation-high": "scenario_5/trajectory_optimisation/geodesic_collocation_high/2022.03.18/121855",
    #     "scenario_7/riemannian-energy": "scenario_7/trajectory_optimisation/riemannian_energy/2022.03.08/151633",
    #     # "scenario_7/riemannian-energy-low": "scenario_7/trajectory_optimisation/riemannian_energy_low/2022.03.08/153724",
    #     "scenario_7/riemannian-energy-low": "scenario_7/trajectory_optimisation/riemannian_energy_low/2022.03.08/195703",
    #     # "scenario_7/riemannian-energy-high": "scenario_7/trajectory_optimisation/riemannian_energy_high/2022.03.08/151851",
    #     "scenario_7/riemannian-energy-high": "scenario_7/trajectory_optimisation/riemannian_energy_high/2022.03.09/114346",
    #     "scenario_7/geodesic-collocation": "scenario_7/trajectory_optimisation/geodesic_collocation/2022.03.08/174538",
    #     "scenario_7/geodesic-collocation-low": "scenario_7/trajectory_optimisation/geodesic_collocation_low/2022.03.08/171012",
    #     "scenario_7/geodesic-collocation-high": "scenario_7/trajectory_optimisation/geodesic_collocation_high/2022.03.09/113857",
    #     # "scenario_5/control-as-inference": "scenario_5/trajectory_optimisation/control_as_inference/2022.03.14/154427",
    #     # "scenario_5/control-as-inference-deterministic": "scenario_5/trajectory_optimisation/control_as_inference_deterministic/2022.03.14/154836",
    #     "scenario_5/control-as-inference": "scenario_5/trajectory_optimisation/control_as_inference/2022.03.16/133955",
    #     "scenario_5/control-as-inference-deterministic": "scenario_5/trajectory_optimisation/control_as_inference_deterministic/2022.03.16/133940",
    #     "scenario_7/control-as-inference": "scenario_7/trajectory_optimisation/control_as_inference/2022.03.14/161004",
    #     "scenario_7/control-as-inference-deterministic": "scenario_7/trajectory_optimisation/control_as_inference_deterministic/2022.03.14/160121",
    # }
    experiment_dict = {
        "scenario_5/riemannian-energy": "scenario_5/trajectory_optimisation/riemannian_energy/2022.03.16/130358",
        # "scenario_5/riemannian-energy-low": "scenario_5/trajectory_optimisation/riemannian_energy_low/2022.03.16/131428",
        "scenario_5/riemannian-energy-low-2": "scenario_5/trajectory_optimisation/riemannian_energy_low_2/2022.03.16/131442",
        "scenario_5/riemannian-energy-high": "scenario_5/trajectory_optimisation/riemannian_energy_high/2022.03.16/130011",
        "scenario_5/geodesic-collocation": "scenario_5/trajectory_optimisation/geodesic_collocation/2022.03.18/123518",
        "scenario_5/geodesic-collocation-mid-point": "scenario_5/trajectory_optimisation/geodesic_collocation/2022.03.18/125548",
        "scenario_5/geodesic-collocation-low": "scenario_5/trajectory_optimisation/geodesic_collocation_low/2022.03.18/154418",
        "scenario_5/geodesic-collocation-high": "scenario_5/trajectory_optimisation/geodesic_collocation_high/2022.03.18/121855",
        "scenario_7/riemannian-energy": "scenario_7/trajectory_optimisation/riemannian_energy/2022.04.22/163121/2022-04-22-16-31",
        "scenario_7/riemannian-energy-low": "scenario_7/trajectory_optimisation/riemannian_energy_low/2022.04.22/162526/2022-04-22-16-25",
        "scenario_7/riemannian-energy-high": "scenario_7/trajectory_optimisation/riemannian_energy_high/2022.04.22/162818/2022-04-22-16-28",
        "scenario_7/geodesic-collocation": "scenario_7/trajectory_optimisation/geodesic_collocation/2022.03.08/174538",
        "scenario_7/geodesic-collocation-low": "scenario_7/trajectory_optimisation/geodesic_collocation_low/2022.03.08/171012",
        "scenario_7/geodesic-collocation-high": "scenario_7/trajectory_optimisation/geodesic_collocation_high/2022.04.22/172005/2022-04-22-17-20",
        "scenario_5/control-as-inference": "scenario_5/trajectory_optimisation/control_as_inference/2022.03.16/133955",
        "scenario_5/control-as-inference-deterministic": "scenario_5/trajectory_optimisation/control_as_inference_deterministic/2022.03.16/133940",
        "scenario_7/control-as-inference": "scenario_7/trajectory_optimisation/control_as_inference/2022.03.14/161004",
        "scenario_7/control-as-inference-deterministic": "scenario_7/trajectory_optimisation/control_as_inference_deterministic/2022.03.14/160121",
    }

    trajectories = {}
    save_dir = "./velocity_controlled_point_mass/reports/saved_trajectories"
    os.makedirs(save_dir, exist_ok=True)
    for key in experiment_dict.keys():
        ckpt_dir = os.path.join(path_to_experiments, experiment_dict[key])
        saved_exp_dir = os.path.join(path_to_saved_experiments, experiment_dict[key])
        shutil.copytree(ckpt_dir, saved_exp_dir, dirs_exist_ok=True)

        mode_optimiser = ModeOpt.load(os.path.join(saved_exp_dir, "ckpts"))
        trajectories = generate_trajectories(mode_optimiser)
        # trajectory_dict_save_filename = os.path.join(
        #     save_dir, f"{experiment_dict[key]}.npz"
        # )
        np.savez(save_dir + f"/{key}.npz", **trajectories)

    print("after")
    print(trajectories)
