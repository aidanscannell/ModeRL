#!/usr/bin/env python3
import gin.tf
from velocity_controlled_point_mass.utils import config_traj_opt

if __name__ == "__main__":
    # mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/mode_remaining_conditioning_energy_traj_opt_config.gin"
    mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/mode_remaining_conditioning_energy_max_entropy_traj_opt_config.gin"
    # mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/mode_remaining_mode_conditioning_traj_opt_config.gin"
    # mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/mode_remaining_riemannian_energy_traj_opt_config.gin"
    # mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/mode_remaining_riemannian_energy_low_cov_traj_opt_config.gin"
    # mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/mode_remaining_riemannian_energy_high_cov_traj_opt_config.gin"
    mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/mode_remaining_riemannian_energy_fail_traj_opt_config.gin"
    # mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/mode_remaining_chance_constraints_traj_opt_config.gin"
    # mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/mode_remaining_prob_traj_opt_config.gin"
    # mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/mode_remaining_prob_var_traj_opt_config.gin"
    # mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/baseline_traj_opt_config.gin"

    # mode_opt_config_file = (
    #     "./velocity_controlled_point_mass/scenario_5/configs/svgp_traj_opt_config.gin"
    # )
    gin.parse_config_files_and_bindings([mode_opt_config_file], None)
    mode_optimiser, training_spec = config_traj_opt(
        mode_opt_config_file=mode_opt_config_file
    )

    mode_optimiser.optimise_policy(mode_optimiser.start_state, training_spec)
