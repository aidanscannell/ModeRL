#!/usr/bin/env python3
import gin.tf

from scenario_4.utils import config_traj_opt

if __name__ == "__main__":
    mode_opt_config_file = (
        # "./scenario_4/configs/mode_remaining_traj_opt_config.gin",
        # "./scenario_4/configs/mode_remaining_riemannian_energy_traj_opt_config.gin",
        # "./scenario_4/configs/mode_remaining_conditioning_energy_traj_opt_config.gin",
        # "./scenario_4/configs/mode_remaining_chance_constraints_traj_opt_config.gin",
        "./scenario_4/configs/mode_remaining_mode_conditioning_traj_opt_config.gin",
        # "./configs/mode_remaining_traj_opt_config.gin",
    )
    gin.parse_config_files_and_bindings([mode_opt_config_file[0]], None)
    mode_optimiser, training_spec = config_traj_opt(
        mode_opt_config_file=mode_opt_config_file[0]
    )

    mode_optimiser.optimise_policy(mode_optimiser.start_state, training_spec)
