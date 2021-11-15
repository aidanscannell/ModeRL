#!/usr/bin/env python3
import gin

from scenario_4.utils import config_traj_opt


if __name__ == "__main__":
    mode_opt_config_file = "./scenario_4/configs/explorative_traj_opt_config.gin"
    gin.parse_config_files_and_bindings([mode_opt_config_file], None)

    mode_optimiser, training_spec = config_traj_opt(
        mode_opt_config_file=mode_opt_config_file
    )

    mode_optimiser.optimise_policy(mode_optimiser.start_state, training_spec)
