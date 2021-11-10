#!/usr/bin/env python3
import os
from datetime import datetime

import gin.tf
import numpy as np
from gpflow import default_float
from modeopt.monitor import init_ModeOpt_monitor
from modeopt.trajectory_optimisers import (
    ModeVariationalTrajectoryOptimiserTrainingSpec,
    VariationalTrajectoryOptimiserTrainingSpec,
)

from scenario_4.utils import (
    init_checkpoint_manager,
    init_mode_opt,
    config_traj_opt,
    velocity_controlled_point_mass_dynamics,
)


if __name__ == "__main__":
    mode_opt_config = (
        # "./scenario_4/configs/mode_remaining_traj_opt_config.gin",
        # "./scenario_4/configs/mode_remaining_riemannian_energy_traj_opt_config.gin",
        # "./scenario_4/configs/mode_remaining_chance_constraints_traj_opt_config.gin",
        "./scenario_4/configs/mode_remaining_mode_conditioning_traj_opt_config.gin",
        # "./configs/mode_remaining_traj_opt_config.gin",
    )
    gin.parse_config_files_and_bindings([mode_opt_config[0]], None)
    mode_optimiser, training_spec = config_traj_opt(mode_opt_config=mode_opt_config[0])

    mode_optimiser.optimise_policy(mode_optimiser.start_state, training_spec)
