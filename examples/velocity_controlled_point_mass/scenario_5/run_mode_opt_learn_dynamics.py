#!/usr/bin/env python3
import gin
import gpflow as gpf

from velocity_controlled_point_mass.utils import config_learn_dynamics

if __name__ == "__main__":
    mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/learn_dynamics_subset_config.gin"
    gin.parse_config_files_and_bindings([mode_opt_config_file], None)

    mode_optimiser, training_spec, train_dataset = config_learn_dynamics(
        mode_opt_config_file=mode_opt_config_file
    )
    gpf.utilities.print_summary(mode_optimiser)

    mode_optimiser.optimise_dynamics(dataset=train_dataset, training_spec=training_spec)
