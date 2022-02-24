#!/usr/bin/env python3
from velocity_controlled_point_mass.data.utils import (
    generate_random_transitions_dataset_from_env,
)


if __name__ == "__main__":

    num_samples = 4000
    # num_samples = 3800
    # num_samples = 1000
    random_seed = 42
    env_name = "velocity-controlled-point-mass/scenario-5"
    # save_dir = "./velocity_controlled_point_mass/scenario_5/data/full_dataset_"
    save_dir = "./velocity_controlled_point_mass/scenario_5/data/full_dataset_t0p25_"
    omit_data_mask = None

    generate_random_transitions_dataset_from_env(
        env_name=env_name,
        save_dir=save_dir,
        num_samples=num_samples,
        omit_data_mask=omit_data_mask,
        plot=True,
        # plot=False,
        random_seed=random_seed,
    )
