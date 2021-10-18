#!/usr/bin/env python3
from .utils import generate_random_transitions_dataset_from_env


if __name__ == "__main__":
    num_samples = 4000
    random_seed = 42
    env_name = "velocity-controlled-point-mass/scenario-4"
    save_dir = "./data/npz/initial_dataset_"
    omit_data_mask = "./data/initial_omit_data_mask.bmp"

    generate_random_transitions_dataset_from_env(
        env_name=env_name,
        save_dir=save_dir,
        num_samples=num_samples,
        omit_data_mask=omit_data_mask,
        plot=True,
        # plot=False,
        random_seed=random_seed,
    )
