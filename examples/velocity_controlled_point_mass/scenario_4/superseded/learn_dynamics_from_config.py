#!/usr/bin/env python3
from scenario_4.dynamics_learning.utils import train_dynamics


if __name__ == "__main__":
    # config_file = "./scenario_4/configs/initial_mogpe_config.toml"
    config_file = "./scenario_4/configs/subset_mogpe_config.toml"

    train_dynamics(config_file)
