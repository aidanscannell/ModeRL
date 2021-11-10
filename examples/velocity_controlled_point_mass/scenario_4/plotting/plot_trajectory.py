#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow as gpf

import gin
from modeopt.monitor import ModeOptPlotter, create_test_inputs
from mogpe.helpers.quadcopter_plotter import QuadcopterPlotter

from scenario_4.utils import config_traj_opt

if __name__ == "__main__":
    # ckpt_dir = "./logs/quadcopter/subset-10/2_experts/batch_size_64/learning_rate_0.01/further_gating_bound/num_inducing_100/11-05-104542"

    ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-10-110119/mode_remaining_chance_constraints_traj_opt/11-10-124104"
    ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-10-110119/mode_remaining_mode_conditioning_traj_opt/11-10-131316"
    ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-10-110119/mode_remaining_mode_conditioning_traj_opt/11-10-133417"
    ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-10-110119/mode_remaining_mode_conditioning_traj_opt/11-10-173052"

    # save_dir = "./scenario_4/images/mode-remaining/chance-constraints"

    save_dir = "./scenario_4/images/mode-remaining/mode-conditioning"
    # save_dir = "./scenario_4/images/mode-remaining/riemannian-energy"

    def init_mode_opt_from_ckpt(ckpt_dir):
        print("ckpt_dir")
        print(ckpt_dir)
        mode_opt_config = os.path.join(ckpt_dir, "mode_opt_config.gin")
        print(mode_opt_config)
        gin.parse_config_files_and_bindings([mode_opt_config], None)
        mode_optimiser, training_spec = config_traj_opt(
            mode_opt_config=mode_opt_config, log_dir=None
        )
        ckpt = tf.train.Checkpoint(model=mode_optimiser)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
        ckpt.restore(manager.latest_checkpoint)
        print("Restored ModeOpt")
        gpf.utilities.print_summary(mode_optimiser)
        return mode_optimiser

    mode_opt = init_mode_opt_from_ckpt(ckpt_dir)

    # mode_opt = init_mode_opt()

    # Create plotter
    test_inputs = create_test_inputs(*mode_opt.dataset)
    mogpe_plotter = QuadcopterPlotter(
        model=mode_opt.dynamics.mosvgpe,
        X=mode_opt.dataset[0],
        Y=mode_opt.dataset[1],
        test_inputs=test_inputs,
    )
    mode_opt_plotter = ModeOptPlotter(mode_opt, mogpe_plotter)
    mode_opt_plotter.plot_model(save_dir=save_dir)
    # plt.show()
