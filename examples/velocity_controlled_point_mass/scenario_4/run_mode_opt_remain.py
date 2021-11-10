#!/usr/bin/env python3
import os
from datetime import datetime
from typing import Union

import gin.tf
import numpy as np
import tensorflow as tf
from gpflow import default_float
from modeopt.monitor import init_ModeOpt_monitor
from modeopt.trajectory_optimisers import (
    ModeVariationalTrajectoryOptimiserTrainingSpec,
    VariationalTrajectoryOptimiserTrainingSpec,
)

from scenario_4.utils import (
    init_checkpoint_manager,
    init_mode_opt,
    velocity_controlled_point_mass_dynamics,
)


def weight_to_matrix(value: Union[list, float], dim: int):
    if isinstance(value, list):
        if len(value) == dim:
            value = tf.constant(value, dtype=default_float())
            return tf.linalg.diag(value)
        else:
            raise NotImplementedError
    else:
        return tf.eye(dim, dtype=default_float()) * value


@gin.configurable
def run_mode_opt_remain(
    # dataset,
    max_iterations,
    method,
    disp,
    # horizon,
    mode_chance_constraint_lower,
    velocity_constraints_lower,
    velocity_constraints_upper,
    compile_loss_fn,
    compile_mode_constraint_fn,
    log_dir,
    num_ckpts,
    fast_tasks_period,
    slow_tasks_period,
    trajectory_optimiser,
    mode_opt_ckpt_dir,
    mode_opt_config,
    state_cost_weight: default_float() = 1.0,
    control_cost_weight: default_float() = 1.0,
    terminal_state_cost_weight: default_float() = 1.0,
    riemannian_metric_cost_weight: default_float() = 1.0,
    riemannian_metric_covariance_weight: default_float() = 1.0,
):
    print("mode_opt_ckpt_dir")
    print(mode_opt_ckpt_dir)
    train_dataset = np.load(os.path.join(mode_opt_ckpt_dir, "train_dataset.npz"))
    dataset = (train_dataset["x"], train_dataset["y"])

    mode_optimiser = init_mode_opt(dataset=dataset, mode_opt_ckpt_dir=mode_opt_ckpt_dir)
    # mode_optimiser = init_mode_opt()

    # log_dir = os.path.join(log_dir, datetime.now().strftime("%m-%d-%H%M%S"))
    log_dir = os.path.join(mode_opt_ckpt_dir, log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime("%m-%d-%H%M%S"))
    os.makedirs(log_dir)
    # log_dir = mode_opt_ckpt_dir + "/" + log_dir

    monitor = init_ModeOpt_monitor(
        mode_optimiser,
        log_dir=log_dir,
        fast_tasks_period=fast_tasks_period,
        slow_tasks_period=slow_tasks_period,
    )

    # Init checkpoint manager for saving model during training
    # ckpt = tf.train.Checkpoint(model=mode_optimiser)
    # manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=num_ckpts)
    manager = init_checkpoint_manager(
        model=mode_optimiser,
        log_dir=log_dir,
        num_ckpts=num_ckpts,
        mode_opt_gin_config=mode_opt_config,
    )
    # mode_opt_gin_config=None,
    # mogpe_toml_config=None,

    # Init cost function weight matrices
    state_dim = mode_optimiser.dynamics.state_dim
    control_dim = mode_optimiser.dynamics.control_dim
    Q = weight_to_matrix(state_cost_weight, state_dim)
    R = weight_to_matrix(control_cost_weight, control_dim)
    Q_terminal = weight_to_matrix(terminal_state_cost_weight, state_dim)

    if trajectory_optimiser == "ModeVariationalTrajectoryOptimiser":
        print("ModeVariationalTrajectoryOptimiser")
        training_spec = ModeVariationalTrajectoryOptimiserTrainingSpec(
            max_iterations=max_iterations,
            method=method,
            disp=disp,
            mode_chance_constraint_lower=mode_chance_constraint_lower,
            compile_mode_constraint_fn=compile_mode_constraint_fn,
            compile_loss_fn=compile_loss_fn,
            monitor=monitor,
            manager=manager,
            Q=Q,
            R=R,
            Q_terminal=Q_terminal,
            riemannian_metric_cost_weight=riemannian_metric_cost_weight,
            riemannian_metric_covariance_weight=riemannian_metric_covariance_weight,
        )
    elif trajectory_optimiser == "VariationalTrajectoryOptimiser":
        print("VariationalTrajectoryOptimiser")
        training_spec = VariationalTrajectoryOptimiserTrainingSpec(
            max_iterations=max_iterations,
            method=method,
            disp=disp,
            mode_chance_constraint_lower=mode_chance_constraint_lower,
            compile_mode_constraint_fn=compile_mode_constraint_fn,
            compile_loss_fn=compile_loss_fn,
            monitor=monitor,
            manager=manager,
            Q=Q,
            R=R,
            Q_terminal=Q_terminal,
            riemannian_metric_cost_weight=riemannian_metric_cost_weight,
            riemannian_metric_covariance_weight=riemannian_metric_covariance_weight,
        )
    else:
        raise NotImplementedError(
            "Specify a correct trajectory optimiser, VariationalTrajectoryOptimiser, ModeVariationalTrajectoryOptimiser "
        )

    mode_optimiser.optimise_policy(mode_optimiser.start_state, training_spec)


if __name__ == "__main__":
    mode_opt_config = (
        # "./scenario_4/configs/mode_remaining_traj_opt_config.gin",
        "./scenario_4/configs/mode_remaining_riemannian_energy_traj_opt_config.gin",
        # "./scenario_4/configs/mode_remaining_chance_constraints_traj_opt_config.gin",
        # "./scenario_4/configs/mode_remaining_mode_conditioning_traj_opt_config.gin",
        # "./configs/mode_remaining_traj_opt_config.gin",
    )
    gin.parse_config_files_and_bindings(
        [mode_opt_config[0]],
        # ["./scenario_4/configs/mode_remaining_riemannian_energy_traj_opt_config.gin"],
        None,
    )
    # train_dataset, test_dataset = load_vcpm_dataset()
    # run_mode_opt_remain(dataset=train_dataset)
    #
    run_mode_opt_remain(mode_opt_config=mode_opt_config[0])
