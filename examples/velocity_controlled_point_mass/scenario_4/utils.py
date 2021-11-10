#!/usr/bin/env python3
import os
import shutil
from datetime import datetime
from functools import partial
from typing import Callable, Union

# import gin
import gin.tf
import gpflow as gpf
import numpy as np
import tensorflow as tf
from gpflow import default_float
from modeopt.dynamics import ModeOptDynamics
from modeopt.dynamics.multimodal import init_ModeOptDynamics_from_mogpe_ckpt
from modeopt.mode_opt import ModeOpt
from modeopt.monitor import init_ModeOpt_monitor
from modeopt.policies import DeterministicPolicy, VariationalGaussianPolicy
from modeopt.trajectory_optimisers import (
    ModeVariationalTrajectoryOptimiserTrainingSpec,
    VariationalTrajectoryOptimiserTrainingSpec,
)
from mogpe.training import MixtureOfSVGPExperts_from_toml
from simenvs.core import make


def velocity_controlled_point_mass_dynamics(
    state_mean, control_mean, state_var=None, control_var=None, delta_time=0.05
):
    velocity = control_mean
    delta_state_mean = velocity * delta_time
    if state_var is not None:
        delta_state_var = control_var * delta_time ** 2
        return delta_state_mean, delta_state_var
    else:
        return delta_state_mean


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
def init_mode_opt(
    env_name,
    delta_time,
    dataset,
    desired_mode,
    start_state,
    target_state,
    test_split_size=0.0,
    policy=None,
    mogpe_ckpt_dir=None,
    mogpe_config_file=None,
    mode_opt_ckpt_dir=None,
    horizon=None,
    mode_chance_constraint_lower=None,
    velocity_constraints_lower=None,
    velocity_constraints_upper=None,
    nominal_dynamics: Callable = velocity_controlled_point_mass_dynamics,
):
    # Configure environment
    env = make(env_name)
    env.state_init = start_state

    # Set boundary conditions
    start_state = tf.constant(
        np.array(start_state).reshape(1, -1), dtype=default_float()
    )
    target_state = tf.constant(
        np.array(target_state).reshape(1, -1), dtype=default_float()
    )
    state_dim = env.observation_spec().shape[0]
    control_dim = env.action_spec().shape[0]

    # Init policy
    control_means = (
        np.ones((horizon, control_dim)) * 0.5
        + np.random.random((horizon, control_dim)) * 0.1
    )
    control_means = control_means * 0.0
    if policy is None:
        policy = DeterministicPolicy
    # if isinstance(policy, DeterministicPolicy):
    if policy == "DeterministicPolicy":
        policy = DeterministicPolicy(
            control_means,
            constraints_lower_bound=velocity_constraints_lower,
            constraints_upper_bound=velocity_constraints_upper,
        )
    elif policy == "VariationalGaussianPolicy":
        # elif isinstance(policy, VariationalGaussianPolicy):
        control_vars = (
            np.ones((horizon, control_dim)) * 0.2
            + np.random.random((horizon, control_dim)) * 0.01
        )
        policy = VariationalGaussianPolicy(
            means=control_means,
            vars=control_vars,
            constraints_lower_bound=velocity_constraints_lower,
            constraints_upper_bound=velocity_constraints_upper,
        )

    # Init dynamics
    nominal_dynamics = partial(
        velocity_controlled_point_mass_dynamics, delta_time=delta_time
    )
    if mogpe_config_file is None and mode_opt_ckpt_dir is not None:
        mogpe_config_file = mode_opt_ckpt_dir + "/mogpe_config.toml"
    dynamics = init_ModeOptDynamics_from_mogpe_ckpt(
        mogpe_config_file=mogpe_config_file,
        mogpe_ckpt_dir=mogpe_ckpt_dir,
        dataset=dataset,
        nominal_dynamics=nominal_dynamics,
        desired_mode=desired_mode,
    )

    mosvgpe = MixtureOfSVGPExperts_from_toml(mogpe_config_file, dataset=dataset)
    dynamics = ModeOptDynamics(
        mosvgpe=mosvgpe,
        desired_mode=desired_mode,
        state_dim=state_dim,
        control_dim=control_dim,
        nominal_dynamics=nominal_dynamics,
        # optimiser=optimiser,
    )

    # Init ModeOpt
    mode_optimiser = ModeOpt(
        start_state=start_state,
        target_state=target_state,
        env=env,
        policy=policy,
        dynamics=dynamics,
        dataset=dataset,
        desired_mode=desired_mode,
        mode_chance_constraint_lower=mode_chance_constraint_lower,
        horizon=horizon,
    )

    if mode_opt_ckpt_dir is not None:
        ckpt = tf.train.Checkpoint(model=mode_optimiser)
        manager = tf.train.CheckpointManager(ckpt, mode_opt_ckpt_dir, max_to_keep=5)
        ckpt.restore(manager.latest_checkpoint)
        print("Restored ModeOpt")
        gpf.utilities.print_summary(mode_optimiser)
    return mode_optimiser


@gin.configurable
def config_traj_opt(
    mode_opt_ckpt_dir,
    mode_opt_config,
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
    state_cost_weight: default_float() = 1.0,
    control_cost_weight: default_float() = 1.0,
    terminal_state_cost_weight: default_float() = 1.0,
    riemannian_metric_cost_weight: default_float() = 1.0,
    riemannian_metric_covariance_weight: default_float() = 1.0,
):
    # Load data set from ckpt dir
    print("mode_opt_ckpt_dir")
    print(mode_opt_ckpt_dir)
    train_dataset = np.load(os.path.join(mode_opt_ckpt_dir, "train_dataset.npz"))
    dataset = (train_dataset["x"], train_dataset["y"])

    # Init mode_opt from gin config
    mode_optimiser = init_mode_opt(dataset=dataset, mode_opt_ckpt_dir=mode_opt_ckpt_dir)

    # Create nested log_dir inside learn_dynamics dir
    print("log_dir")
    print(log_dir)
    if log_dir is not None:
        log_dir = os.path.join(mode_opt_ckpt_dir, log_dir)
        log_dir = os.path.join(log_dir, datetime.now().strftime("%m-%d-%H%M%S"))
        os.makedirs(log_dir)

        monitor = init_ModeOpt_monitor(
            mode_optimiser,
            log_dir=log_dir,
            fast_tasks_period=fast_tasks_period,
            slow_tasks_period=slow_tasks_period,
        )

        # Init checkpoint manager for saving model during training
        manager = init_checkpoint_manager(
            model=mode_optimiser,
            log_dir=log_dir,
            num_ckpts=num_ckpts,
            mode_opt_gin_config=mode_opt_config,
        )
    else:
        monitor = None
        manager = None

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

    return mode_optimiser, training_spec


def init_checkpoint_manager(
    model,
    log_dir,
    num_ckpts,
    mode_opt_gin_config=None,
    mogpe_toml_config=None,
    train_dataset=None,
    test_dataset=None,
):
    if mogpe_toml_config is not None:
        try:
            # shutil.copy(mogpe_toml_config, log_dir + "/mogpe_config.toml")
            shutil.copy(mogpe_toml_config, os.path.join(log_dir, "mogpe_config.toml"))
        except:
            print("Failed to copy mogpe_config to log_dir")
    if mode_opt_gin_config is not None:
        try:
            shutil.copy(
                mode_opt_gin_config, os.path.join(log_dir, "mode_opt_config.gin")
            )
        except:
            print("Failed to copy mode_opt_config to log_dir")
    if train_dataset is not None:
        np.savez(log_dir + "/train_dataset", x=train_dataset[0], y=train_dataset[1])
    if test_dataset is not None:
        np.savez(log_dir + "/test_dataset", x=test_dataset[0], y=test_dataset[1])
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=num_ckpts)
    return manager
