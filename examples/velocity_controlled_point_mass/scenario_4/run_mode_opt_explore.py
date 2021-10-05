#!/usr/bin/env python3
import os
from datetime import datetime
from functools import partial

import gin
import numpy as np
import tensorflow as tf
from gpflow import default_float
from modeopt.dynamics.utils import init_ModeOptDynamics_from_mogpe_ckpt
from modeopt.mode_opt import ModeOpt
from modeopt.monitor import init_ModeOpt_monitor
from modeopt.policies import DeterministicPolicy
from modeopt.trajectory_optimisers import (
    ExplorativeTrajectoryOptimiserTrainingSpec,
)
from simenvs.core import make

# from .dynamics import velocity_controlled_point_mass_dynamics


def velocity_controlled_point_mass_dynamics(
    state_mean, control_mean, state_var=None, control_var=None, delta_time=0.05
):
    velocity = control_mean
    delta_state_mean = velocity * delta_time
    delta_state_var = control_var * delta_time ** 2
    return delta_state_mean, delta_state_var


@gin.configurable
def run_mode_opt(
    mogpe_config_file,
    mogpe_ckpt_dir,
    data_path,
    desired_mode,
    start_state,
    target_state,
    env_name,
    delta_time,
    max_iterations,
    method,
    disp,
    horizon,
    mode_chance_constraint_lower,
    velocity_constraints_lower,
    velocity_constraints_upper,
    compile_loss_fn,
    compile_mode_constraint_fn,
    # nominal_dynamics: Callable,
    state_cost_weight,
    control_cost_weight,
    terminal_state_cost_weight,
    log_dir,
    num_ckpts,
    fast_tasks_period,
    slow_tasks_period,
):
    start_state = tf.constant(
        np.array(start_state).reshape(1, -1), dtype=default_float()
    )
    target_state = tf.constant(
        np.array(target_state).reshape(1, -1), dtype=default_float()
    )

    # Configure environment
    env = make(env_name)
    env.state_init = start_state

    state_dim = env.observation_spec().shape[0]
    control_dim = env.action_spec().shape[0]

    control_means = (
        np.ones((horizon, control_dim)) * 0.5
        + np.random.random((horizon, control_dim)) * 0.1
    )

    control_means = control_means * 0.0
    # control_vars = (
    #     np.ones((horizon, control_dim)) * 0.2
    #     + np.random.random((horizon, control_dim)) * 0.01
    # )

    # policy = VariationalGaussianPolicy(
    #     means=control_means,
    #     vars=control_vars,
    #     constraints_lower_bound=velocity_constraints_lower,
    #     constraints_upper_bound=velocity_constraints_upper,
    # )
    policy = DeterministicPolicy(
        control_means,
        # vars=control_vars,
        constraints_lower_bound=velocity_constraints_lower,
        constraints_upper_bound=velocity_constraints_upper,
    )

    nominal_dynamics = partial(
        velocity_controlled_point_mass_dynamics, delta_time=delta_time
    )
    dynamics, dataset = init_ModeOptDynamics_from_mogpe_ckpt(
        mogpe_config_file,
        mogpe_ckpt_dir,
        data_path,
        nominal_dynamics=nominal_dynamics,
        desired_mode=desired_mode,
        return_dataset=True,
    )

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
        state_cost_weight=state_cost_weight,
        control_cost_weight=control_cost_weight,
        terminal_state_cost_weight=terminal_state_cost_weight,
    )

    log_dir = os.path.join(log_dir, datetime.now().strftime("%m-%d-%H%M%S"))
    monitor = init_ModeOpt_monitor(
        mode_optimiser,
        log_dir=log_dir,
        fast_tasks_period=fast_tasks_period,
        slow_tasks_period=slow_tasks_period,
    )

    # Init checkpoint manager for saving model during training
    ckpt = tf.train.Checkpoint(model=mode_optimiser)
    manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=num_ckpts)

    training_spec = ExplorativeTrajectoryOptimiserTrainingSpec(
        max_iterations=max_iterations,
        method=method,
        disp=disp,
        mode_chance_constraint_lower=mode_chance_constraint_lower,
        compile_mode_constraint_fn=compile_mode_constraint_fn,
        compile_loss_fn=compile_loss_fn,
        monitor=monitor,
        manager=manager,
    )

    mode_optimiser.optimise_policy(start_state, training_spec)


if __name__ == "__main__":
    gin.parse_config_files_and_bindings(
        [
            "./configs/explorative_traj_opt_config.gin",
        ],
        None,
    )
    run_mode_opt()
