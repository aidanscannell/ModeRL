#!/usr/bin/env python3
from functools import partial
from typing import Callable

# import gin
import gin.tf
import gpflow as gpf
import numpy as np
import tensorflow as tf
from gpflow import default_float
from modeopt.dynamics import ModeOptDynamics
from modeopt.dynamics.multimodal import init_ModeOptDynamics_from_mogpe_ckpt
from modeopt.mode_opt import ModeOpt
from modeopt.policies import DeterministicPolicy, VariationalGaussianPolicy
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


@gin.configurable
def init_mode_opt(
    env_name,
    delta_time,
    mogpe_config_file,
    dataset,
    desired_mode,
    start_state,
    target_state,
    test_split_size=0.0,
    policy=None,
    mogpe_ckpt_dir=None,
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
