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
from gpflow.monitor import Monitor
from gpflow.utilities import print_summary
from modeopt.cost_functions import (
    ControlQuadraticCostFunction,
    ModeProbCostFunction,
    RiemannianEnergyCostFunction,
    StateQuadraticCostFunction,
    TargetStateCostFunction,
    ZeroCostFunction,
    StateVarianceCostFunction,
    # state_control_quadratic_cost_fn,
    # terminal_state_cost_fn,
)
from modeopt.dynamics import ModeOptDynamics, ModeOptDynamicsTrainingSpec
from modeopt.mode_opt import ModeOpt
from modeopt.monitor import create_test_inputs, init_ModeOpt_monitor
from modeopt.policies import DeterministicPolicy, VariationalGaussianPolicy
from modeopt.rollouts import rollout_policy_in_dynamics
from modeopt.trajectory_optimisers import (
    ExplorativeTrajectoryOptimiserTrainingSpec,
    ModeVariationalTrajectoryOptimiserTrainingSpec,
    VariationalTrajectoryOptimiserTrainingSpec,
)
from mogpe.helpers.plotter import Plotter2D
from mogpe.training import MixtureOfSVGPExperts_from_toml
from mogpe.training.utils import (
    create_log_dir,
    create_tf_dataset,
    init_fast_tasks_bounds,
)
from simenvs.core import make

# from scenario_4.data.load_data import load_vcpm_dataset
from velocity_controlled_point_mass.data.utils import load_vcpm_dataset


def velocity_controlled_point_mass_dynamics(
    state_mean,
    control_mean,
    state_var=None,
    control_var=None,
    delta_time=0.05,
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
    elif value is None or value == 0.0:
        return None
    else:
        return tf.eye(dim, dtype=default_float()) * value


def init_policy(
    policy_name: str,
    horizon: int,
    control_dim: int,
    velocity_constraints_lower,
    velocity_constraints_upper,
):
    control_means = (
        # np.ones((horizon, control_dim))
        # np.ones((horizon, control_dim)) * [-5.0, 0.5]
        np.ones((horizon, control_dim)) * [-1.0, 1.0]
        + np.random.random((horizon, control_dim)) * 0.1
    )
    control_means = control_means * 0.0
    # control_means = control_means * 100.0
    if policy_name is None:
        policy = DeterministicPolicy
    # if isinstance(policy, DeterministicPolicy):
    if policy_name == "DeterministicPolicy":
        policy = DeterministicPolicy(
            control_means,
            constraints_lower_bound=velocity_constraints_lower,
            constraints_upper_bound=velocity_constraints_upper,
        )
    elif policy_name == "VariationalGaussianPolicy":
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
    else:
        raise NotImplementedError(
            "policy should be DeterministicPolicy or VariationalGaussianPolicy"
        )
    return policy


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
    velocity_constraints_lower=None,
    velocity_constraints_upper=None,
    # nominal_dynamics: Callable = velocity_controlled_point_mass_dynamics,
    nominal_dynamics: bool = True,
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

    # # Init policy
    # control_means = (
    #     np.ones((horizon, control_dim))
    #     # np.ones((horizon, control_dim)) * [-5.0, 0.5]
    #     + np.random.random((horizon, control_dim)) * 0.1
    # )
    # # control_means = control_means * 0.0
    # # control_means = control_means * 100000000.0
    # if policy is None:
    #     policy = DeterministicPolicy
    # # if isinstance(policy, DeterministicPolicy):
    # if policy == "DeterministicPolicy":
    #     policy = DeterministicPolicy(
    #         control_means,
    #         constraints_lower_bound=velocity_constraints_lower,
    #         constraints_upper_bound=velocity_constraints_upper,
    #     )
    # elif policy == "VariationalGaussianPolicy":
    #     # elif isinstance(policy, VariationalGaussianPolicy):
    #     control_vars = (
    #         np.ones((horizon, control_dim)) * 0.2
    #         + np.random.random((horizon, control_dim)) * 0.01
    #     )
    #     policy = VariationalGaussianPolicy(
    #         means=control_means,
    #         vars=control_vars,
    #         constraints_lower_bound=velocity_constraints_lower,
    #         constraints_upper_bound=velocity_constraints_upper,
    #     )
    # else:
    #     raise NotImplementedError(
    #         "policy should be DeterministicPolicy or VariationalGaussianPolicy"
    #     )
    # Init policy
    policy = init_policy(
        policy,
        horizon,
        control_dim,
        velocity_constraints_lower,
        velocity_constraints_upper,
    )

    # Init dynamics
    if nominal_dynamics:
        print("adding nominal dynamics")
        nominal_dynamics = partial(
            velocity_controlled_point_mass_dynamics, delta_time=delta_time
        )
    else:
        print("no nominal dynamics")
        nominal_dynamics = None
    if mogpe_config_file is None and mode_opt_ckpt_dir is not None:
        mogpe_config_file = mode_opt_ckpt_dir + "/mogpe_config.toml"
    mosvgpe = MixtureOfSVGPExperts_from_toml(mogpe_config_file, dataset=dataset)
    # gpf.set_trainable(mosvgpe.experts.experts_list[0], False)
    # gpf.set_trainable(mosvgpe.gating_network, False)

    dynamics = ModeOptDynamics(
        mosvgpe=mosvgpe,
        desired_mode=desired_mode,
        state_dim=state_dim,
        control_dim=control_dim,
        nominal_dynamics=nominal_dynamics,
        # optimiser=optimiser,
    )

    # import matplotlib.pyplot as plt
    # states, state_vars = rollout_policy_in_dynamics(policy, dynamics, start_state)
    # plt.plot(states[:, 0], states[:, 1], color="b")
    # # Init policy
    # control_means = np.ones((horizon, control_dim)) * [-5.0, 0.5]
    # policy = DeterministicPolicy(
    #     control_means,
    #     constraints_lower_bound=velocity_constraints_lower,
    #     constraints_upper_bound=velocity_constraints_upper,
    # )
    # states, state_vars = rollout_policy_in_dynamics(policy_2, dynamics, start_state)
    # plt.plot(states[:, 0], states[:, 1], color="r")
    # plt.show()

    # Init ModeOpt
    mode_optimiser = ModeOpt(
        start_state=start_state,
        target_state=target_state,
        env=env,
        policy=policy,
        dynamics=dynamics,
        dataset=dataset,
        desired_mode=desired_mode,
        # horizon=horizon,
    )

    if mode_opt_ckpt_dir is not None:
        ckpt = tf.train.Checkpoint(model=mode_optimiser)
        manager = tf.train.CheckpointManager(ckpt, mode_opt_ckpt_dir, max_to_keep=5)
        ckpt.restore(manager.latest_checkpoint)
        print("Restored ModeOpt")
        gpf.utilities.print_summary(mode_optimiser)

    # dynamics_q_mu = mode_optimiser.dynamics.dynamics_gp.q_mu
    # dynamics_q_mu = dynamics_q_mu + 0.5 * tf.ones(
    #     dynamics_q_mu.shape, dtype=default_float()
    # )
    return mode_optimiser


@gin.configurable
def config_learn_svgp(
    mode_opt_config_file,
    mogpe_config_file,
    log_dir,
    num_epochs,
    batch_size,
    learning_rate,
    # optimiser,
    logging_epoch_freq,
    fast_tasks_period,
    slow_tasks_period,
    num_ckpts,
    compile_loss_fn: bool = True,
):
    train_dataset, test_dataset = load_vcpm_dataset()

    # mode_optimiser = init_mode_opt(
    #     dataset=train_dataset, mogpe_config_file=mogpe_config_file
    # )
    # if mogpe_config_file is None and mode_opt_ckpt_dir is not None:
    # mogpe_config_file = mode_opt_ckpt_dir + "/mogpe_config.toml"
    # mosvgpe = MixtureOfSVGPExperts_from_toml(mogpe_config_file, dataset=dataset)
    # print_summary(mo)
    model = init_SVGP()

    # Create monitor tasks (plots/elbo/model params etc)
    log_dir = create_log_dir(
        log_dir,
        num_experts=1,
        batch_size=batch_size,
        # learning_rate=optimiser.learning_rate,
        learning_rate=learning_rate,
        bound="ELBO",
        num_inducing=model.inducing_variable.inducing_variables[0].Z.shape[0],
    )

    test_inputs = create_test_inputs(*train_dataset)
    # mogpe_plotter = QuadcopterPlotter(
    # mogpe_plotter = Plotter2D(
    #     model=
    #     X=mode_optimiser.dataset[0],
    #     Y=mode_optimiser.dataset[1],
    #     test_inputs=test_inputs,
    #     # static=False,
    # )

    train_dataset_tf, num_batches_per_epoch = create_tf_dataset(
        train_dataset, num_data=train_dataset[0].shape[0], batch_size=batch_size
    )
    test_dataset_tf, _ = create_tf_dataset(
        test_dataset, num_data=test_dataset[0].shape[0], batch_size=batch_size
    )

    training_loss = model.build_training_loss(train_dataset_tf, compile=compile_loss_fn)

    fast_tasks = init_fast_tasks_bounds(
        log_dir,
        train_dataset_tf,
        model,
        test_dataset=test_dataset_tf,
        # training_loss=training_loss,
        fast_tasks_period=fast_tasks_period,
    )
    # slow_tasks = mogpe_plotter.tf_monitor_task_group(
    #     log_dir,
    #     slow_period=slow_tasks_period
    #     # slow_tasks_period=slow_tasks_period,
    # )
    # monitor = Monitor(fast_tasks, slow_tasks)
    monitor = Monitor(fast_tasks)

    # Init checkpoint manager for saving model during training
    manager = init_checkpoint_manager(
        model=model,
        log_dir=log_dir,
        num_ckpts=num_ckpts,
        mode_opt_gin_config=mode_opt_config_file,
        mogpe_toml_config=mogpe_config_file,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    training_spec = ModeOptDynamicsTrainingSpec(
        num_epochs=num_epochs,
        batch_size=batch_size,
        # optimiser=optimiser,
        learning_rate=learning_rate,
        logging_epoch_freq=logging_epoch_freq,
        compile_loss_fn=compile_loss_fn,
        monitor=monitor,
        manager=manager,
    )
    return mode_optimiser, training_spec, train_dataset


@gin.configurable
def config_learn_dynamics(
    mode_opt_config_file,
    mogpe_config_file,
    log_dir,
    num_epochs,
    batch_size,
    learning_rate,
    # optimiser,
    logging_epoch_freq,
    fast_tasks_period,
    slow_tasks_period,
    num_ckpts,
    compile_loss_fn: bool = True,
):
    train_dataset, test_dataset = load_vcpm_dataset()

    mode_optimiser = init_mode_opt(
        dataset=train_dataset, mogpe_config_file=mogpe_config_file
    )
    print_summary(mode_optimiser)

    # Create monitor tasks (plots/elbo/model params etc)
    log_dir = create_log_dir(
        log_dir,
        mode_optimiser.dynamics.mosvgpe.num_experts,
        batch_size,
        # learning_rate=optimiser.learning_rate,
        learning_rate=learning_rate,
        bound=mode_optimiser.dynamics.mosvgpe.bound,
        num_inducing=mode_optimiser.dynamics.mosvgpe.experts.experts_list[0]
        .inducing_variable.inducing_variables[0]
        .Z.shape[0],
    )

    test_inputs = create_test_inputs(*mode_optimiser.dataset)
    # mogpe_plotter = QuadcopterPlotter(
    mogpe_plotter = Plotter2D(
        model=mode_optimiser.dynamics.mosvgpe,
        X=mode_optimiser.dataset[0],
        Y=mode_optimiser.dataset[1],
        test_inputs=test_inputs,
        # static=False,
    )

    train_dataset_tf, num_batches_per_epoch = create_tf_dataset(
        train_dataset, num_data=train_dataset[0].shape[0], batch_size=batch_size
    )
    test_dataset_tf, _ = create_tf_dataset(
        test_dataset, num_data=test_dataset[0].shape[0], batch_size=batch_size
    )

    # training_loss = mode_optimiser.dynamics.build_training_loss(
    #     train_dataset_tf, compile=compile_loss_fn
    # )

    fast_tasks = init_fast_tasks_bounds(
        log_dir,
        train_dataset_tf,
        mode_optimiser.dynamics.mosvgpe,
        test_dataset=test_dataset_tf,
        # training_loss=training_loss,
        fast_tasks_period=fast_tasks_period,
    )
    slow_tasks = mogpe_plotter.tf_monitor_task_group(
        log_dir,
        slow_period=slow_tasks_period
        # slow_tasks_period=slow_tasks_period,
    )
    monitor = Monitor(fast_tasks, slow_tasks)

    # Init checkpoint manager for saving model during training
    manager = init_checkpoint_manager(
        model=mode_optimiser,
        log_dir=log_dir,
        num_ckpts=num_ckpts,
        mode_opt_gin_config=mode_opt_config_file,
        mogpe_toml_config=mogpe_config_file,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    training_spec = ModeOptDynamicsTrainingSpec(
        num_epochs=num_epochs,
        batch_size=batch_size,
        # optimiser=optimiser,
        learning_rate=learning_rate,
        logging_epoch_freq=logging_epoch_freq,
        compile_loss_fn=compile_loss_fn,
        monitor=monitor,
        manager=manager,
    )
    return mode_optimiser, training_spec, train_dataset


@gin.configurable
def config_traj_opt(
    mode_opt_ckpt_dir,
    mode_opt_config_file,
    max_iterations,
    # method,
    disp,
    mode_chance_constraint_lower,
    compile_loss_fn,
    compile_mode_constraint_fn,
    log_dir,
    num_ckpts,
    fast_tasks_period,
    slow_tasks_period,
    trajectory_optimiser,
    velocity_constraints_lower=None,
    velocity_constraints_upper=None,
    method: str = None,
    horizon: int = None,
    horizon_new: int = None,
    state_cost_weight: default_float() = None,
    control_cost_weight: default_float() = None,
    terminal_state_cost_weight: default_float() = None,
    riemannian_metric_cost_weight: default_float() = None,
    riemannian_metric_covariance_weight: default_float() = 1.0,
    prob_cost_weight=None,
    state_var_cost_weight=None,
):
    # Load data set from ckpt dir
    train_dataset = np.load(os.path.join(mode_opt_ckpt_dir, "train_dataset.npz"))
    dataset = (train_dataset["x"], train_dataset["y"])

    # Init mode_opt from gin config
    mode_optimiser = init_mode_opt(dataset=dataset, mode_opt_ckpt_dir=mode_opt_ckpt_dir)

    # Init policy
    print("velocity_constraints_lower")
    print(velocity_constraints_lower)
    print(velocity_constraints_upper)
    if horizon_new is not None:
        if isinstance(mode_optimiser.policy, DeterministicPolicy):
            policy_name = "DeterministicPolicy"
        elif isinstance(mode_optimiser.policy, VariationalGaussianPolicy):
            policy_name = "VariationalGaussianPolicy"
        policy = init_policy(
            policy_name,
            horizon_new,
            mode_optimiser.dynamics.control_dim,
            velocity_constraints_lower,
            velocity_constraints_upper,
        )
        mode_optimiser.policy = policy

    # Create nested log_dir inside learn_dynamics dir
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
            mode_opt_gin_config=mode_opt_config_file,
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
    prob_cost_weight = weight_to_matrix(prob_cost_weight, 1)
    riemannian_metric_weight_matrix = weight_to_matrix(
        riemannian_metric_cost_weight, state_dim + control_dim
    )

    state_means, state_vars = rollout_policy_in_dynamics(
        mode_optimiser.policy, mode_optimiser.dynamics, mode_optimiser.start_state
    )

    cost_fn = ZeroCostFunction()
    if Q_terminal is not None:
        cost_fn += TargetStateCostFunction(
            weight_matrix=Q_terminal, target_state=mode_optimiser.target_state
        )
        print("Using quadratic TERMINAL STATE cost")
    if R is not None:
        cost_fn += ControlQuadraticCostFunction(weight_matrix=R)
        print("Using quadratic CONTROL cost")
    if Q is not None:
        cost_fn += StateQuadraticCostFunction(weight_matrix=Q)
        print("Using quadratic STATE cost")
    if riemannian_metric_cost_weight is not None:
        cost_fn += RiemannianEnergyCostFunction(
            gp=mode_optimiser.dynamics.gating_gp,
            covariance_weight=riemannian_metric_covariance_weight,
            riemannian_metric_weight_matrix=riemannian_metric_weight_matrix,
        )
        print("Using RIEMANNIAN ENERGY cost")
    if prob_cost_weight is not None:
        cost_fn += ModeProbCostFunction(
            prob_fn=mode_optimiser.dynamics.predict_mode_probability,
            weight=prob_cost_weight,
        )
        print("Using MODE PROB cost")
    if state_var_cost_weight is not None:
        print("Using STATE VAR cost")
        cost_fn += StateVarianceCostFunction(weight=state_var_cost_weight)

    # weight_matrix = tf.eye(control_dim, dtype=default_float())
    # Q = weight_matrix * 0.0
    # R = weight_matrix * 0.1
    # # R = weight_matrix * 10.0
    # H = weight_matrix * 1000.0
    # cost_fn = TargetStateCostFunction(
    #     weight_matrix=H, target_state=mode_optimiser.target_state
    # )
    # print("target_s")
    # cost_fn += ControlQuadraticCostFunction(weight_matrix=R)

    # integral_cost_fn = partial(state_control_quadratic_cost_fn, Q=Q, R=R)
    # terminal_cost_fn = partial(
    #     terminal_state_cost_fn, Q=H, target_state=mode_optimiser.target_state
    # )

    # def cost_fn(state, control, state_var, control_var):
    #     # print("inside cost_fn")
    #     # print(state.shape)
    #     # print(control.shape)
    #     # print(state_var.shape)
    #     # print(control_var.shape)
    #     int_cost = integral_cost_fn(
    #         state=state[:-1, :],
    #         control=control,
    #         state_var=state_var[:-1, :],
    #         control_var=control_var,
    #     )
    #     terminal_cost = terminal_cost_fn(
    #         state=state[-1:, :], state_var=state_var[-1:, :]
    #     )
    #     # terminal_cost = terminal_cost_fn(state=state[-1:, :])
    #     cost = tf.reduce_sum(int_cost) + tf.reduce_sum(terminal_cost)
    #     # cost = tf.reduce_sum(terminal_cost)
    #     return cost

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
            cost_fn=cost_fn,
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
            cost_fn=cost_fn,
        )
    elif trajectory_optimiser == "ExplorativeTrajectoryOptimiser":
        print("ExplorativeTrajectoryOptimiser")
        training_spec = ExplorativeTrajectoryOptimiserTrainingSpec(
            max_iterations=max_iterations,
            method=method,
            disp=disp,
            mode_chance_constraint_lower=mode_chance_constraint_lower,
            compile_mode_constraint_fn=compile_mode_constraint_fn,
            compile_loss_fn=compile_loss_fn,
            monitor=monitor,
            manager=manager,
            cost_fn=cost_fn,
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


def init_mode_traj_opt_from_ckpt(ckpt_dir):
    mode_opt_config_file = os.path.join(ckpt_dir, "mode_opt_config.gin")
    gin.parse_config_files_and_bindings([mode_opt_config_file], None)
    mode_optimiser, training_spec = config_traj_opt(
        mode_opt_config_file=mode_opt_config_file, log_dir=None
    )
    gpf.utilities.print_summary(mode_optimiser)
    controls_init = mode_optimiser.policy()

    ckpt = tf.train.Checkpoint(model=mode_optimiser)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)
    print("Restored ModeOpt")
    gpf.utilities.print_summary(mode_optimiser)
    return mode_optimiser, controls_init, training_spec


def init_mode_opt_learn_dynamics_from_ckpt(ckpt_dir):
    mode_opt_config_file = os.path.join(ckpt_dir, "mode_opt_config.gin")
    gin.parse_config_files_and_bindings([mode_opt_config_file], None)
    mode_optimiser, training_spec, train_dataset = config_learn_dynamics(
        mode_opt_config_file=mode_opt_config_file
    )
    gpf.utilities.print_summary(mode_optimiser)

    ckpt = tf.train.Checkpoint(model=mode_optimiser)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)
    print("Restored ModeOpt")
    gpf.utilities.print_summary(mode_optimiser)
    return mode_optimiser, training_spec, train_dataset
