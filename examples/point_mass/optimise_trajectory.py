#!/usr/bin/env python3
import time
from datetime import datetime
from functools import partial

import gpflow as gpf
import numpy as np
import pkg_resources
import tensorflow as tf
from gpflow.config import default_float
from gpflow.monitor import (
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
from matplotlib import patches
from mogpe.training.monitor import ImageWithCbarToTensorBoard
from mogpe.training.utils import update_model_from_checkpoint
from simenvs.core import make
from simenvs.point_mass.plotter import Plotter2D
from vimpc.controllers import VIMPC
from vimpc.cost_functions import (
    state_control_terminal_cost_fn,
    state_control_quadratic_cost_fn,
)
from vimpc.dynamics import GPDynamics
from vimpc.policies import GaussianPolicy, GaussianMixturePolicy

from point_mass.learn_dynamics import create_model


def env_rollout(policy):
    """Rollout a given policy on the environment

    :param policy: Callable representing policy to rollout
    :param timesteps: number of timesteps to rollout
    :returns: (states, delta_states)
    """
    num_time_steps = policy.num_time_steps
    time_step = env.reset(start_state.numpy())
    states = time_step.observation
    transitions = []
    delta_states = []
    # print("env rollout")
    # print(states)
    for t in range(num_time_steps - 1):
        # env.render()
        control = policy(t).numpy()
        next_time_step = env.step(control)
        state = next_time_step.observation
        # transitions.append([time_step, action, next_time_step])
        delta_states.append(next_time_step.observation - states[-1, :])
        states = np.concatenate([states, next_time_step.observation])
        time_step = next_time_step
    return np.stack(states), None
    # return np.stack(states), np.stack(delta_states)


def rollout(policy):
    """Rollout a given policy on the environment

    :param policy: Callable representing policy to rollout
    :param timesteps: number of timesteps to rollout
    :returns: (states, delta_states)
    """
    num_time_steps = policy.num_time_steps
    # time_step = env.reset(start_state.numpy())
    state_means = start_state
    state_var = np.array([[0.0, 0.0]])
    state_vars = state_var
    state = start_state
    # transitions = []
    delta_states = []
    control_means = policy.variational_dist.mean().numpy()
    control_vars = policy.variational_dist.variance().numpy()
    # actions = []
    for t in range(num_time_steps):
        # control = policy(t).numpy()
        control = control_means[t : t + 1, :]
        control_var = control_vars[t : t + 1, :]
        # print("control")
        # print(control)
        # next_state_mean, next_state_var = controller.dynamics(state, control)
        next_state_mean, next_state_var = controller.dynamics(
            state, control, state_var, control_var
        )
        state = next_state_mean
        state_var = next_state_var
        state_means = np.concatenate([state_means, next_state_mean], 0)
        state_vars = np.concatenate([state_vars, next_state_var], 0)
    return state_means, state_vars
    # return np.stack(states), np.stack(delta_states)


def plot_boundary_conditions(fig, ax, start_state, target_state):
    ax.scatter(start_state[0, 0], start_state[0, 1], marker="x", color="k")
    ax.scatter(target_state[0, 0], target_state[0, 1], color="k", marker="x")
    ax.annotate(
        "Start $\mathbf{x}_0$",
        (start_state[0, 0], start_state[0, 1]),
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax.annotate(
        "End $\mathbf{x}_f$",
        (target_state[0, 0], target_state[0, 1]),
        horizontalalignment="left",
        verticalalignment="top",
    )
    return fig, ax


# def quadratic_cost_fn(state, control, Q, R, state_var=None, control_var=None):
#     # print("cost")
#     # print(state.shape)
#     # print(control.shape)
#     # print(control_var.shape)
#     # target_state_broadcast = tf.expand_dims(target_state, 0)
#     # state_error = state - target_state_broadcast
#     # terminal_state_cost = state_error @ Q @ tf.transpose(state_error, [0, 2, 1])
#     # state_cost = state @ Q @ tf.transpose(state, [0, 2, 1])
#     state_cost = state @ Q @ tf.transpose(state)
#     control_cost = control @ R @ tf.transpose(control)
#     if control_var is not None:
#         control_cost += tf.linalg.trace(control_var @ R)
#     if state_var is not None:
#         state_cost += tf.linalg.trace(state_var @ Q)
#     # control_cost = control @ R @ tf.transpose(control, [0, 2, 1])
#     return state_cost + control_cost
#     # return state_cost
#     # return state_cost + terminal_state_cost
#     # return terminal_state_cost


if __name__ == "__main__":
    # Configure environment
    scenario = "scenario-1"
    scenario = "scenario-3"
    scenario = "scenario-4"
    env = make("point-mass/" + scenario)
    log_dir = (
        "./logs/point_mass/"
        + scenario
        + "/two_experts/opt_traj/"
        + datetime.now().strftime("%m-%d-%H%M%S")
    )

    # Set start/target states
    start_state = np.array([-3.0, -2.0, 0.0, 0.0])
    target_state = np.array([3.0, 2.5, 0.0, 0.0])
    # start_state = tf.constant(np.array([[-3.0, -2.0]]), dtype=default_float())
    start_state = tf.constant(np.array([[-3.0, 0.0]]), dtype=default_float())
    target_state = tf.constant(np.array([[3.0, 2.5]]), dtype=default_float())
    # target_state = tf.constant(np.array([[-0.5, 2.5]]), dtype=default_float())
    # target_state = tf.constant(np.array([[1.5, 2.5]]), dtype=default_float())

    # Load data set from npz file
    data_path = pkg_resources.resource_filename(
        "simenvs",
        "point_mass/" + scenario + "/data/point_mass_random_10000.npz"
        # "simenvs", "point_mass/" + scenario + "/data/point_mass_1000_4.npz"
    )
    data = np.load(data_path)
    X = data["x"]
    Y = data["y"]
    output_dim = Y.shape[1]
    input_dim = X.shape[1]
    num_data = Y.shape[0]
    state_dim = output_dim
    control_dim = input_dim - state_dim

    # Load trained mogpe and use it to initialise GPDynamics
    num_inducing = 150
    num_samples = 1
    model = create_model(
        X, output_dim=output_dim, num_inducing=num_inducing, num_samples=num_samples
    )
    # ckpt_dir = "./logs/point_mass/scenario-4/learn/two_experts/07-14-193958"
    ckpt_dir = "./logs/point_mass/scenario-4/learn/two_experts/07-15-001953"
    model = update_model_from_checkpoint(model, ckpt_dir)
    gp = model.experts.experts_list[1]
    print("Expert")
    gpf.utilities.print_summary(gp)
    gp_dynamics = GPDynamics(gp)
    env.state_init = start_state

    # Configure training tasks (plotting etc)
    plotter = Plotter2D(gp, X, Y)
    # plotter.plot_model()
    # plt.show()

    # Params for variational optimal control
    num_timesteps = 15
    num_epochs = 15000
    num_batches_per_epoch = 1
    logging_epoch_freq = 1
    slow_tasks_period = 1
    fast_tasks_period = 1
    logging_epoch_freq = 50
    slow_tasks_period = 100
    fast_tasks_period = 10
    num_ckpts = 5
    gpf.set_trainable(model, False)
    gating_gp = model.gating_network

    # covariance_weight = 1.0
    # manifold = GPManifold(gp=gating_gp, covariance_weight=covariance_weight)
    # print("manifold")
    # print(manifold)

    Q = tf.eye(state_dim, dtype=default_float()) * 0.01
    R = tf.eye(control_dim, dtype=default_float()) * 0.01
    cost_fn = partial(state_control_quadratic_cost_fn, Q=Q, R=R)

    Q_terminal = tf.eye(state_dim, dtype=default_float()) * 1.0
    terminal_cost_fn_ = partial(
        state_control_terminal_cost_fn,
        Q=Q_terminal,
        R=None,
        target_state=target_state,
        target_control=None,
    )

    control_means = (
        np.ones((num_timesteps, state_dim)) * 0.5
        + np.random.random((num_timesteps, state_dim)) * 10
    ) * np.array([[1.0, -1.0]])
    control_vars = np.ones((num_timesteps, state_dim)) * 0.2 + np.random.random(
        (num_timesteps, state_dim)
    )
    policy = GaussianPolicy(control_means, control_vars)

    # control_means = []
    # control_vars = []
    # num_mixtures = 3
    # mixture_probs = []
    # for _ in range(num_timesteps):
    #     mixture_probs.append([0.5, 0.5, 0.5])
    # control_means = np.ones(
    #     (num_timesteps, num_mixtures, state_dim)
    # ) * 0.5 + np.random.random(
    #     (num_timesteps, num_mixtures, state_dim)
    # ) * 10 * np.array(
    #     [[1.0, -1.0]]
    # )
    # control_vars = np.ones(
    #     (num_timesteps, num_mixtures, state_dim)
    # ) * 0.2 + np.random.random((num_timesteps, num_mixtures, state_dim))
    # policy = GaussianMixturePolicy(control_means, control_vars, mixture_probs)

    controller = VIMPC(
        cost_fn=cost_fn,
        terminal_cost_fn=terminal_cost_fn_,
        dynamics=gp_dynamics,
        gating_gp=gating_gp,
        policy=policy,
        # monotonic_fn=monotonic_fn,
    )
    print("Controller")
    gpf.utilities.print_summary(controller)

    def training_loss_closure():
        return controller.training_loss(start_state)

    plot_boundary_conditions = partial(
        plot_boundary_conditions, start_state=start_state, target_state=target_state
    )

    def plot_traj_over_prob(fig, ax):
        prob = model.predict_mixing_probs(plotter.test_inputs)[:, 1]
        prob_contf = ax.tricontourf(
            plotter.test_inputs[:, 0], plotter.test_inputs[:, 1], prob
        )
        cbars = plotter.cbar(fig, ax, prob_contf)
        state_trajectory, state_trajectory_vars = rollout(controller._policy)
        control_vars = controller._policy.variational_dist.variance()
        # print("state_trajectory")
        # print(state_trajectory)
        # print(state_trajectory.shape)
        if len(state_trajectory.shape) == 3:
            state_trajectory = state_trajectory[:, 0, :]
            if state_trajectory_vars is not None:
                state_trajectory_vars = state_trajectory_vars[:, 0, :]
        fig, ax = plot_boundary_conditions(fig, ax)
        ax.plot(state_trajectory[:, 0], state_trajectory[:, 1], marker="x", color="k")
        return cbars

    def plot_traj(fig, axs, rollout):
        plot_var = False
        # plot_var = True
        plot_state_var = True
        means, vars = controller.dynamics.gp.predict_f(plotter.test_inputs)
        cbars = plotter.plot_gps_shared_cbar(fig, axs, means, vars)
        state_trajectory, state_trajectory_vars = rollout(controller._policy)
        control_vars = controller._policy.variational_dist.variance()
        # print("state_trajectory")
        # print(state_trajectory)
        # print(state_trajectory.shape)
        if len(state_trajectory.shape) == 3:
            state_trajectory = state_trajectory[:, 0, :]
            if state_trajectory_vars is not None:
                state_trajectory_vars = state_trajectory_vars[:, 0, :]
        for ax in axs.flatten():
            fig, ax = plot_boundary_conditions(fig, ax)
            ax.plot(
                state_trajectory[:, 0], state_trajectory[:, 1], marker="x", color="k"
            )
            if plot_var:
                for state, control_var in zip(state_trajectory, control_vars.numpy()):
                    # print("control_var")
                    # print(control_var)
                    ax.add_patch(
                        patches.Ellipse(
                            (state[0], state[1]),
                            control_var[0],
                            control_var[1],
                            # control_var[0] * 100,
                            # control_var[1] * 100,
                            facecolor="none",
                            edgecolor="b",
                            linewidth=0.1,
                            alpha=0.6,
                        )
                    )
        if plot_state_var:
            if state_trajectory_vars is not None:
                for state, state_var in zip(state_trajectory, state_trajectory_vars):
                    ax.add_patch(
                        patches.Ellipse(
                            (state[0], state[1]),
                            state_var[0],
                            state_var[1],
                            # control_var[0] * 100,
                            # control_var[1] * 100,
                            facecolor="none",
                            edgecolor="magenta",
                            linewidth=0.1,
                            alpha=0.6,
                        )
                    )
        return cbars

    # fig, axs = plt.subplots(2, 2)
    # plot_traj(fig, axs)
    # plt.show()

    # image_task_traj = ImageWithCbarToTensorBoard(
    image_task_traj = ImageWithCbarToTensorBoard(
        log_dir,
        partial(plot_traj, rollout=rollout),
        name="simulated_trajectory_internal",
        fig_kw={"figsize": (10, 2)},
        subplots_kw={"nrows": 2, "ncols": 2},
    )
    image_task_env_traj = ImageWithCbarToTensorBoard(
        log_dir,
        partial(plot_traj, rollout=env_rollout),
        name="simulated_trajectory_env",
        fig_kw={"figsize": (10, 2)},
        subplots_kw={"nrows": 2, "ncols": 2},
    )
    image_task_prob_traj = ImageWithCbarToTensorBoard(
        log_dir,
        plot_traj_over_prob,
        name="simulated_trajectory_prob",
        fig_kw={"figsize": (10, 2)},
        subplots_kw={"nrows": 1, "ncols": 1},
    )
    image_tasks = [image_task_traj, image_task_env_traj, image_task_prob_traj]

    # slow_tasks = plotter.tf_monitor_task_group(log_dir, slow_tasks_period)
    slow_tasks = MonitorTaskGroup(image_tasks, period=slow_tasks_period)
    elbo_task = ScalarToTensorBoard(log_dir, training_loss_closure, "elbo")
    controller_task = ModelToTensorBoard(log_dir, controller)
    # cost_task = ScalarToTensorBoard(log_dir, cost_fn, "cost")
    fast_tasks = MonitorTaskGroup(
        [controller_task, elbo_task], period=fast_tasks_period
    )
    monitor = Monitor(fast_tasks, slow_tasks)

    ckpt = tf.train.Checkpoint(model=controller)
    manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=num_ckpts)

    # optimizer = tf.optimizers.Adam(0.01)
    optimizer = tf.optimizers.Adam(0.05)
    # optimizer = tf.optimizers.Adam(0.1)

    @tf.function
    def tf_optimization_step():
        optimizer.minimize(training_loss_closure, controller.trainable_variables)

    t = time.time()
    for epoch in range(num_epochs):
        tf_optimization_step()
        monitor(epoch)
        epoch_id = epoch + 1
        print(f"Epoch {epoch_id}\n-------------------------------")
        if epoch_id % logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss_closure()}")
            if manager is not None:
                manager.save()
            duration = time.time() - t
            print("Iteration duration: ", duration)
            t = time.time()

    # monitored_training_loop(
    #     controller,
    #     training_loss,
    #     epochs=num_epochs,
    #     fast_tasks=fast_tasks,
    #     slow_tasks=slow_tasks,
    #     num_batches_per_epoch=num_batches_per_epoch,
    #     logging_epoch_freq=logging_epoch_freq,
    #     manager=manager,
    # )
