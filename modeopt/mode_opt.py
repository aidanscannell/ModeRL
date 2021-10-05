#!/usr/bin/env python3
from functools import partial
from typing import NewType, Tuple, Union

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import Module, default_float
from tensor_annotations import axes
from tensor_annotations.axes import Batch
from tf_agents.environments import tf_py_environment

from modeopt.constraints import build_mode_chance_constraints_scipy
from modeopt.cost_functions import (
    state_control_quadratic_cost_fn,
    terminal_state_cost_fn,
)
from modeopt.dynamics.multimodal import ModeOptDynamics
from modeopt.policies import VariationalPolicy, VariationalGaussianPolicy
from modeopt.rollouts import rollout_policy_in_dynamics, rollout_policy_in_env
from modeopt.trajectory_optimisers import (
    ExplorativeTrajectoryOptimiser,
    ExplorativeTrajectoryOptimiserTrainingSpec,
    ModeVariationalTrajectoryOptimiser,
    ModeVariationalTrajectoryOptimiserTrainingSpec,
)

StateDim = NewType("StateDim", axes.Axis)
ControlDim = NewType("ControlDim", axes.Axis)


class ModeOpt(Module):
    def __init__(
        self,
        start_state: ttf.Tensor2[Batch, StateDim],
        target_state: ttf.Tensor2[Batch, StateDim],
        env,
        policy: VariationalPolicy,
        dynamics: ModeOptDynamics,
        dataset: Tuple,
        desired_mode: int = 1,
        mode_chance_constraint_lower=0.5,
        horizon: int = 10,
        state_cost_weight: default_float() = 1.0,
        control_cost_weight: default_float() = 1.0,
        terminal_state_cost_weight: default_float() = 1.0,
    ):
        self.start_state = start_state
        self.target_state = target_state
        self.dynamics = dynamics
        self.dataset = dataset
        self.desired_mode = desired_mode
        self.mode_chance_constraint_lower = mode_chance_constraint_lower
        self.horizon = horizon
        self.state_cost_weight = state_cost_weight
        self.control_cost_weight = control_cost_weight
        self.terminal_state_cost_weight = terminal_state_cost_weight

        if policy is None:
            control_means = (
                np.ones((horizon, self.dynamics.control_dim)) * 0.5
                + np.random.random((horizon, self.dynamics.control_dim)) * 0.1
            )
            control_vars = (
                np.ones((horizon, self.dynamics.control_dim)) * 0.2
                + np.random.random((horizon, self.dynamics.control_dim)) * 0.01
            )
            self.policy = VariationalGaussianPolicy(
                means=control_means, vars=control_vars
            )

        else:
            self.policy = policy

        # Init integral quadratic costs on state and controls
        self.Q = (
            tf.eye(self.dynamics.state_dim, dtype=default_float()) * state_cost_weight
        )
        self.R = (
            tf.eye(self.dynamics.control_dim, dtype=default_float())
            * control_cost_weight
        )
        self.Q_terminal = tf.eye(
            self.dynamics.state_dim, dtype=default_float()
        ) * np.array(terminal_state_cost_weight).reshape(1, -1)
        self.cost_fn = partial(state_control_quadratic_cost_fn, Q=self.Q, R=self.R)
        # self.cost_fn = partial(state_control_quadratic_cost_fn, Q=Q, R=R)

        # Init terminal quadratic costs on states (Euclidean distance)
        self.terminal_cost_fn = partial(
            terminal_state_cost_fn, Q=self.Q_terminal, target_state=self.target_state
        )

        # Init tf environment
        self.env = env
        self.tf_env = tf_py_environment.TFPyEnvironment(env)

        self.mode_trajectory_optimiser = ModeVariationalTrajectoryOptimiser(
            self.policy,
            self.dynamics,
            self.cost_fn,
            self.terminal_cost_fn,
        )

        self.explorative_trajectory_optimiser = ExplorativeTrajectoryOptimiser(
            self.policy,
            self.dynamics,
            self.cost_fn,
            self.terminal_cost_fn,
        )
        # self.trajectory_optimiser = self.mode_trajectory_optimiser
        self.trajectory_optimiser = self.explorative_trajectory_optimiser

    def optimise_policy(
        self,
        start_state,
        training_spec: Union[
            ModeVariationalTrajectoryOptimiserTrainingSpec,
            ExplorativeTrajectoryOptimiserTrainingSpec,
        ],
    ):
        if (
            training_spec.mode_chance_constraint_lower is None
            or training_spec.mode_chance_constraint_lower <= 0.0
        ):
            mode_chance_constraints = []
            print(
                "Turning mode chance constraints off because training_spec.mode_chance_constraint_lower is None or <=0.0"
            )
        else:
            mode_chance_constraints = build_mode_chance_constraints_scipy(
                mode_opt_dynamics=self.dynamics,
                start_state=start_state,
                horizon=self.horizon,
                lower_bound=training_spec.mode_chance_constraint_lower,
                upper_bound=1.0,
                compile=training_spec.compile_mode_constraint_fn,
            )
        return self.trajectory_optimiser.optimise(
            start_state=self.start_state,
            training_spec=training_spec,
            constraints=mode_chance_constraints,
        )

    def dynamics_rollout(self, start_state, start_state_var=None):
        return rollout_policy_in_dynamics(
            self.policy, self.dynamics, start_state, start_state_var=start_state_var
        )

    def env_rollout(self, start_state):
        return rollout_policy_in_env(self.env, self.policy, start_state=start_state)

    # def optimise_dynamics(self, dataset, num_epochs=2000, batch_size=256):
    #     # Create tf dataset that can be iterated and build training loss closure
    #     train_dataset, num_batches_per_epoch = create_tf_dataset(
    #         dataset, num_data=dataset[0].shape[0], batch_size=batch_size
    #     )
    #     training_loss = self.mogpe_model.training_loss_closure(iter(train_dataset))

    #     @tf.function
    #     def tf_optimization_step():
    #         self.optimizer.minimize(training_loss, self.mogpe_model.trainable_variables)

    #     # monitor = Monitor(fast_tasks, slow_tasks)

    #     for epoch in range(num_epochs):
    #         for _ in range(num_batches_per_epoch):
    #             tf_optimization_step()
    #         if self.monitor is not None:
    #             self.monitor(epoch)
    #         epoch_id = epoch + 1
    #         if epoch_id % self.logging_epoch_freq == 0:
    #             tf.print(f"Epoch {epoch_id}: Dynamics ELBO (train) {training_loss()}")
    #             if self.manager is not None:
    #                 self.manager.save()
