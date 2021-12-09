#!/usr/bin/env python3
import typing
from dataclasses import dataclass

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from modeopt.dynamics import GPDynamics
from modeopt.cost_functions import CostFunction
from modeopt.policies import VariationalPolicy
from modeopt.rollouts import rollout_policy_in_dynamics
from modeopt.trajectory_optimisers.base import (
    TrajectoryOptimiser,
    TrajectoryOptimiserTrainingSpec,
)
from tensor_annotations import axes
from tensor_annotations.axes import Batch

StateDim = typing.NewType("StateDim", axes.Axis)
ControlDim = typing.NewType("ControlDim", axes.Axis)


@dataclass
class VariationalTrajectoryOptimiserTrainingSpec(TrajectoryOptimiserTrainingSpec):
    """
    Specification data class for model training. Models that require additional parameters for
    training should create a subclass of this class and add additional properties.
    """

    max_iterations: int = 100
    method: str = "SLSQP"
    disp: bool = True
    mode_chance_constraint_lower: float = None  # lower bound on mode probability over traj, set as None to turn off mode constraints
    compile_mode_constraint_fn: bool = True  # constraints fn in tf.function?
    compile_loss_fn: bool = True  # loss function in tf.function?
    monitor: gpf.monitor.Monitor = None
    manager: tf.train.CheckpointManager = None


@dataclass
class ModeVariationalTrajectoryOptimiserTrainingSpec:
    """
    Specification data class for model training. Models that require additional parameters for
    training should create a subclass of this class and add additional properties.
    """

    max_iterations: int = 100
    method: str = "SLSQP"
    disp: bool = True
    mode_chance_constraint_lower: float = None  # lower bound on mode probability over traj, set as None to turn off mode constraints
    compile_mode_constraint_fn: bool = True  # constraints fn in tf.function?
    compile_loss_fn: bool = True  # loss function in tf.function?
    monitor: gpf.monitor.Monitor = None
    manager: tf.train.CheckpointManager = None
    cost_fn: CostFunction = None


class VariationalTrajectoryOptimiser(TrajectoryOptimiser):
    """
    A trajectory optimiser optimises a sequence of actions given a model of the
    the environment that is used for virtual rollouts.
    """

    def __init__(
        self,
        policy: VariationalPolicy,
        dynamics: GPDynamics,
        cost_fn: CostFunction,
        optimiser=gpf.optimizers.Scipy(),
    ):
        super().__init__(
            policy=policy, dynamics=dynamics, cost_fn=cost_fn, optimiser=optimiser
        )
        self._training_loss = None

    def objective(self, start_state: ttf.Tensor2[Batch, StateDim]):
        return self.elbo(start_state=start_state)

    def elbo(self, start_state: ttf.Tensor2[Batch, StateDim]):
        """Evidence LOwer Bound"""
        entropy = self.policy.entropy()  # calculate entropy of policy dist

        # Rollout controls in dynamics
        state_means, state_vars = rollout_policy_in_dynamics(
            self.policy, self.dynamics, start_state
        )

        # Calculate costs
        control_means, control_vars = self.policy()
        expected_costs = self.cost_fn(
            state=state_means,
            control=control_means,
            state_var=state_vars,
            control_var=control_vars,
        )

        elbo = -expected_costs + entropy
        return elbo


class ModeVariationalTrajectoryOptimiser(VariationalTrajectoryOptimiser):
    """
    Trajectory optimiser that optimises a policy using variational inference.
    The evidence lower bound includes a conditioning on a mode indicator
    variable, resulting trajectories that attempt to remain in a desired mode.
    """

    def __init__(
        self,
        policy: VariationalPolicy,
        dynamics: GPDynamics,
        cost_fn: CostFunction,
        optimiser=gpf.optimizers.Scipy(),
    ):
        super().__init__(
            policy=policy, dynamics=dynamics, cost_fn=cost_fn, optimiser=optimiser
        )

    def elbo(self, start_state: ttf.Tensor2[Batch, StateDim]):
        """Optimise trajectories starting from an initial state"""
        entropy = self.policy.entropy()  # calculate entropy of policy dist

        # Rollout controls in dynamics
        state_means, state_vars = rollout_policy_in_dynamics(
            self.policy, self.dynamics, start_state
        )

        # Calculate costs
        control_means, control_vars = self.policy()
        expected_costs = self.cost_fn(
            state=state_means,
            control=control_means,
            state_var=state_vars,
            control_var=control_vars,
        )

        # Calulate variational expectation over mode indicator
        mode_var_exp = self.dynamics.mode_variational_expectation(
            state_means[:-1, :], control_means, state_vars[:-1, :], control_vars
        )

        elbo = -mode_var_exp - expected_costs + entropy
        return elbo
