#!/usr/bin/env python3
import typing
from dataclasses import dataclass

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from modeopt.cost_functions import (
    CostFunction,
    quadratic_cost_fn,
)
from modeopt.dynamics import GPDynamics
from modeopt.policies import VariationalPolicy
from modeopt.rollouts import rollout_policy_in_dynamics
from modeopt.trajectory_optimisers.base import TrajectoryOptimiser
from tensor_annotations import axes
from tensor_annotations.axes import Batch

tfd = tfp.distributions

StateDim = typing.NewType("StateDim", axes.Axis)
ControlDim = typing.NewType("ControlDim", axes.Axis)


def binary_entropy(probs):
    return -probs * tf.math.log(probs) - (1 - probs) * tf.math.log(1 - probs)


def entropy_approx(h_means, h_vars, mode_probs):
    C = tf.constant(np.sqrt(math.pi * np.log(2.0) / 2.0), dtype=default_float())
    param_entropy = C * tf.exp(-(h_means ** 2) / (2 * (h_vars ** 2 + C ** 2)))
    param_entropy = param_entropy / (tf.sqrt(h_vars ** 2 + C ** 2))
    print("param_entropy")
    print(param_entropy)
    model_entropy = binary_entropy(mode_probs)
    print(model_entropy)
    return model_entropy - param_entropy


@dataclass
class ExplorativeTrajectoryOptimiserTrainingSpec:
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


class ExplorativeTrajectoryOptimiser(TrajectoryOptimiser):
    def __init__(
        self,
        policy: VariationalPolicy,
        dynamics: GPDynamics,
        cost_fn: CostFunction,
    ):
        super().__init__(
            policy=policy,
            dynamics=dynamics,
            cost_fn=cost_fn,
        )

    def objective(
        self,
        start_state: ttf.Tensor2[Batch, StateDim],
        start_state_var: ttf.Tensor2[Batch, StateDim] = None,
    ):
        """Optimise trajectories starting from an initial state"""
        entropy = self.policy.entropy()  # calculate entropy of policy dist

        # Rollout controls in dynamics
        state_means, state_vars = rollout_policy_in_dynamics(
            self.policy,
            self.dynamics,
            start_state,
            start_state_var=start_state_var,
        )

        # Calculate costs
        control_means, control_vars = self.policy()
        expected_costs = self.cost_fn(
            state=state_means,
            control=control_means,
            state_var=state_vars,
            control_var=control_vars,
        )
        print("expected_cost")
        print(expected_costs)
        # expected_integral_costs, expected_terminal_cost = expected_quadratic_costs(
        #     cost_fn=self.cost_fn,
        #     terminal_cost_fn=self.terminal_cost_fn,
        #     state_means=state_means,
        #     state_vars=state_vars,
        #     policy=self.policy,
        # )  # [Batch,], []

        # self.dynamics.predict_mode_probability_given_latent(
        #     h_mean=gating_means, h_var=gating_vars
        # )
        # predict_mode_probability(state_mean, control_mean, state_var, control_var)
        # control_means, control_vars = self.policy()
        mode_var_exp = self.dynamics.mode_variational_expectation(
            state_means[:-1, :], control_means, state_vars[:-1, :], control_vars
        )

        # gating_means, gating_vars = self.dynamics.uncertain_predict_gating(
        #     state_means[:-1, :], control_means, state_vars[:-1, :], control_vars
        # )
        gating_means, gating_vars = self.dynamics.uncertain_predict_gating(
            state_means[:-1, :], control_means
        )
        gating_entropy_old = tfd.Normal(gating_means, gating_vars)
        gating_entropy_old = gating_entropy_old.entropy()
        gating_entropy_sum_old = tf.reduce_sum(gating_entropy_old)

        gating_means, gating_vars = self.dynamics.gating_conditional_entropy(
            state_means[:-1, :], control_means, state_vars[:-1, :], control_vars
        )
        gating_entropy = tfd.Normal(gating_means, gating_vars)
        gating_entropy = gating_entropy.entropy()
        tf.print("gating_entropy_old")
        tf.print(gating_entropy_old)
        tf.print("gating_entropy")
        tf.print(gating_entropy)
        gating_entropy_sum = tf.reduce_sum(gating_entropy)
        tf.print("gating_entropy_sum")
        tf.print(gating_entropy_sum)

        tf.print(mode_var_exp)

        probs = self.dynamics.predict_mode_probability(
            state_means[:-1, :], control_means, state_vars[:-1, :], control_vars
        )
        print("probs")
        print(probs)
        prob_errors = (probs - 0.7) ** 2
        tf.print("prob_errors")
        tf.print(prob_errors)
        prob_errors_sum = tf.reduce_sum(prob_errors)
        tf.print("prob_errors_sum")
        tf.print(prob_errors_sum)

        # manifold = GPManifold(self.dynamics.gating_gp, covariance_weight=0.05)
        input_mean = tf.concat([state_means[:-1, :], control_means], -1)
        input_var = tf.concat([state_vars[:-1, :], control_vars], -1)
        velocities = input_mean[1:, :] - input_mean[:-1, :]
        velocities_var = input_var[1:, :]
        length_weight_matrix = (
            tf.eye(velocities.shape[1], dtype=default_float()) * 0.001
        )
        print("length_weight_matrix")
        print(length_weight_matrix)
        euclidean_energy = tf.reduce_sum(
            quadratic_cost_fn(
                vector=velocities,
                weight_matrix=length_weight_matrix,
                vector_var=None,
            )
        )
        tf.print("euclidean_energy")
        tf.print(euclidean_energy)

        J = (
            euclidean_energy
            - prob_errors_sum
            # -mode_var_exp * 0.1
            # -mode_var_exp * 0.1
            # -mode_var_exp * 0.05
            # -expected_terminal_cost
            # - tf.reduce_sum(expected_integral_costs)
            # tf.reduce_sum(expected_integral_costs)
            # + entropy
            # + gating_entropy_sum_old
            # + gating_entropy_sum
            # gating_entropy_sum
            # tf.reduce_sum(control_means)
        )
        tf.print("J")
        tf.print(J)
        tf.print("expected costs")
        tf.print(-tf.reduce_sum(expected_integral_costs))
        # print("mode_var_exp")
        # print(mode_var_exp)
        # print("expected_terminal_cost")
        # print(expected_terminal_cost)
        # print("expected_integral_cost")
        # print(expected_integral_costs)
        # print("entropy")
        # print(entropy)
        return J
