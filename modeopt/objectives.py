#!/usr/bin/env python3
from typing import Callable

import tensor_annotations.tensorflow as ttf
import tensorflow as tf

from modeopt.cost_functions import (
    ControlQuadraticCostFunction,
    CostFunction,
    RiemannianEnergyCostFunction,
    TargetStateCostFunction,
)
from modeopt.custom_types import State, StateDim
from modeopt.dynamics import ModeOptDynamics, SVGPDynamicsWrapper
from modeopt.rollouts import rollout_controls_in_dynamics

from .trajectories import BaseTrajectory, ControlTrajectoryDist

ObjectiveFn = Callable[[BaseTrajectory], ttf.Tensor0]


def build_riemannian_energy_objective(
    start_state,
    target_state,
    initial_solution,
    dynamics: ModeOptDynamics,
    riemannian_metric_cost_matrix: ttf.Tensor2[StateDim, StateDim],
    riemannian_metric_covariance_weight: float,
    terminal_state_cost_matrix: ttf.Tensor2[StateDim, StateDim],
    control_cost_weight,
) -> ObjectiveFn:
    energy_cost_fn = RiemannianEnergyCostFunction(
        gp=dynamics.gating_gp,
        riemannian_metric_weight_matrix=riemannian_metric_cost_matrix,
        covariance_weight=riemannian_metric_covariance_weight,
    )
    terminal_cost_fn = TargetStateCostFunction(
        weight_matrix=terminal_state_cost_matrix, target_state=target_state
    )
    control_cost_fn = ControlQuadraticCostFunction(weight_matrix=control_cost_weight)
    cost_fn = energy_cost_fn + terminal_cost_fn + control_cost_fn

    objective_fn = build_variational_objective(
        dynamics, initial_solution, cost_fn, start_state
    )
    return objective_fn


def build_variational_objective(
    dynamics: SVGPDynamicsWrapper,
    # control_trajectory: ControlTrajectoryDist,
    cost_fn: CostFunction,
    start_state: State,
) -> ObjectiveFn:
    def variational_objective(initial_solution: ControlTrajectoryDist) -> ttf.Tensor0:
        """Negative Evidence Lower BOund"""
        entropy = initial_solution.entropy()

        # Rollout controls in dynamics
        control_means, control_vars = initial_solution(variance=True)
        tf.print("control_vars IN BOUNd")
        tf.print(control_means)
        tf.print(control_vars)
        state_means, state_vars = rollout_controls_in_dynamics(
            dynamics=dynamics,
            start_state=start_state,
            control_means=control_means,
            control_vars=control_vars,
        )
        tf.print("state_means IN BOUNd")
        tf.print(state_means)
        tf.print(state_vars)

        # Calculate costs
        expected_costs = cost_fn(
            state=state_means,
            control=control_means,
            state_var=state_vars,
            control_var=control_vars,
        )
        # tf.print("Expected costs: {}".format(expected_costs.numpy()))
        # tf.print("Entropy: {}".format(entropy.numpy()))
        elbo = -expected_costs + entropy
        return -elbo

    return variational_objective


def build_mode_variational_objective(
    dynamics: SVGPDynamicsWrapper, cost_fn: CostFunction, start_state: State
) -> ObjectiveFn:
    def mode_variational_objective(
        initial_solution: ControlTrajectoryDist,
    ) -> ttf.Tensor0:
        """Evidence Lower BOund"""
        entropy = initial_solution.entropy()

        # Rollout controls in dynamics
        control_means, control_vars = initial_solution()
        state_means, state_vars = rollout_controls_in_dynamics(
            dynamics=dynamics,
            start_state=start_state,
            control_means=control_means,
            control_vars=control_vars,
        )

        # Calculate costs
        expected_costs = cost_fn(
            state=state_means,
            control=control_means,
            state_var=state_vars,
            control_var=control_vars,
        )

        # Calulate variational expectation over mode indicator
        mode_var_exp = dynamics.mode_variational_expectation(
            state_means[:-1, :], control_means, state_vars[:-1, :], control_vars
        )

        horizon = control_means.shape[0]
        elbo = mode_var_exp - expected_costs + entropy / horizon
        return elbo
