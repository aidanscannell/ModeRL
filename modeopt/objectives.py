#!/usr/bin/env python3
from functools import partial
from typing import Callable, Optional

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from geoflow.manifolds import GPManifold
from gpflow.models import GPModel
from modeopt.cost_functions import (
    CostFunction,
    ControlQuadraticCostFunction,
    CostFunction,
    RiemannianEnergyCostFunction,
    TargetStateCostFunction,
)
from modeopt.custom_types import ControlTrajectory, State, StateDim
from modeopt.dynamics import ModeOptDynamics, SVGPDynamicsWrapper
from modeopt.mode_opt import ModeOpt
from modeopt.rollouts import rollout_controls_in_dynamics

from .constraints import hermite_simpson_collocation_constraints_fn
from .trajectories import BaseTrajectory, ControlTrajectoryDist, FlatOutputTrajectory

ObjectiveFn = Callable[[BaseTrajectory], ttf.Tensor0]


def build_riemannian_energy_objective_from_ModeOpt(
    mode_optimiser: ModeOpt,
    initial_solution,
    riemannian_metric_cost_matrix: ttf.Tensor2[StateDim, StateDim],
    riemannian_metric_covariance_weight: float,
    terminal_state_cost_matrix: ttf.Tensor2[StateDim, StateDim],
    control_cost_weight,
) -> ObjectiveFn:
    return build_riemannian_energy_objective(
        mode_optimiser.start_state,
        mode_optimiser.target_state,
        mode_optimiser.dynamics,
        riemannian_metric_cost_matrix,
        riemannian_metric_covariance_weight,
        terminal_state_cost_matrix,
        control_cost_weight,
    )


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


def build_collocation_objective(trajectory, cost_fn: CostFunction):
    def collocation_objective(initial_solution: FlatOutputTrajectory):
        states = trajectory.flat_output_to_state(initial_solution)
        controls = trajectory.flat_output_to_control(initial_solution)

        cost = cost_fn(states, controls)
        print("cost")
        print(cost.shape)
        cost = tf.reduce_sum(cost)
        print(cost.shape)
        return cost


# def build_geodesic_collocation_lagrange_objective(
#     gp: GPModel, covariance_weight: float = 1.0, cost_fn: Optional[CostFunction] = None
# ) -> Callable[[FlatOutputTrajectory], ttf.Tensor0]:

#     manifold = GPManifold(gp=gp, covariance_weight=covariance_weight)
#     geodesic_constraints_fn = partial(
#         hermite_simpson_collocation_constraints_fn, ode_fn=manifold.geodesic_ode
#     )

#     def objective_fn(initial_solution: FlatOutputTrajectory) -> ttf.Tensor0:
#         eq_constraints = geodesic_constraints_fn(initial_solution=initial_solution)
#         print("eq_constraints")
#         print(eq_constraints)
#         lagrange_term = tf.reduce_sum(
#             tf.reshape(initial_solution.lagrange_multipliers, [-1]) * eq_constraints
#         )
#         print("lagrange_term")
#         print(lagrange_term)
#         lagrange_objective = lagrange_term
#         return lagrange_objective

#     if cost_fn is not None:

#         def objective_fn_with_cost(
#             initial_solution: FlatOutputTrajectory,
#         ) -> ttf.Tensor0:
#             cost = cost_fn(
#                 state=initial_solution.states, control=initial_solution.controls
#             )
#             print("cost")
#             print(cost)
#             lagrange_objective = objective_fn(initial_solution)
#             return cost + lagrange_objective

#         return objective_fn_with_cost
#     else:
#         return objective_fn


def build_geodesic_collocation_lagrange_objective(
    gp: GPModel, cost_fn: CostFunction, covariance_weight: float = 1.0
) -> Callable[[FlatOutputTrajectory], ttf.Tensor0]:

    manifold = GPManifold(gp=gp, covariance_weight=covariance_weight)
    geodesic_constraints_fn = partial(
        hermite_simpson_collocation_constraints_fn, ode_fn=manifold.geodesic_ode
    )

    def objective_fn(initial_solution: FlatOutputTrajectory) -> ttf.Tensor0:
        return cost_fn(state=initial_solution.states, control=initial_solution.controls)

    return build_lagrange_dual_objective(
        eq_constraints_fn=geodesic_constraints_fn, objective_fn=objective_fn
    )


def build_lagrange_dual_objective(
    eq_constraints_fn, objective_fn: Optional[CostFunction] = None
) -> Callable[[FlatOutputTrajectory], ttf.Tensor0]:
    def lagrange_dual_objective_fn(
        initial_solution: FlatOutputTrajectory,
    ) -> ttf.Tensor0:
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(initial_solution.trainable_variables)
            eq_constraints = eq_constraints_fn(initial_solution=initial_solution)
            loss = objective_fn(initial_solution) - tf.reduce_sum(
                tf.reshape(initial_solution.lagrange_multipliers, [-1]) * eq_constraints
            )
            print("loss")
            print(loss)
        grad_loss = tape.gradient(loss, initial_solution.trainable_variables)
        print("grad_loss")
        print(grad_loss)
        lagrange_loss = 0
        for grads in grad_loss:
            lagrange_loss += tf.reduce_sum(grads)
        print("lagrange_loss")
        print(lagrange_loss)
        # return grad_loss
        return -lagrange_loss
        # lagrange_objective = grad_loss + eq

    return lagrange_dual_objective_fn
