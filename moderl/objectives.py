#!/usr/bin/env python3
from typing import Callable

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float, default_jitter
from gpflow.conditionals import base_conditional, uncertain_conditional

from moderl.cost_functions import (
    ControlQuadraticCostFunction,
    CostFunction,
    RiemannianEnergyCostFunction,
    TargetStateCostFunction,
)
from moderl.custom_types import State, StateDim
from moderl.dynamics import ModeRLDynamics, SVGPDynamicsWrapper
from moderl.dynamics.conditionals import svgp_covariance_conditional
from moderl.mode_opt import ModeRL
from moderl.rollouts import rollout_controls_in_dynamics
from moderl.utils import combine_state_controls_to_input

from .trajectories import BaseTrajectory, ControlTrajectoryDist

tfd = tfp.distributions

ObjectiveFn = Callable[[BaseTrajectory], ttf.Tensor0]


def build_riemannian_energy_objective(
    start_state,
    target_state,
    initial_solution,
    dynamics: ModeRLDynamics,
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
        control_means, control_vars = initial_solution(variance=True)
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
        # elbo = -expected_costs + entropy / horizon
        # elbo = -expected_costs
        # elbo = mode_var_exp - expected_costs - entropy
        # elbo = mode_var_exp - expected_costs + entropy
        elbo = mode_var_exp - expected_costs + entropy / horizon
        # elbo = mode_var_exp - expected_costs
        return -elbo

    return mode_variational_objective


def build_explorative_objective(
    dynamics: SVGPDynamicsWrapper, cost_fn: CostFunction, start_state: State
) -> ObjectiveFn:
    def explorative_objective(
        initial_solution: ControlTrajectoryDist,
    ) -> ttf.Tensor0:
        # Rollout controls in dynamics
        control_means, control_vars = initial_solution(variance=True)
        state_means, state_vars = rollout_controls_in_dynamics(
            dynamics=dynamics,
            start_state=start_state,
            control_means=control_means,
            #         control_vars=control_vars,
        )

        h_means_prior, h_vars_prior = dynamics.uncertain_predict_gating(
            state_means[1:, :], control_means
        )
        gating_gp = dynamics.desired_mode_gating_gp

        input_means, input_vars = combine_state_controls_to_input(
            state_means[1:, :],
            control_means,
            state_vars[1:, :],
            control_vars,
        )

        h_means, h_vars = h_means_prior[0:1, :], h_vars_prior[0:1, :]
        for t in range(1, initial_solution.horizon):
            Xnew = input_means[t : t + 1, :]
            Xobs = input_means[0:t, :]
            f = h_means_prior[0:t, :]

            Knn = svgp_covariance_conditional(X1=Xnew, X2=Xnew, svgp=gating_gp)[0, 0, :]
            Kmm = svgp_covariance_conditional(X1=Xobs, X2=Xobs, svgp=gating_gp)[0, :, :]
            Kmn = svgp_covariance_conditional(X1=Xobs, X2=Xnew, svgp=gating_gp)[0, :, :]
            Kmm += tf.eye(Kmm.shape[0], dtype=default_float()) * default_jitter()
            # Lm = tf.linalg.cholesky(Kmm)
            # A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]
            h_mean, h_var = base_conditional(
                Kmn=Kmn,
                Kmm=Kmm,
                Knn=Knn,
                f=f,
                full_cov=False,
                q_sqrt=None,
                white=False,
            )
            h_means = tf.concat([h_means, h_mean], 0)
            h_vars = tf.concat([h_vars, h_var], 0)
        h_dist = tfd.MultivariateNormalDiag(h_means, h_vars)
        gating_entropy = h_dist.entropy()

        return -tf.reduce_sum(gating_entropy) + cost_fn(
            state_means, control_means, state_vars, control_vars
        )

    return explorative_objective
