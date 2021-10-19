#!/usr/bin/env python3
from functools import partial
from typing import Callable, NewType, Optional, Union

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from geoflow.manifolds import GPManifold
from gpflow import default_float
from gpflow.models import GPModel
from tensor_annotations import axes
from tensor_annotations.axes import Batch

from modeopt.policies import DeterministicPolicy, VariationalGaussianPolicy

StateDim = NewType("StateDim", axes.Axis)
ControlDim = NewType("ControlDim", axes.Axis)
InputDim = Union[StateDim, ControlDim]
One = NewType("One", axes.Axis)
Trajectory = NewType("Trajectory", axes.Axis)


def quadratic_cost_fn(
    vector: ttf.Tensor2[Batch, InputDim],
    weight_matrix: Union[
        ttf.Tensor2[InputDim, InputDim], ttf.Tensor3[Batch, InputDim, InputDim]
    ],
    vector_var: Optional[ttf.Tensor2[Batch, InputDim]] = None,
):
    assert len(vector.shape) == 2
    vector = tf.expand_dims(vector, -2)
    cost = vector @ weight_matrix @ tf.transpose(vector, [0, 2, 1])
    if vector_var is not None:
        assert len(vector_var.shape) == 2
        vector_var = tf.expand_dims(vector_var, -2)  # [Batch, 1, Dim]
        trace = tf.linalg.trace(vector_var @ weight_matrix)  # [Batch,]
        cost += trace  # [Batch, 1, 1]
    return cost[:, 0, 0]


def state_control_quadratic_cost_fn(
    state: ttf.Tensor2[Batch, StateDim],
    control: ttf.Tensor2[Batch, ControlDim],
    Q: Union[ttf.Tensor2[StateDim, StateDim], ttf.Tensor3[Batch, StateDim, StateDim]],
    R: Union[
        ttf.Tensor2[ControlDim, ControlDim], ttf.Tensor3[Batch, ControlDim, ControlDim]
    ],
    state_var: Optional[ttf.Tensor2[Batch, StateDim]] = None,
    control_var: Optional[ttf.Tensor2[Batch, ControlDim]] = None,
):
    state_cost = quadratic_cost_fn(state, Q, state_var)
    control_cost = quadratic_cost_fn(control, R, control_var)
    return state_cost + control_cost


def terminal_state_cost_fn(
    state: ttf.Tensor2[One, StateDim],
    Q: Union[ttf.Tensor2[StateDim, StateDim], ttf.Tensor3[Batch, StateDim, StateDim]],
    target_state: ttf.Tensor2[One, StateDim],
    state_var: Optional[ttf.Tensor2[Batch, StateDim]] = None,
):
    error = state - target_state
    terminal_cost = quadratic_cost_fn(error, Q, state_var)
    return terminal_cost


def build_riemmanian_energy_cost_fn(
    gp: GPModel,
    riemmanian_metric_cost_weight: float = 1.0,
    covariance_weight: float = 1.0,
) -> Callable:
    manifold = GPManifold(gp, covariance_weight=covariance_weight)
    return partial(
        riemmanian_energy_cost_fn,
        manifold=manifold,
        riemmanian_metric_cost_weight=riemmanian_metric_cost_weight,
    )


def riemmanian_energy_cost_fn(
    state_trajectory: ttf.Tensor2[Trajectory, StateDim],
    control_trajectory: ttf.Tensor2[Trajectory, ControlDim],
    manifold: GPManifold,
    riemmanian_metric_cost_weight: float = 1.0,
    state_trajectory_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
    control_trajectory_var: Optional[ttf.Tensor2[Trajectory, ControlDim]] = None,
):
    # Calcualted the expeted metric at each point along trajectory
    input_mean = tf.concat([state_trajectory, control_trajectory], -1)
    expected_riemannian_metric = (
        manifold.metric(input_mean[1:, :])
        * riemmanian_metric_cost_weight
        # manifold.metric(input_mean[:-1 :]) * riemmanian_metric_cost_weight
    )

    velocities = input_mean[1:, :] - input_mean[:-1, :]
    velocities_var = None

    if state_trajectory_var is not None and control_trajectory_var is not None:
        input_var = tf.concat([state_trajectory_var, control_trajectory_var], -1)
        velocities_var = input_var[1:, :]

    riemannian_energy = quadratic_cost_fn(
        vector=velocities,
        weight_matrix=expected_riemannian_metric,
        # vector_var=None,
        vector_var=velocities_var,
    )
    riemannian_energy_sum = tf.reduce_sum(riemannian_energy)
    return riemannian_energy_sum


def build_state_control_riemannian_energy_quadratic_cost_fn(
    Q: ttf.Tensor2[StateDim, StateDim],
    R: ttf.Tensor2[ControlDim, ControlDim],
    gp: GPModel,
    riemannian_metric_cost_weight: default_float(),
    riemannian_metric_covariance_weight: default_float(),
):
    """Build quadratic cost func with state, control and Riemannian energy

    c(x(t),u(t)) = x(t)^T Q x(t) + u(t)^T R u(t) + \dot{v(t)}^T G(v(t)) \dot{v(t)}^T

    where v = (x, u) and where G(v) is the Riemannian metric at v.
    """
    state_control_cost_fn = partial(state_control_quadratic_cost_fn, Q=Q, R=R)
    riemannian_energy_cost_fn = build_riemmanian_energy_cost_fn(
        gp=gp,
        riemmanian_metric_cost_weight=riemannian_metric_cost_weight,
        covariance_weight=riemannian_metric_covariance_weight,
    )

    def state_control_riemannian_energy_quadratic_cost_fn(
        state, control, state_var, control_var
    ):
        state_control_cost = state_control_cost_fn(
            state=state,
            control=control,
            state_var=state_var,
            control_var=control_var,
        )
        state_control_cost_sum = tf.reduce_sum(state_control_cost)
        riemannian_cost = riemannian_energy_cost_fn(
            state_trajectory=state,
            control_trajectory=control,
            # state_trajectory_var=state_var,
            # control_trajectory_var=control_var,
            state_trajectory_var=None,
            control_trajectory_var=None,
        )
        return state_control_cost_sum + riemannian_cost

    return state_control_riemannian_energy_quadratic_cost_fn


# def terminal_cost_fn(terminal, Q, target, terminal_var=None):
#     error = terminal - target
#     terminal_cost = quadratic_cost_fn(error, Q, terminal_var)
#     return terminal_cost


# def state_control_terminal_cost_fn(
#     terminal_state,
#     terminal_control,
#     Q,
#     R,
#     target_state=None,
#     target_control=None,
#     terminal_state_var=None,
#     terminal_control_var=None,
# ):
#     terminal_cost = 0
#     state_cost = terminal_cost_fn(terminal_state, Q, target_state, terminal_state_var)
#     terminal_cost += state_cost
#     # if Q is not None:
#     #     state_cost = terminal_cost_fn(
#     #         terminal_state, Q, target_state, terminal_state_var
#     #     )
#     #     terminal_cost += state_cost
#     # if R is not None:
#     #     control_cost = terminal_cost_fn(
#     #         terminal_control, R, target_control, terminal_control_var
#     #     )
#     #     terminal_cost += control_cost
#     return terminal_cost


def expected_quadratic_costs(
    cost_fn: Union[
        quadratic_cost_fn,
        state_control_quadratic_cost_fn,
    ],
    terminal_cost_fn: terminal_state_cost_fn,
    state_means: ttf.Tensor2[Batch, StateDim],
    state_vars: Optional[ttf.Tensor2[Batch, StateDim]],
    policy: Union[VariationalGaussianPolicy, DeterministicPolicy],
):
    """Calculate expected costs under Gaussian states"""
    if isinstance(policy, VariationalGaussianPolicy):
        control_means, control_vars = policy()
        expected_integral_costs = cost_fn(
            state=state_means[:-1, :],
            control=control_means,
            state_var=state_vars[:-1, :],
            control_var=control_vars,
        )
        expected_terminal_cost = terminal_cost_fn(
            state=state_means[-1:, :], state_var=state_vars[-1:, :]
        )
    elif isinstance(policy, DeterministicPolicy):
        control_means, control_vars = policy()
        expected_integral_costs = cost_fn(
            state=state_means[:-1, :],
            control=control_means,
            state_var=state_vars[:-1, :],
            control_var=control_vars,
        )
        expected_terminal_cost = terminal_cost_fn(
            state=state_means[-1:, :], state_var=state_vars[-1:, :]
        )
    else:
        # TODO approximate expected cost with samples in non Gaussian case?
        raise NotImplementedError
    return expected_integral_costs, expected_terminal_cost[0]


# def expected_quadratic_costs(cost_fn, state_means, state_vars, policy):
#     """Calculate expected costs under Gaussian states"""
#     if isinstance(policy, VariationalGaussianPolicy):
#         control_means, control_vars = policy()
#         expected_integral_costs = cost_fn(
#             state=state_means[:-1, :],
#             control=control_means,
#             state_var=state_vars[:-1, :],
#             control_var=control_vars,
#         )
#         expected_terminal_cost = terminal_cost_fn(
#             state=state_means[-1:, :], state_var=state_vars[-1:, :]
#         )
#     elif isinstance(policy, DeterministicPolicy):
#         control_means, control_vars = policy()
#         print("deterministic policy mean and vars")
#         print(control_means)
#         print(control_vars)
#         expected_integral_costs = cost_fn(
#             state=state_means[:-1, :],
#             control=control_means,
#             state_var=state_vars[:-1, :],
#             control_var=control_vars,
#         )
#         expected_terminal_cost = terminal_cost_fn(
#             state=state_means[-1:, :], state_var=state_vars[-1, :]
#         )
#     else:
#         # TODO approximate expected cost with samples in non Gaussian case?
#         raise NotImplementedError
#     return expected_integral_costs, expected_terminal_cost[0]
