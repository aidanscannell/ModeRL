#!/usr/bin/env python3
import tensorflow as tf

from modeopt.policies import (
    DeterministicPolicy,
    VariationalGaussianPolicy,
    VariationalPolicy,
)

# def quadratic(state, control, state_weight, control_weight):
#     state_cost = tf.transpose(state) @ state_weight @ state
#     control_cost = tf.transpose(control) @ control_weight @ control
#     cost = state_cost + control_cost
#     return cost


# def expected_quadratic(state, control, state_weight, control_weight):
#     state_cost = tf.transpose(state) @ state_weight @ state
#     control_cost = tf.transpose(control) @ control_weight @ control
#     cost = state_cost + control_cost
#     return cost


def quadratic_cost_fn(vector, weight_matrix, vector_var=None):
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
    state, control, Q, R, state_var=None, control_var=None
):
    print("state")
    print(state)
    print(state_var)
    print(Q)
    state_cost = quadratic_cost_fn(state, Q, state_var)
    print("state_cost")
    print(state_cost)
    control_cost = quadratic_cost_fn(control, R, control_var)
    return state_cost + control_cost


def terminal_state_cost_fn(state, Q, target_state, state_var=None):
    error = state - target_state
    terminal_cost = quadratic_cost_fn(error, Q, state_var)
    return terminal_cost


def terminal_cost_fn(terminal, Q, target, terminal_var=None):
    error = terminal - target
    terminal_cost = quadratic_cost_fn(error, Q, terminal_var)
    return terminal_cost


def state_control_terminal_cost_fn(
    terminal_state,
    terminal_control,
    Q,
    R,
    target_state=None,
    target_control=None,
    terminal_state_var=None,
    terminal_control_var=None,
):
    terminal_cost = 0
    state_cost = terminal_cost_fn(terminal_state, Q, target_state, terminal_state_var)
    terminal_cost += state_cost
    # if Q is not None:
    #     state_cost = terminal_cost_fn(
    #         terminal_state, Q, target_state, terminal_state_var
    #     )
    #     terminal_cost += state_cost
    # if R is not None:
    #     control_cost = terminal_cost_fn(
    #         terminal_control, R, target_control, terminal_control_var
    #     )
    #     terminal_cost += control_cost
    return terminal_cost


def expected_quadratic_costs(
    cost_fn, terminal_cost_fn, state_means, state_vars, policy
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
        print("deterministic policy mean and vars")
        print(control_means)
        print(control_vars)
        expected_integral_costs = cost_fn(
            state=state_means[:-1, :],
            control=control_means,
            state_var=state_vars[:-1, :],
            control_var=control_vars,
        )
        expected_terminal_cost = terminal_cost_fn(
            state=state_means[-1:, :], state_var=state_vars[-1, :]
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
