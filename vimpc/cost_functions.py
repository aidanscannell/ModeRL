#!/usr/bin/env python3
import tensorflow as tf


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
    cost = vector @ weight_matrix @ tf.transpose(vector)
    if vector_var is not None:
        cost += tf.linalg.trace(vector_var @ weight_matrix)
    return cost


def state_control_quadratic_cost_fn(
    state, control, Q, R, state_var=None, control_var=None
):
    # print("cost")
    # print(state.shape)
    # print(control.shape)
    # print(control_var.shape)
    state_cost = quadratic_cost_fn(state, Q, state_var)
    control_cost = quadratic_cost_fn(control, R, control_var)
    return state_cost + control_cost


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
