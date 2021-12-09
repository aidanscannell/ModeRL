#!/usr/bin/env python3
from functools import partial
from typing import Callable, NewType, Optional, Union

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from geoflow.manifolds import GPManifold
from gpflow import default_float
from gpflow.models import GPModel
from tensor_annotations import axes
from modeopt.policies import (
    VariationalGaussianPolicy,
    VariationalPolicy,
    DeterministicPolicy,
)

StateDim = NewType("StateDim", axes.Axis)
ControlDim = NewType("ControlDim", axes.Axis)
InputDim = Union[StateDim, ControlDim]
One = NewType("One", axes.Axis)
Trajectory = NewType("Trajectory", axes.Axis)


class CostFunction:
    """
    The base cost function class.

    To implement a mean function, write the __call__ method.
    """

    def __call__(
        self,
        state: ttf.Tensor2[Trajectory, StateDim],
        control: ttf.Tensor2[Trajectory, ControlDim],
        state_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Trajectory, ControlDim]] = None,
    ):
        """Expected cost func under Normally distributed states and controls

        E[ c(x_T, u_T) ] = \mu_{e_T}^T H \mu_{e_T} + tr(H \Sigma_{e_T})
        with:
            e = x_T - target_state ~ N(\mu_e, \Sigma_e)
            x ~ N(\mu_x, \Sigma_x)
            T is final time step

        :param state: Tensor representing a state trajectory
        :param control: Tensor representing control trajectory
        :param state_var: Tensor representing the variance over state trajectory
        :param control_var: Tensor representing control trajectory
        :returns: scalar cost
        """
        raise NotImplementedError(
            "Implement the __call__ method for this cost function"
        )

    def __add__(self, other):
        return Additive(self, other)


class Additive(CostFunction):
    def __init__(self, first_part, second_part):
        CostFunction.__init__(self)
        self.add_1 = first_part
        self.add_2 = second_part

    def __call__(
        self,
        state: ttf.Tensor2[Trajectory, StateDim],
        control: ttf.Tensor2[Trajectory, ControlDim],
        state_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Trajectory, ControlDim]] = None,
    ):
        return tf.add(
            self.add_1(
                state=state,
                control=control,
                state_var=state_var,
                control_var=control_var,
            ),
            self.add_2(
                state=state,
                control=control,
                state_var=state_var,
                control_var=control_var,
            ),
        )


class ZeroCostFunction(CostFunction):
    def __call__(
        self,
        state: ttf.Tensor2[Trajectory, StateDim],
        control: ttf.Tensor2[Trajectory, ControlDim],
        state_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Trajectory, ControlDim]] = None,
    ):
        return tf.constant(0.0, dtype=default_float())


class StateQuadraticCostFunction(CostFunction):
    def __init__(
        self,
        weight_matrix: Union[
            ttf.Tensor2[StateDim, StateDim], ttf.Tensor3[Trajectory, StateDim, StateDim]
        ],
    ):
        self.weight_matrix = weight_matrix

    def __call__(
        self,
        state: ttf.Tensor2[Trajectory, StateDim],
        control: ttf.Tensor2[Trajectory, ControlDim],
        state_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Trajectory, ControlDim]] = None,
    ):
        """Expected quadratic integral state cost func

        E[ \Sum_{t=0}^{T-1} c(x_t, u_t) ] = \mu_{x_t}^T Q \mu_{x_t} + tr(Q \Sigma_{x_t})

        with:
            x ~ N(\mu_x, \Sigma_x)

        :param state: Tensor representing a state trajectory
        :param control: Tensor representing control trajectory
        :param state_var: Tensor representing the variance over state trajectory
        :param control_var: Tensor representing control trajectory
        :returns: scalar cost
        """
        if state_var is None:
            vector_var = None
        else:
            vector_var = state_var[:-1, :]
        state_costs = quadratic_cost_fn(
            vector=state[:-1, :],
            # vector=state,
            weight_matrix=self.weight_matrix,
            vector_var=vector_var,
            # vector_var=state_var,
        )
        return tf.reduce_sum(state_costs)


class ControlQuadraticCostFunction(CostFunction):
    def __init__(
        self,
        weight_matrix: Union[
            ttf.Tensor2[StateDim, StateDim], ttf.Tensor3[Trajectory, StateDim, StateDim]
        ],
    ):
        self.weight_matrix = weight_matrix

    def __call__(
        self,
        state: ttf.Tensor2[Trajectory, StateDim],
        control: ttf.Tensor2[Trajectory, ControlDim],
        state_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Trajectory, ControlDim]] = None,
    ):
        """Expected quadratic integral control cost func

        E[ \Sum_{t=0}^{T-1} c(x_t, u_t) ] = \mu_{u_t}^T R \mu_{u_t} + tr(R \Sigma_{u_t})

        with:
            u ~ N(\mu_u, \Sigma_u)

        :param state: Tensor representing a state trajectory
        :param control: Tensor representing control trajectory
        :param state_var: Tensor representing the variance over state trajectory
        :param control_var: Tensor representing control trajectory
        :returns: scalar cost
        """
        control_costs = quadratic_cost_fn(
            vector=control,
            weight_matrix=self.weight_matrix,
            vector_var=control_var,
        )
        return tf.reduce_sum(control_costs)


class TargetStateCostFunction(CostFunction):
    def __init__(
        self,
        weight_matrix: Union[
            ttf.Tensor2[StateDim, StateDim], ttf.Tensor3[Trajectory, StateDim, StateDim]
        ],
        target_state: ttf.Tensor2[One, StateDim],
    ):
        self.weight_matrix = weight_matrix
        self.target_state = target_state

    def __call__(
        self,
        state: ttf.Tensor2[Trajectory, StateDim],
        control: ttf.Tensor2[Trajectory, ControlDim],
        state_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Trajectory, ControlDim]] = None,
    ):
        """Expected quadratic terminal state cost func

        E[ c(x_T, u_T) ] = \mu_{e_T}^T H \mu_{e_T} + tr(H \Sigma_{e_T})

        with:
            e = x_T - target_state ~ N(\mu_e, \Sigma_e)
            x ~ N(\mu_x, \Sigma_x)
            T is final time step

        :param state: Tensor representing a state trajectory
        :param control: Tensor representing control trajectory
        :param state_var: Tensor representing the variance over state trajectory
        :param control_var: Tensor representing control trajectory
        :returns: scalar cost
        """
        error = state[-1:, :] - self.target_state
        # error = state - self.target_state
        if state_var is None:
            terminal_state_var = None
        else:
            terminal_state_var = state_var[-1:, :]
        terminal_cost = quadratic_cost_fn(
            vector=error,
            weight_matrix=self.weight_matrix,
            vector_var=terminal_state_var,
            # vector_var=state_var,
        )
        return terminal_cost
        # return terminal_cost[0]


class RiemannianEnergyCostFunction(CostFunction):
    """Riemannian energy cost function class."""

    def __init__(
        self,
        gp: GPModel,
        riemannian_metric_weight_matrix: Union[
            ttf.Tensor2[StateDim, StateDim], ttf.Tensor3[Trajectory, StateDim, StateDim]
        ],
        covariance_weight: default_float() = 1.0,
    ):
        self.gp = gp
        self.covariance_weight = covariance_weight
        self.manifold = GPManifold(gp=self.gp, covariance_weight=self.covariance_weight)
        self.riemannian_metric_weight_matrix = riemannian_metric_weight_matrix

    def __call__(
        self,
        state: ttf.Tensor2[Trajectory, StateDim],
        control: ttf.Tensor2[Trajectory, ControlDim],
        state_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Trajectory, ControlDim]] = None,
    ):
        """Expected Riemannian energy cost func

        It uses the expected Riemannian metric tensor E[G(v_t)] at time t

        E[ \Sum_{t=0}^{T-1} c(x_t, u_t) ] \approx
                \Sum_{t=0}^{T-1} \dot{\mu_{v_t}}^T E[G(v_t)] \dot{\mu{v_t}}^T + tr(E[G(v_t)] \Sigma_{v_t})

        with:
            x ~ N(\mu_x, \Sigma_x)
            u ~ N(\mu_u, \Sigma_u)
            v = (v, u) ~ N(\mu_v, \Sigma_v)
            E[G(v_t)] = E_{J(x_t)} [J(x_t)^T J(x_t)] = \mu_{x_t}^T \mu_{x_t} + covariance_weight \Sigma_{J_t}

        :param state: Tensor representing a state trajectory
        :param control: Tensor representing control trajectory
        :param state_var: Tensor representing the variance over state trajectory
        :param control_var: Tensor representing control trajectory
        :returns: scalar cost
        """
        if state_var is None:
            state_var = None
        else:
            state_var = state_var[:-1, :]
        energy_costs = riemmanian_energy_cost_fn(
            manifold=self.manifold,
            riemmanian_metric_weight_matrix=self.riemmanian_metric_weight_matrix,
            state_trajectory=state[:-1, :],
            # state_trajectory=state,
            control_trajectory=control,
            # state_trajectory_var=state_var,
            # control_trajectory_var=control_var,
            state_trajectory_var=None,
            control_trajectory_var=None,
        )
        return tf.reduce_sum(energy_costs)


class ModeProbCostFunction(CostFunction):
    """Simple mode probability cost function class. """

    def __init__(
        self,
        prob_fn: Callable,  # ModeOptDynamics.predict_mode_probability
        weight: default_float() = 1.0,
    ):
        self.prob_fn = prob_fn
        self.weight = weight

    def __call__(
        self,
        state: ttf.Tensor2[Trajectory, StateDim],
        control: ttf.Tensor2[Trajectory, ControlDim],
        state_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Trajectory, ControlDim]] = None,
    ):
        if state_var is None:
            state_var = None
        else:
            state_var = state_var[:-1, :]

        probs = self.prob_fn(
            state_mean=state[:-1, :],
            control_mean=control,
            state_var=state_var,
            control_var=control_var,
        )
        negative_probs = -probs * self.weight
        return tf.reduce_sum(negative_probs)


def quadratic_cost_fn(
    vector: ttf.Tensor2[Trajectory, InputDim],
    weight_matrix: Union[
        ttf.Tensor2[InputDim, InputDim], ttf.Tensor3[Trajectory, InputDim, InputDim]
    ],
    vector_var: Optional[ttf.Tensor2[Trajectory, InputDim]] = None,
):
    assert len(vector.shape) == 2
    vector = tf.expand_dims(vector, -2)
    cost = vector @ weight_matrix @ tf.transpose(vector, [0, 2, 1])
    if vector_var is not None:
        assert len(vector_var.shape) == 2
        vector_var = tf.expand_dims(vector_var, -2)  # [Trajectory, 1, Dim]
        trace = tf.linalg.trace(vector_var @ weight_matrix)  # [Trajectory,]
        cost += trace  # [Trajectory, 1, 1]
    return cost[:, 0, 0]


def state_control_quadratic_cost_fn(
    state: ttf.Tensor2[Trajectory, StateDim],
    control: ttf.Tensor2[Trajectory, ControlDim],
    Q: Union[
        ttf.Tensor2[StateDim, StateDim], ttf.Tensor3[Trajectory, StateDim, StateDim]
    ],
    R: Union[
        ttf.Tensor2[ControlDim, ControlDim],
        ttf.Tensor3[Trajectory, ControlDim, ControlDim],
    ],
    state_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
    control_var: Optional[ttf.Tensor2[Trajectory, ControlDim]] = None,
):
    state_cost = quadratic_cost_fn(state, Q, state_var)
    tf.print("state_cost yo")
    tf.print(state_cost)
    control_cost = quadratic_cost_fn(control, R, control_var)
    tf.print(control_cost)
    print("state_cost yo")
    print(state_cost)
    print(control_cost)
    # return state_cost + control_cost
    return tf.reduce_sum(state_cost) + tf.reduce_sum(control_cost)


def terminal_state_cost_fn(
    state: ttf.Tensor2[One, StateDim],
    Q: Union[
        ttf.Tensor2[StateDim, StateDim], ttf.Tensor3[Trajectory, StateDim, StateDim]
    ],
    target_state: ttf.Tensor2[One, StateDim],
    state_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
):
    error = state - target_state
    terminal_cost = quadratic_cost_fn(error, Q, state_var)
    return terminal_cost


# def build_riemannian_energy_cost_fn(
#     gp: GPModel,
#     riemannian_metric_weight_matrix: float = 1.0,
#     covariance_weight: float = 1.0,
# ) -> Callable:
#     manifold = GPManifold(gp, covariance_weight=covariance_weight)
#     return partial(
#         riemannian_energy_cost_fn,
#         manifold=manifold,
#         riemannian_metric_weight_matrix=riemannian_metric_weight_matrix,
#     )


def riemannian_energy_cost_fn(
    state_trajectory: ttf.Tensor2[Trajectory, StateDim],
    control_trajectory: ttf.Tensor2[Trajectory, ControlDim],
    manifold: GPManifold,
    riemannian_metric_weight_matrix: float = 1.0,
    state_trajectory_var: Optional[ttf.Tensor2[Trajectory, StateDim]] = None,
    control_trajectory_var: Optional[ttf.Tensor2[Trajectory, ControlDim]] = None,
):
    # Calcualted the expeted metric at each point along trajectory
    input_mean = tf.concat([state_trajectory, control_trajectory], -1)
    expected_riemannian_metric = (
        manifold.metric(input_mean[1:, :])
        @ riemmanian_metric_weight_matrix  # manifold.metric(input_mean[:-1 :]) * riemmanian_metric_cost_weight
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


# def build_state_control_riemannian_energy_quadratic_cost_fn(
#     Q: ttf.Tensor2[StateDim, StateDim],
#     R: ttf.Tensor2[ControlDim, ControlDim],
#     gp: GPModel,
#     riemannian_metric_weight_matrix: default_float(),
#     riemannian_metric_covariance_weight: default_float(),
# ):
#     """Build quadratic cost func with state, control and Riemannian energy

#     c(x(t),u(t)) = x(t)^T Q x(t) + u(t)^T R u(t) + \dot{v(t)}^T G(v(t)) \dot{v(t)}^T

#     where v = (x, u) and where G(v) is the Riemannian metric at v.
#     """
#     state_control_cost_fn = partial(state_control_quadratic_cost_fn, Q=Q, R=R)
#     riemannian_energy_cost_fn = build_riemannian_energy_cost_fn(
#         gp=gp,
#         riemannian_metric_weight_matrix=riemannian_metric_weight_matrix,
#         covariance_weight=riemannian_metric_covariance_weight,
#     )

#     def state_control_riemannian_energy_quadratic_cost_fn(
#         state, control, state_var, control_var
#     ):
#         state_control_cost = state_control_cost_fn(
#             state=state,
#             control=control,
#             state_var=state_var,
#             control_var=control_var,
#         )
#         tf.print("state_control_costs")
#         tf.print(state_control_cost)
#         state_control_cost_sum = tf.reduce_sum(state_control_cost)
#         riemannian_cost = riemannian_energy_cost_fn(
#             state_trajectory=state,
#             control_trajectory=control,
#             # state_trajectory_var=state_var,
#             # control_trajectory_var=control_var,
#             state_trajectory_var=None,
#             control_trajectory_var=None,
#         )
#         tf.print("riemannian_cost")
#         tf.print(riemannian_cost)
#         return state_control_cost_sum + riemannian_cost

#     return state_control_riemannian_energy_quadratic_cost_fn


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


# def expected_quadratic_costs(
#     cost_fn: Union[
#         quadratic_cost_fn,
#         state_control_quadratic_cost_fn,
#     ],
#     terminal_cost_fn: terminal_state_cost_fn,
#     state_means: ttf.Tensor2[Trajectory, StateDim],
#     state_vars: Optional[ttf.Tensor2[Trajectory, StateDim]],
#     policy: Union[VariationalGaussianPolicy, DeterministicPolicy],
# ):
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
#             state=state_means[-1:, :],
#             state_var=state_vars[-1:, :],
#             control=None,
#             control_var=None,
#         )
#     elif isinstance(policy, DeterministicPolicy):
#         control_means, control_vars = policy()
#         expected_integral_costs = cost_fn(
#             state=state_means[:-1, :],
#             control=control_means,
#             state_var=state_vars[:-1, :],
#             control_var=control_vars,
#         )
#         expected_terminal_cost = terminal_cost_fn(
#             state=state_means[-1:, :],
#             state_var=state_vars[-1:, :],
#             control=None,
#             control_var=None,
#         )
#     else:
#         # TODO approximate expected cost with samples in non Gaussian case?
#         raise NotImplementedError
#     tf.print("terminal_cost")
#     tf.print(expected_terminal_cost[0])
#     return expected_integral_costs, expected_terminal_cost[0]


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
