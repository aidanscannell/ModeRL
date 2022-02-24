#!/usr/bin/env python3
from typing import Callable, NewType, Optional, Union

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from geoflow.manifolds import GPManifold
from gpflow import default_float
from gpflow.models import GPModel
from tensor_annotations import axes

from modeopt.utils import append_zero_control, combine_state_controls_to_input

StateDim = NewType("StateDim", axes.Axis)
ControlDim = NewType("ControlDim", axes.Axis)
InputDim = Union[StateDim, ControlDim]
One = NewType("One", axes.Axis)
Horizon = NewType("Horizon", axes.Axis)
HorizonPlusOne = NewType("HorizonPlusOne", axes.Axis)
Batch = NewType("Batch", axes.Axis)


class CostFunction:
    """Base cost function class.

    To implement a mean function, write the __call__ method.
    """

    def __call__(
        self,
        state: ttf.Tensor2[HorizonPlusOne, StateDim],
        control: ttf.Tensor2[Horizon, ControlDim],
        state_var: Optional[ttf.Tensor2[HorizonPlusOne, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Horizon, ControlDim]] = None,
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
        state: ttf.Tensor2[HorizonPlusOne, StateDim],
        control: ttf.Tensor2[Horizon, ControlDim],
        state_var: Optional[ttf.Tensor2[HorizonPlusOne, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Horizon, ControlDim]] = None,
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
        state: ttf.Tensor2[HorizonPlusOne, StateDim],
        control: ttf.Tensor2[Horizon, ControlDim],
        state_var: Optional[ttf.Tensor2[HorizonPlusOne, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Horizon, ControlDim]] = None,
    ):
        return tf.constant(0.0, dtype=default_float())


class StateQuadraticCostFunction(CostFunction):
    def __init__(
        self,
        weight_matrix: Union[
            ttf.Tensor2[StateDim, StateDim],
            ttf.Tensor3[HorizonPlusOne, StateDim, StateDim],
        ],
    ):
        self.weight_matrix = weight_matrix

    def __call__(
        self,
        state: ttf.Tensor2[HorizonPlusOne, StateDim],
        control: ttf.Tensor2[Horizon, ControlDim],
        state_var: Optional[ttf.Tensor2[HorizonPlusOne, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Horizon, ControlDim]] = None,
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
            ttf.Tensor2[StateDim, StateDim], ttf.Tensor3[Horizon, StateDim, StateDim]
        ],
    ):
        self.weight_matrix = weight_matrix

    def __call__(
        self,
        state: ttf.Tensor2[HorizonPlusOne, StateDim],
        control: ttf.Tensor2[Horizon, ControlDim],
        state_var: Optional[ttf.Tensor2[HorizonPlusOne, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Horizon, ControlDim]] = None,
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
        weight_matrix: ttf.Tensor2[StateDim, StateDim],
        target_state: ttf.Tensor2[One, StateDim],
    ):
        self.weight_matrix = weight_matrix
        self.target_state = target_state

    def __call__(
        self,
        state: ttf.Tensor2[HorizonPlusOne, StateDim],
        control: ttf.Tensor2[Horizon, ControlDim],
        state_var: Optional[ttf.Tensor2[HorizonPlusOne, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Horizon, ControlDim]] = None,
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
            ttf.Tensor2[StateDim, StateDim],
            ttf.Tensor3[HorizonPlusOne, StateDim, StateDim],
        ],
        covariance_weight: default_float() = 1.0,
    ):
        self.gp = gp
        self.covariance_weight = covariance_weight
        self.manifold = GPManifold(gp=self.gp, covariance_weight=self.covariance_weight)
        self.riemannian_metric_weight_matrix = riemannian_metric_weight_matrix

    def __call__(
        self,
        state: ttf.Tensor2[HorizonPlusOne, StateDim],
        control: ttf.Tensor2[Horizon, ControlDim],
        state_var: Optional[ttf.Tensor2[HorizonPlusOne, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Horizon, ControlDim]] = None,
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

        energy_costs = riemannian_energy_cost_fn(
            manifold=self.manifold,
            riemannian_metric_weight_matrix=self.riemannian_metric_weight_matrix,
            state_trajectory=state,
            control_trajectory=control,
            state_trajectory_var=state_var,
            control_trajectory_var=control_var,
            # state_trajectory_var=None,
            # control_trajectory_var=None,
        )
        return tf.reduce_sum(energy_costs)


class ModeProbCostFunction(CostFunction):
    """Simple mode probability cost function class."""

    def __init__(
        self,
        prob_fn: Callable,  # ModeOptDynamics.predict_mode_probability
        weight: default_float() = 1.0,
    ):
        self.prob_fn = prob_fn
        self.weight = weight

    def __call__(
        self,
        state: ttf.Tensor2[HorizonPlusOne, StateDim],
        control: ttf.Tensor2[Horizon, ControlDim],
        state_var: Optional[ttf.Tensor2[HorizonPlusOne, StateDim]] = None,
        control_var: Optional[ttf.Tensor2[Horizon, ControlDim]] = None,
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
        vector_var = tf.expand_dims(vector_var, -2)  # [Horizon, 1, Dim]
        trace = tf.linalg.trace(vector_var @ weight_matrix)  # [Horizon,]
        cost += trace  # [Horizon, 1, 1]
    return cost[:, 0, 0]


# def state_control_quadratic_cost_fn(
#     state: ttf.Tensor2[HorizonPlusOne, StateDim],
#     control: ttf.Tensor2[Horizon, ControlDim],
#     Q: Union[
#         ttf.Tensor2[StateDim, StateDim], ttf.Tensor3[HorizonPlusOne, StateDim, StateDim]
#     ],
#     R: Union[
#         ttf.Tensor2[ControlDim, ControlDim],
#         ttf.Tensor3[Horizon, ControlDim, ControlDim],
#     ],
#     state_var: Optional[ttf.Tensor2[HorizonPlusOne, StateDim]] = None,
#     control_var: Optional[ttf.Tensor2[Horizon, ControlDim]] = None,
# ):
#     state_cost = quadratic_cost_fn(state, Q, state_var)
#     control_cost = quadratic_cost_fn(control, R, control_var)
#     return state_cost + control_cost
#     # return tf.reduce_sum(state_cost) + tf.reduce_sum(control_cost)


def terminal_state_cost_fn(
    state: ttf.Tensor2[One, StateDim],
    Q: ttf.Tensor2[StateDim, StateDim],
    target_state: ttf.Tensor2[One, StateDim],
    state_var: Optional[ttf.Tensor2[One, StateDim]] = None,
):
    assert len(state.shape) == 2
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
    state_trajectory: ttf.Tensor2[HorizonPlusOne, StateDim],
    control_trajectory: ttf.Tensor2[HorizonPlusOne, ControlDim],
    manifold: GPManifold,
    riemannian_metric_weight_matrix: float = 1.0,
    state_trajectory_var: Optional[ttf.Tensor2[HorizonPlusOne, StateDim]] = None,
    control_trajectory_var: Optional[ttf.Tensor2[HorizonPlusOne, ControlDim]] = None,
):
    # Append zeros to control trajectory
    control_trajectory = tf.concat(
        [
            control_trajectory,
            tf.zeros([1, tf.shape(control_trajectory)[1]], dtype=default_float()),
        ],
        0,
    )
    if control_trajectory_var is not None:
        control_trajectory_var = tf.concat(
            [
                control_trajectory_var,
                tf.zeros(
                    [1, tf.shape(control_trajectory_var)[1]], dtype=default_float()
                ),
            ],
            0,
        )

    # Calcualted the expeted metric at each point along trajectory
    input_mean, input_var = combine_state_controls_to_input(
        state_trajectory,
        control_trajectory,
        state_trajectory_var,
        control_trajectory_var,
    )
    # input_mean = tf.concat([state_trajectory, control_trajectory], -1)
    expected_riemannian_metric = (
        manifold.metric(input_mean[:-1, :]) @ riemannian_metric_weight_matrix
    )

    velocities = input_mean[1:, :] - input_mean[:-1, :]
    if input_var is None:
        velocities_var = None
    else:
        velocities_var = input_var[:-1, :]
    # velocities_var = None
    # if state_trajectory_var is not None and control_trajectory_var is not None:
    #     input_var = tf.concat([state_trajectory_var, control_trajectory_var], -1)
    #     # velocities_var = input_var[1:, :]
    #     velocities_var = input_var[:-1, :]

    riemannian_energy = quadratic_cost_fn(
        vector=velocities,
        weight_matrix=expected_riemannian_metric,
        vector_var=velocities_var,
    )
    riemannian_energy_sum = tf.reduce_sum(riemannian_energy)
    return riemannian_energy_sum
