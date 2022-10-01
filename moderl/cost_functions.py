#!/usr/bin/env python3
from typing import Callable, List, Optional, Union
import abc

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import default_float
from gpflow.models import GPModel
from tensor_annotations.axes import Batch

from moderl.custom_types import (
    ControlDim,
    Horizon,
    HorizonPlusOne,
    InputDim,
    One,
    StateDim,
)


class CostFunction(abc.ABC):
    """Base cost function class.

    To implement a mean function, write the __call__ method.
    """

    @abc.abstractmethod
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

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, cfg: dict):
        # TODO Need to implement from_config() for cost_fns to instantiate weight_matrix properly
        return cls(**cfg)


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

    def get_config(self) -> dict:
        return {"weight_matrix": self.weight_matrix.numpy()}


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

    def get_config(self) -> dict:
        return {"weight_matrix": self.weight_matrix.numpy()}


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

    def get_config(self) -> dict:
        return {
            "weight_matrix": self.weight_matrix.numpy(),
            "target_state": self.target_state.numpy(),
        }


class StateDiffCostFunction(CostFunction):
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
        """Expected quadratic state diff cost func

        :param state: Tensor representing a state trajectory
        :param control: Tensor representing control trajectory
        :param state_var: Tensor representing the variance over state trajectory
        :param control_var: Tensor representing control trajectory
        :returns: scalar cost
        """
        state_diffs = state[1:, :] - self.target_state
        # state_diffs = state[1:-1, :] - self.target_state
        state_diffs_cost = quadratic_cost_fn(
            vector=state_diffs,
            weight_matrix=self.weight_matrix,
            vector_var=state_var,
        )
        return tf.reduce_sum(state_diffs_cost)

    def get_config(self) -> dict:
        return {
            "weight_matrix": self.weight_matrix.numpy(),
            "target_state": self.target_state.numpy(),
        }


class ModeProbCostFunction(CostFunction):
    """Simple mode probability cost function class."""

    def __init__(
        self,
        prob_fn: Callable,  # ModeRLDynamics.predict_mode_probability
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


COST_FUNCTIONS = [
    ZeroCostFunction,
    StateQuadraticCostFunction,
    StateDiffCostFunction,
    TargetStateCostFunction,
    ControlQuadraticCostFunction,
]

COST_FUNCTION_OBJECTS = {cost_fn.__name__: cost_fn for cost_fn in COST_FUNCTIONS}
