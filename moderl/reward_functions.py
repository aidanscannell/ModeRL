#!/usr/bin/env python3
import abc
from typing import Optional, Union

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from mosvgpe.keras.utils import try_array_except_none
from tensor_annotations.axes import Batch

from moderl.custom_types import Horizon, InputDim, One, StateDim

tfd = tfp.distributions


class RewardFunction(abc.ABC):
    """Base reward function class."""

    @abc.abstractmethod
    def __call__(
        self,
        state: Union[tfd.Normal, tfd.Deterministic],  # [Horizon, StateDim]
        control: Union[tfd.Normal, tfd.Deterministic],  # [Horizon, ControlDim]
    ) -> ttf.Tensor0:
        """Expected reward func under Normally distributed states and controls

        E[ c(x_T, u_T) ] = \mu_{e_T}^T H \mu_{e_T} + tr(H \Sigma_{e_T})
        with:
            e = x_T - target_state ~ N(\mu_e, \Sigma_e)
            x ~ N(\mu_x, \Sigma_x)
            T is final time step
        """
        raise NotImplementedError(
            "Implement the __call__ method for this reward function"
        )

    def __add__(self, other):
        return Additive(self, other)

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, cfg: dict):
        # TODO Need to implement from_config() for reward_fns to instantiate weight_matrix properly
        return cls(**cfg)


class Additive(RewardFunction):
    def __init__(self, first_part, second_part):
        RewardFunction.__init__(self)
        self.add_1 = first_part
        self.add_2 = second_part

    def __call__(
        self,
        state: Union[tfd.Normal, tfd.Deterministic],  # [Horizon, StateDim]
        control: Union[tfd.Normal, tfd.Deterministic],  # [Horizon, ControlDim]
    ) -> ttf.Tensor0:
        return tf.add(
            self.add_1(state=state, control=control),
            self.add_2(state=state, control=control),
        )

    def get_config(self) -> dict:
        return {
            "first_part": tf.keras.utils.serialize_keras_object(self.add_1),
            "second_part": tf.keras.utils.serialize_keras_object(self.add_2),
        }

    @classmethod
    def from_config(cls, cfg: dict):
        # TODO Need to implement from_config() for reward_fns to instantiate weight_matrix properly
        first_part = tf.keras.layers.deserialize(
            cfg["first_part"], custom_objects=REWARD_FUNCTION_OBJECTS
        )
        second_part = tf.keras.layers.deserialize(
            cfg["second_part"], custom_objects=REWARD_FUNCTION_OBJECTS
        )
        return cls(first_part=first_part, second_part=second_part)


class ControlQuadraticRewardFunction(RewardFunction):
    def __init__(
        self,
        weight_matrix: Union[
            ttf.Tensor2[StateDim, StateDim], ttf.Tensor3[Horizon, StateDim, StateDim]
        ],
    ):
        self.weight_matrix = weight_matrix

    def __call__(
        self,
        state: Union[tfd.Normal, tfd.Deterministic],  # [Horizon, StateDim]
        control: Union[tfd.Normal, tfd.Deterministic],  # [Horizon, ControlDim]
    ) -> ttf.Tensor0:
        """Expected quadratic integral control reward func

        E[ \Sum_{t=0}^{T-1} r(x_t, u_t) ] = - \mu_{u_t}^T R \mu_{u_t} + tr(R \Sigma_{u_t})

        with:
            u ~ N(\mu_u, \Sigma_u)
        """
        control_rewards = quadratic_reward_fn(
            vector=control.mean(),
            weight_matrix=self.weight_matrix,
            vector_var=control.variance(),
        )
        return tf.reduce_sum(control_rewards)

    def get_config(self) -> dict:
        return {"weight_matrix": self.weight_matrix.numpy()}

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(weight_matrix=try_array_except_none(cfg, "weight_matrix"))


class TargetStateRewardFunction(RewardFunction):
    def __init__(
        self,
        weight_matrix: ttf.Tensor2[StateDim, StateDim],
        target_state: ttf.Tensor2[One, StateDim],
    ):
        self.weight_matrix = weight_matrix
        self.target_state = target_state

    def __call__(
        self,
        state: Union[tfd.Normal, tfd.Deterministic],  # [Horizon, StateDim]
        control: Union[tfd.Normal, tfd.Deterministic],  # [Horizon, ControlDim]
    ) -> ttf.Tensor0:
        """Expected quadratic terminal state reward func

        E[ r(x_T, u_T) ] = -\mu_{e_T}^T H \mu_{e_T} + tr(H \Sigma_{e_T})

        with:
            e = x_T - target_state ~ N(\mu_e, \Sigma_e)
            x ~ N(\mu_x, \Sigma_x)
            T is final time step
        """
        error = state.mean()[-1:, :] - self.target_state
        if isinstance(state, tfd.Deterministic):
            terminal_state_var = None
        else:
            terminal_state_var = state.variance()[-1:, :]
        terminal_reward = quadratic_reward_fn(
            vector=error,
            weight_matrix=self.weight_matrix,
            vector_var=terminal_state_var,
        )
        return terminal_reward

    def get_config(self) -> dict:
        return {
            "weight_matrix": self.weight_matrix.numpy(),
            "target_state": self.target_state.numpy(),
        }

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(
            weight_matrix=try_array_except_none(cfg, "weight_matrix"),
            target_state=try_array_except_none(cfg, "target_state"),
        )


class StateDiffRewardFunction(RewardFunction):
    def __init__(
        self,
        weight_matrix: ttf.Tensor2[StateDim, StateDim],
        target_state: ttf.Tensor2[One, StateDim],
    ):
        self.weight_matrix = weight_matrix
        self.target_state = target_state

    def __call__(
        self,
        state: Union[tfd.Normal, tfd.Deterministic],  # [Horizon, StateDim]
        control: Union[tfd.Normal, tfd.Deterministic],  # [Horizon, ControlDim]
    ) -> ttf.Tensor0:
        """Expected quadratic state diff reward func"""
        state_diffs = state.mean()[1:, :] - self.target_state
        # state_diffs = state[1:-1, :] - self.target_state
        state_diff_rewards = quadratic_reward_fn(
            vector=state_diffs,
            weight_matrix=self.weight_matrix,
            vector_var=state.variance(),
        )
        return tf.reduce_sum(state_diff_rewards)

    def get_config(self) -> dict:
        return {
            "weight_matrix": self.weight_matrix.numpy(),
            "target_state": self.target_state.numpy(),
        }

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(
            weight_matrix=try_array_except_none(cfg, "weight_matrix"),
            target_state=try_array_except_none(cfg, "target_state"),
        )


def quadratic_reward_fn(
    vector: ttf.Tensor2[Batch, InputDim],
    weight_matrix: Union[
        ttf.Tensor2[InputDim, InputDim], ttf.Tensor3[Batch, InputDim, InputDim]
    ],
    vector_var: Optional[ttf.Tensor2[Batch, InputDim]] = None,
):
    assert len(vector.shape) == 2
    vector = tf.expand_dims(vector, -2)
    reward = -vector @ weight_matrix @ tf.transpose(vector, [0, 2, 1])
    if vector_var is not None:
        assert len(vector_var.shape) == 2
        vector_var = tf.expand_dims(vector_var, -2)  # [Horizon, 1, Dim]
        trace = tf.linalg.trace(vector_var @ weight_matrix)  # [Horizon,]
        reward += trace  # [Horizon, 1, 1]
    return reward[:, 0, 0]


REWARD_FUNCTIONS = [
    Additive,
    StateDiffRewardFunction,
    TargetStateRewardFunction,
    ControlQuadraticRewardFunction,
]

REWARD_FUNCTION_OBJECTS = {
    reward_fn.__name__: reward_fn for reward_fn in REWARD_FUNCTIONS
}
