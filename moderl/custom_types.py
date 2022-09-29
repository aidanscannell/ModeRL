#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, NewType, Optional, Tuple, Union

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from tensor_annotations import axes
from tensor_annotations.axes import Batch

tfd = tfp.distributions


# Dim = NewType("Dim", axes.Axis)
StateDim = NewType("StateDim", axes.Axis)
ControlDim = NewType("ControlDim", axes.Axis)
StateControlDim = NewType("StateControlDim", axes.Axis)
One = NewType("One", axes.Axis)
Horizon = NewType("Horizon", axes.Axis)
# BatchTimesHorizon = NewType("BatchTimesHorizon", axes.Axis)
NumData = NewType("NumData", axes.Axis)
InputDim = NewType("InputDim", axes.Axis)
OutputDim = NewType("OutputDim", axes.Axis)
# TwoStateDim = NewType("StateDim", axes.Axis)
HorizonPlusOne = NewType("HorizonPlusOne", axes.Axis)

Times = None


InputData = ttf.Tensor2[NumData, InputDim]
OutputData = ttf.Tensor2[NumData, OutputDim]
Dataset = Tuple[ttf.Tensor2[Batch, StateControlDim], ttf.Tensor2[Batch, StateDim]]

State = ttf.Tensor1[StateDim]
Control = ttf.Tensor1[ControlDim]
NextState = ttf.Tensor1[StateDim]

ControlTrajectoryMean = ttf.Tensor2[Horizon, ControlDim]
ControlTrajectoryVariance = ttf.Tensor2[Horizon, ControlDim]
# ControlTrajectory = Tuple[ControlTrajectoryMean, ControlTrajectoryVariance]

StateTrajectoryMean = ttf.Tensor2[Horizon, StateDim]
StateTrajectoryVariance = ttf.Tensor2[Horizon, StateDim]
StateTrajectory = Tuple[StateTrajectoryMean, StateTrajectoryVariance]


@dataclass
class ControlTrajectory(tf.Module):
    # dist: Union[tfd.MultivariateNormalDiag, tfd.Deterministic]  # [horizon, control_dim]
    dist: tfd.Distribution  # [horizon, control_dim]

    def __call__(
        self, timestep: Optional[int] = None
    ) -> Union[ttf.Tensor2[One, ControlDim], ttf.Tensor2[Horizon, ControlDim]]:
        if timestep is not None:
            return self.controls[timestep : timestep + 1]
        else:
            return self.controls

    @property
    # def controls(self) -> ControlTrajectoryMean:
    def controls(self) -> tfd.Distribution:
        return self.dist

    @property
    def horizon(self) -> int:
        return self.controls.mean().shape[0]

    @property
    def control_dim(self) -> int:
        return self.controls.mean().shape[1]

    def copy(self):
        return ControlTrajectory(self.dist.copy())


class Transition(NamedTuple):
    state: ttf.Tensor1[StateDim]
    control: ttf.Tensor1[ControlDim]
    next_state: ttf.Tensor1[StateDim]
    timestep: int


@dataclass
class TransitionBatch:
    states: ttf.Tensor2[Batch, StateDim]
    controls: ttf.Tensor2[Batch, ControlDim]
    next_states: ttf.Tensor2[Batch, StateDim]
    times: List[int]

    def __len__(self):
        self.state.shape[0]

    def as_tuple(self) -> Transition:
        return Transition(
            states=self.states,
            controls=self.controls,
            next_states=self.next_states,
            times=self.times,
        )

    def as_dataset(self, predict_state_differences: Optional[bool] = True) -> Dataset:
        if predict_state_differences:
            return (
                tf.concat([self.states, self.controls], -1),
                self.next_states - self.states,
            )
        else:
            return tf.concat([self.states, self.controls], -1), self.next_states

    def as_tf_dataset(self, batch_size: int):
        prefetch_size = tf.data.experimental.AUTOTUNE
        shuffle_buffer_size = self.num_data // 2
        self.num_batches_per_epoch = self.num_data // batch_size
        train_dataset = tf.data.Dataset.from_tensor_slices(self.as_dataset())
        train_dataset = (
            train_dataset.repeat()
            .prefetch(prefetch_size)
            .shuffle(buffer_size=shuffle_buffer_size)
            .batch(batch_size, drop_remainder=True)
        )
        return train_dataset

    @property
    def num_data(self):
        return self.states.shape[0]


ObjectiveFn = Callable[[ControlTrajectory], ttf.Tensor0]

# StateMean = ttf.Tensor2[Batch, StateDim]
# StateVariance = ttf.Tensor2[Batch, StateDim]
# StateMeanAndVariance = Tuple[StateMean, StateVariance]


# ControlMean = ttf.Tensor2[Batch, ControlDim]
# ControlVariance = ttf.Tensor2[Batch, ControlDim]
# ControlMeanAndVariance = Tuple[ControlMean, ControlVariance]

# SingleStateMean = ttf.Tensor1[StateDim]
# SingleStateVariance = ttf.Tensor1[StateDim]
# SingleStateMeanAndVariance = Tuple[SingleStateMean, SingleStateVariance]

# SingleControlMean = ttf.Tensor1[ControlDim]
# SingleControlVariance = ttf.Tensor1[ControlDim]
# SingleControlMeanAndVariance = Tuple[SingleControlMean, SingleControlVariance]

# SingleStateControlMean = ttf.Tensor1[StateControlDim]
# SingleStateControlVariance = ttf.Tensor1[StateControlDim]
# SingleStateControlMeanAndVariance = Tuple[
#     SingleStateControlMean, SingleStateControlVariance
# ]

# StateControlMean = ttf.Tensor2[Batch, StateControlDim]
# StateControlVariance = ttf.Tensor2[Batch, StateControlDim]
# StateControlMeanAndVariance = Tuple[StateControlMean, StateControlVariance]

# ControlMean = ttf.Tensor2[Batch, ControlDim]
# ControlVariance = ttf.Tensor2[Batch, ControlDim]

# ODEFunction = Callable[[State], State]


# BatchedState = ttf.Tensor2[Batch, StateDim]
# BatchedControl = ttf.Tensor2[Batch, ControlDim]


# @dataclass
# class BatchedGaussianState:
#     mean: ttf.Tensor2[Batch, StateDim]
#     var: ttf.Tensor2[Batch, StateDim] = None

#     def __add__(self, state):
#         assert isinstance(state, BatchedGaussianState)
#         new_state = BatchedGaussianState(mean=self.mean + state.mean)
#         if self.var is None:
#             new_state.var = state.var
#         elif state.var is None:
#             new_state.var = self.var
#         else:
#             new_state.var = state.var + self.var
#         return new_state


# @dataclass
# class BatchedGaussianControl:
#     mean: ttf.Tensor2[Batch, ControlDim]
#     var: ttf.Tensor2[Batch, ControlDim] = None


# @dataclass
# class BatchedGaussianStateControl:
#     state: BatchedGaussianState
#     control: BatchedGaussianControl

#     def to_tensor(self) -> StateControlMeanAndVariance:
#         input_mean = tf.concat([self.state.mean, self.control.mean], -1)
#         if self.state.var is None and self.control.var is None:
#             input_var = None
#             # state_var = tf.zeros(control_var.shape, dtype=default_float())
#             # control_var = tf.zeros(state_var.shape, dtype=default_float())
#         elif self.state.var is None and self.control.var is not None:
#             assert len(self.control.mean.shape) == 2
#             state_var = tf.zeros(self.control.var.shape, dtype=default_float())
#             input_var = tf.concat([state_var, self.control.var], -1)
#         elif self.state.var is not None and self.control.var is None:
#             assert len(self.state.mean.shape) == 2
#             control_var = tf.zeros(self.state.var.shape, dtype=default_float())
#             input_var = tf.concat([self.state.var, control_var], -1)
#         else:
#             input_var = tf.concat([self.state.var, self.control.var], -1)
#         return input_mean, input_var


# class StateControlDistribution:
#     def __init__(
#         self,
#         state_means: ttf.Tensor2[Horizon, StateDim],
#         control_means: ttf.Tensor2[Horizon, ControlDim],
#         state_vars: ttf.Tensor2[Horizon, StateDim] = None,
#         control_vars: ttf.Tensor2[Horizon, ControlDim] = None,
#     ):
#         self.states = tfd.MultivariateNormalDiag(
#             loc=state_means, scale_diag=tf.math.sqrt(state_vars)
#         )
#         self.controls = tfd.MultivariateNormalDiag(
#             loc=control_means, scale_diag=tf.math.sqrt(control_vars)
#         )
#         self.joint_state_control_trajectory: tfd.JointDistributionNamedAutoBatched(
#             {"states": self.states, "controls": self.controls}, batch_ndims=1
#         )

#     def to_means_and_vars(self):
#         means = tf.concat(tf.nest.flatten(self.joint_state_control.mean()), -1)
#         vars = tf.concat(tf.nest.flatten(self.joint_state_control.variance()), -1)
#         return means, vars
