#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Callable, NewType, Optional, Tuple, Union

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
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

    def get_config(self) -> dict:
        return {"dist": type(self.dist).__name__, "mean": self.dist.mean().numpy()}

    @classmethod
    def from_config(cls, cfg: dict):
        if "Deterministic" in cfg["dist"]:
            means = tf.Variable(cfg["mean"], dtype=default_float())
            dist = tfd.Deterministic(loc=means)
        else:
            raise NotImplementedError(
                "Only implemented serialisation for tfd.Deterministic"
            )
        return cls(dist=dist)


ObjectiveFn = Callable[[ControlTrajectory], ttf.Tensor0]
