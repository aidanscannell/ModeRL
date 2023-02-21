#!/usr/bin/env python3
import abc
from dataclasses import dataclass
from typing import Optional, Union

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from moderl.custom_types import ControlDim, Horizon, State
from moderl.optimisers import TrajectoryOptimiser


tfd = tfp.distributions


class ControllerInterface(tf.Module, abc.ABC):
    # class Controller(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State = None, timestep: int = None):
        raise NotImplementedError

    @abc.abstractmethod
    def optimise(self):
        raise NotImplementedError

    # def constraints(self) -> Union[LinearConstraint, NonlinearConstraint]:
    #     raise NotImplementedError

    def control_dim(self) -> int:
        raise NotImplementedError


@dataclass
class TrajectoryOptimisationController(ControllerInterface):
    trajectory_optimiser: TrajectoryOptimiser

    def __call__(
        self, timestep: Optional[int] = None, dist: bool = False
    ) -> Union[
        tfd.Distribution,
        Union[ttf.Tensor1[ControlDim], ttf.Tensor2[Horizon, ControlDim]],
    ]:
        if dist:  # tfd.Distribution
            if timestep is not None:
                return self.trajectory_optimiser.previous_solution.dist[timestep]
            else:
                return self.trajectory_optimiser.previous_solution.dist
        else:
            return self.trajectory_optimiser.previous_solution(timestep=timestep).mean()

    def optimise(self):
        optimisation_result = self.trajectory_optimiser.optimise(self.callback)
        return optimisation_result

    @property
    def horizon(self):
        return self.trajectory_optimiser.initial_solution.horizon

    @property
    def control_dim(self):
        return self.trajectory_optimiser.initial_solution.control_dim

    def rollout_in_dynamics(self) -> tfd.Distribution:  # [Horizon, StateDim]
        """Rollout a ControlTrajectory in dynamics"""
        # state_dist = tfd.Normal(loc=self.start_state, scale=0.0)
        state_dist = tfd.Deterministic(loc=self.start_state)
        state_means = state_dist.mean()
        state_vars = state_dist.variance()
        # control_trajectory = self()
        for t in range(self.horizon):
            state_dist = self.dynamics.forward(
                state=state_dist,
                control=self.trajectory_optimiser.initial_solution(timestep=t),
                predict_state_difference=False,
            )
            state_means = tf.concat([state_means, state_dist.mean()], 0)
            state_vars = tf.concat([state_vars, state_dist.variance()], 0)
        return tfd.Normal(loc=state_means, scale=tf.math.sqrt(state_vars))
