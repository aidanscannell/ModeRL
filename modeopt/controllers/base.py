#!/usr/bin/env python3
import abc
from typing import Union

import tensorflow_probability as tfp
from modeopt.custom_types import State
from scipy.optimize import LinearConstraint, NonlinearConstraint

tfd = tfp.distributions


class Controller(abc.ABC):
    @abc.abstractmethod
    def optimise(self):
        raise NotImplementedError

    def constraints(self) -> Union[LinearConstraint, NonlinearConstraint]:
        raise NotImplementedError

    def control_dim(self) -> int:
        raise NotImplementedError


class FeedbackController(Controller, abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State):
        raise NotImplementedError


class NonFeedbackController(Controller, abc.ABC):
    @abc.abstractmethod
    def __call__(self, timestep: int):
        raise NotImplementedError


# class Policy(FeedbackController):
#     @abc.abstractmethod
#     def __call__(self, state: State, timestep: int):
#         raise NotImplementedError


# @dataclass
# class VariationalTrajectory(NonFeedbackController):
#     """
#     A trainable trajectory that can be optimised by a TrajectoryOptimiser and used for control
#     """

#     dist: Union[tfd.MultivariateNormalDiag, tfd.Deterministic]  # [horizon, control_dim]
#     constraints_lower_bound: Optional[float] = None
#     constraints_upper_bound: Optional[float] = None

#     def __call__(
#         self, time_step: Optional[int] = None
#     ) -> Union[SingleControlMeanAndVariance, ControlMeanAndVariance]:
#         control_means = self.dist.mean()
#         control_vars = self.dist.variance()
#         if time_step is None:
#             return control_means, control_vars
#         else:
#             return (
#                 control_means[time_step : time_step + 1, :],
#                 control_vars[time_step : time_step + 1, :],
#             )

#     def entropy(
#         self, sum_over_traj: Optional[bool] = True
#     ) -> Union[ttf.Tensor0, ttf.Tensor1[Horizon]]:
#         if sum_over_traj:
#             return tf.reduce_sum(self.dist.entropy())
#         else:
#             return self.dist.entropy()

#     # def trainable_parameters(self):
#     #     return self.dist.trainable_variables


# @dataclass
# class DeterministicVariationalTrajectory(VariationalTrajectory):
#     dist: tfd.Deterministic  # [horizon, control_dim]
#     constraints_lower_bound: Optional[float] = None
#     constraints_upper_bound: Optional[float] = None

#     def constraints(self):
#         """Linear constraints on the mean of the control dist."""
#         if self.constraints_upper_bound is None or self.constraints_lower_bound is None:
#             return None
#         constraints_lower_bound = (
#             np.ones((self.horizon, 1)) * self.constraints_lower_bound
#         )
#         constraints_upper_bound = (
#             np.ones((self.horizon, 1)) * self.constraints_upper_bound
#         )
#         control_constraint_matrix = np.eye(self.horizon * self.control_dim)
#         return LinearConstraint(
#             control_constraint_matrix,
#             constraints_lower_bound.reshape(-1),
#             constraints_upper_bound.reshape(-1),
#         )


# @dataclass
# class GaussianVariationalTrajectory(VariationalTrajectory):
#     dist: tfd.MultivariateNormalDiag  # [horizon, control_dim]
#     constraints_lower_bound: Optional[float] = None
#     constraints_upper_bound: Optional[float] = None

#     def constraints(self):
#         """Linear constraints on the mean of the control dist."""
#         # if self.constraints_upper_bound is None or self.constraints_lower_bound is None:
#         #     return None
#         # constraints_lower_bound = (
#         #     np.ones((self.horizon, 1)) * self.constraints_lower_bound
#         # )
#         # constraints_upper_bound = (
#         #     np.ones((self.horizon, 1)) * self.constraints_upper_bound
#         # )
#         control_constraint_matrix = np.eye(
#             N=self.horizon * self.control_dim, M=self.horizon * self.control_dim * 2
#         )
#         return build_linear_constraints(
#             control_constraint_matrix, constraints_lower_bound, constraints_upper_bound
#         )

#         # control_constraint_matrix = np.eye(self.horizon * self.control_dim)
#         # return LinearConstraint(
#         #     control_constraint_matrix,
#         #     constraints_lower_bound.reshape(-1),
#         #     constraints_upper_bound.reshape(-1),
#         # )


# def constraints(
#     # horizon: int,
#     # control_dim: int,
#     control_constraint_matrix,
#     constraints_lower_bound: float,
#     constraints_upper_bound: float,
# ) -> LinearConstraint:
#     """Linear constraints on the mean of the control dist."""
#     constraints_lower_bound = np.ones((horizon, 1)) * constraints_lower_bound
#     constraints_upper_bound = np.ones((horizon, 1)) * constraints_upper_bound
#     return LinearConstraint(
#         control_constraint_matrix,
#         constraints_lower_bound.reshape(-1),
#         constraints_upper_bound.reshape(-1),
#     )
