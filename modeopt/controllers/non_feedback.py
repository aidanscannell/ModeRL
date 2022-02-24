#!/usr/bin/env python3
from functools import partial
from typing import Callable, List, Optional, Sequence, Union

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.optimizers.scipy import NonlinearConstraintClosure
from modeopt.custom_types import ControlMeanAndVariance
from modeopt.trajectories import BaseTrajectory
from scipy.optimize import LinearConstraint, NonlinearConstraint

from .base import NonFeedbackController

tfd = tfp.distributions

ObjectiveFn = Callable[[BaseTrajectory], ttf.Tensor0]


class TrajectoryOptimisationController(NonFeedbackController):
    """
    Non-feedback controller that optimises a sequence of controls using a
    TrajectoryOptimiser and a ModeOptDynamics model.
    """

    def __init__(
        self,
        num_iterations: int,
        initial_solution: BaseTrajectory,
        objective_fn: ObjectiveFn,
        constraints_lower_bound: Optional[Union[float, Sequence[float]]] = None,
        constraints_upper_bound: Optional[Union[float, Sequence[float]]] = None,
        keep_last_solution: bool = True,
        constraints: Optional[List[Union[LinearConstraint, NonlinearConstraint]]] = [],
        nonlinear_constraint_closure: NonlinearConstraintClosure = None,
        nonlinear_constraint_kwargs: dict = {"lb": -0.1, "ub": 0.1},
        method: Optional[str] = "SLSQP",
    ):
        self.num_iterations = num_iterations
        self.objective_fn = objective_fn
        self.keep_last_solution = keep_last_solution
        self.method = method

        if initial_solution:
            self.initial_solution = initial_solution
        else:
            raise NotImplementedError("Please provide initial_solution")
        self.previous_solution = self.initial_solution
        self.initial_solution = self.initial_solution.copy()

        # control_constraints = build_control_constraints(
        #     trajectory=self.initial_solution,
        #     constraints_lower_bound=constraints_lower_bound,
        #     constraints_upper_bound=constraints_upper_bound,
        # )
        # if constraints:
        #     self.constraints = constraints.append(control_constraints)
        # else:
        #     self.constraints = control_constraints
        # print("self.constraints")
        # print(self.constraints)
        self.nonlinear_constraint_closure = nonlinear_constraint_closure
        self.nonlinear_constraint_kwargs = nonlinear_constraint_kwargs

        self.optimiser = gpf.optimizers.Scipy()

    def __call__(
        self,
        timestep: Optional[int] = None,
        variance: bool = False
        # self,
        # variance: bool = False,
    ) -> ControlMeanAndVariance:
        # self.optimise()
        # control_mean, control_var = self.previous_solution(variance=variance)
        return self.previous_solution(timestep=timestep, variance=variance)

    def optimise(
        self,
        callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = [],
    ) -> ControlMeanAndVariance:
        """Returns a sequence of controls of length self.horizon"""
        if self.objective_fn is None:
            raise RuntimeError(
                "Please set `objective_fn` before using TrajectoryOptimisationController"
            )
        objective_fn_closure = partial(
            self.objective_fn, initial_solution=self.previous_solution
        )
        optimisation_result = self.optimiser.minimize(
            objective_fn_closure,
            variables=self.previous_solution.trainable_variables,
            method=self.method,
            # constraints=self.constraints,
            nonlinear_constraint_closure=self.nonlinear_constraint_closure,
            nonlinear_constraint_kwargs=self.nonlinear_constraint_kwargs,
            step_callback=callback,
            # compile=False,
            compile=True,
            options={"maxiter": self.num_iterations},
        )

        # if self.keep_last_solution:
        #     self.previous_solution = best_solution.roll(-self.replan_freq, dims=0)
        #     # Note that initial_solution[i] is the same for all values of [i],
        #     # so just pick i = 0
        #     self.previous_solution[-self.replan_freq :] = self.initial_solution[0]
        print("Optimisation result:")
        print(optimisation_result)

        # get trajectory from opt_result
        # self.previous_solution = best_solution
        # return best_solution

        # plan = self.trajectory_optimiser.optimise(self.objective_fn)
        # return plan

    def reset(self, horizon: Optional[int] = None):
        """Resets the trajectory."""
        if horizon:
            raise NotImplementedError("how to set trajectory with new horizon?")
            # self.trajectory_optimiser = trajectoryoptimiser(
            #     self.num_iterations,
            #     horizon,
            #     self.control_dim,
            #     self.initial_solution,
            #     lower_bound=self.constraints_lower_bound,
            #     upper_bound=self.constraints_upper_bound,
            #     keep_last_solution=self.keep_last_solution,
            #     constraints=self.constraints,
            #     method=self.method,
            # )
            self.previous_solution = self.initial_solution.copy()
        else:
            self.previous_solution = self.initial_solution.copy()

    @property
    def horizon(self) -> int:
        return self.previous_solution.horizon

    @property
    def control_dim(self) -> int:
        return self.previous_solution.control_dim


# class VariationalTrajectoryOptimisationController(TrajectoryOptimisationController):
#     """
#     A trainable trajectory that can be optimised by a TrajectoryOptimiser and used for control
#     """

#     def __init__(
#         self,
#         dist: Union[
#             tfd.MultivariateNormalDiag, tfd.Deterministic
#         ],  # [horizon, control_dim]
#         start_state,
#         constraints_lower_bound: Optional[float] = None,
#         constraints_upper_bound: Optional[float] = None,
#         dynamics: ModeOptDynamics = None,
#         cost_fn: CostFunction = None,
#     ):
#         super().__init__(
#             num_iterations=num_iterations,
#             horizon=horizon,
#             initial_solution=initial_solution,
#             constraints_lower_bound=constraints_lower_bound,
#             constraints_upper_bound=constraints_upper_bound,
#             keep_last_solution=keep_last_solution,
#             constraints=constraints,
#             method=method,
#         )
#         self.start_state = start_state
#         self.cost_fn = cost_fn
#         self.dynamics

#     def objective(self) -> ttf.Tensor0:
#         """Evidence Lower BOund"""
#         entropy = self.dist.entropy()

#         # Rollout controls in dynamics
#         control_means = self.dist.mean()
#         control_vars = self.dist.variance()
#         state_means, state_vars = rollout_controls_in_dynamics(
#             dynamics=self.dynamics,
#             start_state=self.start_state,
#             control_means=control_means,
#             control_vars=control_vars,
#         )

#         # Calculate costs
#         expected_costs = self.cost_fn(
#             state=state_means,
#             control=control_means,
#             state_var=state_vars,
#             control_var=control_vars,
#         )
#         elbo = -expected_costs + entropy
#         return elbo

#     def entropy(
#         self, sum_over_traj: Optional[bool] = True
#     ) -> Union[ttf.Tensor0, ttf.Tensor1[Horizon]]:
#         if sum_over_traj:
#             return tf.reduce_sum(self.dist.entropy())
#         else:
#             return self.dist.entropy()


# class VariationalTrajectory(NonFeedbackController):
#     # class TrajectoryOptimisationController(NonFeedbackController):
#     """
#     A trainable trajectory that can be optimised by a TrajectoryOptimiser and used for control
#     """

#     def __init__(
#         self,
#         dist: Union[
#             tfd.MultivariateNormalDiag, tfd.Deterministic
#         ],  # [horizon, control_dim]
#         constraints_lower_bound: Optional[float] = None,
#         constraints_upper_bound: Optional[float] = None,
#         trajectory_optimiser: TrajectoryOptimiser = None,
#         dynamics: ModeOptDynamics = None,
#     ):
#         self.dist = dist

#     def __call__(
#         self, time_step: Optional[int] = None
#     ) -> Union[SingleControlMeanAndVariance, ControlMeanAndVariance]:
#         self.optimiser.optimiser()
#         control_means = self.dist.mean()
#         control_vars = self.dist.variance()
#         if time_step is None:
#             return control_means, control_vars
#         else:
#             return (
#                 control_means[time_step : time_step + 1, :],
#                 control_vars[time_step : time_step + 1, :],
#             )

#     def objective(self) -> ttf.Tensor0:
#         """Evidence Lower BOund"""
#         entropy = self.dist.entropy()

#         # Rollout controls in dynamics
#         control_means = self.dist.mean()
#         control_vars = self.dist.variance()
#         state_means, state_vars = rollout_controls_in_dynamics(
#             dynamics=self.dynamics,
#             start_state=self.start_state,
#             control_means=control_means,
#             control_vars=control_vars,
#         )

#         # Calculate costs
#         expected_costs = self.cost_fn(
#             state=state_means,
#             control=control_means,
#             state_var=state_vars,
#             control_var=control_vars,
#         )
#         elbo = -expected_costs + entropy
#         return elbo

#     def entropy(
#         self, sum_over_traj: Optional[bool] = True
#     ) -> Union[ttf.Tensor0, ttf.Tensor1[Horizon]]:
#         if sum_over_traj:
#             return tf.reduce_sum(self.dist.entropy())
#         else:
#             return self.dist.entropy()

#     @property
#     def horizon(self) -> int:
#         return self.previous_solution.horizon

#     @property
#     def control_dim(self) -> int:
#         return self.previous_solution.control_dim

#     # def copy(self):
#     #     return VariationalTrajectory(self.dist.copy())


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
