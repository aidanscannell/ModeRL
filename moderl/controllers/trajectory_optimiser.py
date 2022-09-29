#!/usr/bin/env python3
from functools import partial
from typing import Callable, List, Optional, Sequence, Union
import scipy

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp

# from gpflow.optimizers.scipy import NonlinearConstraintClosure
# from moderl.constraints import build_linear_control_constraints
# from moderl.custom_types import ControlMeanAndVariance
from moderl.custom_types import ControlTrajectory, ObjectiveFn

# from scipy.optimize import LinearConstraint, NonlinearConstraint


tfd = tfp.distributions


class TrajectoryOptimiser:
    """
    Non-feedback controller that optimises a trajectory given an objective function
    """

    def __init__(
        self,
        max_iterations: int,
        initial_solution: ControlTrajectory,
        objective_fn: Optional[ObjectiveFn] = None,
        # constraints_lower_bound: Optional[Union[float, Sequence[float]]] = None,
        # constraints_upper_bound: Optional[Union[float, Sequence[float]]] = None,
        # constraints: Optional[List[Union[LinearConstraint, NonlinearConstraint]]] = [],
        # nonlinear_constraint_closure: NonlinearConstraintClosure = None,
        # nonlinear_constraint_kwargs: dict = {"lb": -0.1, "ub": 0.1},
        keep_last_solution: bool = True,
        method: Optional[str] = "SLSQP",
    ):
        self.max_iterations = max_iterations
        self.objective_fn = objective_fn
        # self.constraints_lower_bound = constraints_lower_bound
        # self.constraints_upper_bound = constraints_upper_bound
        self.keep_last_solution = keep_last_solution
        self.method = method

        if initial_solution:
            self.initial_solution = initial_solution
        else:
            raise NotImplementedError("Please provide initial_solution")
        self.previous_solution = self.initial_solution
        self.initial_solution = self.initial_solution.copy()

        # control_constraints = build_linear_control_constraints(
        #     trajectory=self.initial_solution,
        #     lower_bound=constraints_lower_bound,
        #     upper_bound=constraints_upper_bound,
        # )
        # if len(constraints) > 0 and control_constraints is not None:
        #     self.constraints = constraints.append(control_constraints)
        # elif len(constraints) > 0:
        #     self.constraints = constraints
        # elif control_constraints is not None:
        #     self.constraints = control_constraints
        # else:
        #     self.constraints = []

        # self.nonlinear_constraint_closure = nonlinear_constraint_closure
        # self.nonlinear_constraint_kwargs = nonlinear_constraint_kwargs

        self.optimiser = gpf.optimizers.Scipy()

    def optimise(
        self,
        callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = None,
    ):
        # ) -> ControlMeanAndVariance:
        """Returns a sequence of controls of length self.horizon"""
        if self.objective_fn is None:
            raise RuntimeError(
                "Please set `objective_fn` before using TrajectoryOptimisationController"
            )
        if not self.keep_last_solution:
            self.reset()

        objective_fn_closure = partial(
            self.objective_fn, initial_solution=self.previous_solution
        )

        # tolerance = 1e-10
        # tfp.optimizer.lbfgs_minimize(
        #     self.objective_fn,
        #     initial_position=self.previous_solution,
        #     tolerance=tolerance,
        # )
        # optimisation_result = scipy.optimize.minimize(
        #     fun=self.objective_fn,
        #     # fun=objective_fn_closure,
        #     x0=self.previous_solution.trainable_variables,
        #     method=self.method,
        #     # step_callback=callback,
        #     options={"maxiter": self.max_iterations},
        # )

        optimisation_result = self.optimiser.minimize(
            objective_fn_closure,
            variables=self.previous_solution.trainable_variables,
            method=self.method,
            # constraints=self.constraints,
            # nonlinear_constraint_closure=self.nonlinear_constraint_closure,
            # nonlinear_constraint_kwargs=self.nonlinear_constraint_kwargs,
            step_callback=callback,
            # compile=False,
            compile=True,
            options={"maxiter": self.max_iterations},
        )

        # if self.keep_last_solution:
        #     self.previous_solution = best_solution.roll(-self.replan_freq, dims=0)
        #     # Note that initial_solution[i] is the same for all values of [i],
        #     # so just pick i = 0
        #     self.previous_solution[-self.replan_freq :] = self.initial_solution[0]
        print("Optimisation result:")
        print(optimisation_result)
        return optimisation_result

    def reset(self, horizon: Optional[int] = None):
        """Resets the trajectory."""
        if horizon is not None:
            raise NotImplementedError("how to set trajectory with new horizon?")
        else:
            self.previous_solution = self.initial_solution.copy()
