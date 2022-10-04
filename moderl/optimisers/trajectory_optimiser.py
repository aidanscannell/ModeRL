#!/usr/bin/env python3
from functools import partial
from typing import Callable, List, Optional, Sequence, Union

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from moderl.custom_types import ControlTrajectory, ObjectiveFn
from scipy.optimize import LinearConstraint, NonlinearConstraint

tfd = tfp.distributions


class TrajectoryOptimiser:
    """Optimises a trajectory given an objective function"""

    def __init__(
        self,
        max_iterations: int,
        initial_solution: ControlTrajectory,
        objective_fn: Optional[ObjectiveFn] = None,
        constraints: Optional[List[Union[LinearConstraint, NonlinearConstraint]]] = [],
        keep_last_solution: bool = True,
        method: Optional[str] = "SLSQP",
    ):
        self.max_iterations = max_iterations
        self.objective_fn = objective_fn
        self.constraints = constraints
        self.keep_last_solution = keep_last_solution
        self.method = method

        if initial_solution:
            self.previous_solution = initial_solution
            self.initial_solution = initial_solution.copy()
        else:
            raise NotImplementedError("Please provide initial_solution")

        self.optimiser = gpf.optimizers.Scipy()

    def optimise(
        self, callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = None
    ):
        # ) -> ControlMeanAndVariance:
        """Returns a sequence of controls of length self.horizon"""
        if self.objective_fn is None:
            raise RuntimeError(
                "Please set `objective_fn` before using TrajectoryOptimisationController"
            )
        if not self.keep_last_solution:
            self.reset()

        optimisation_result = self.optimiser.minimize(
            closure=partial(self.objective_fn, initial_solution=self.previous_solution),
            variables=self.previous_solution.trainable_variables,
            method=self.method,
            constraints=self.constraints,
            step_callback=callback,
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
