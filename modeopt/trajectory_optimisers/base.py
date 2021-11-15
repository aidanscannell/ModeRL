#!/usr/bin/env python3
import abc
import typing
from dataclasses import dataclass
from typing import Callable

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import default_float
from modeopt.dynamics import Dynamics
from modeopt.policies import VariationalPolicy
from tensor_annotations import axes
from tensor_annotations.axes import Batch

StateDim = typing.NewType("StateDim", axes.Axis)
ControlDim = typing.NewType("ControlDim", axes.Axis)


@dataclass
class TrajectoryOptimiserTrainingSpec:
    """
    Specification data class for model training. Models that require additional parameters for
    training should create a subclass of this class and add additional properties.
    """

    max_iterations: int = 100
    method: str = "SLSQP"
    disp: bool = True
    compile_loss_fn: bool = True  # loss function in tf.function?
    monitor: gpf.monitor.Monitor = None
    manager: tf.train.CheckpointManager = None


class TrajectoryOptimiser(abc.ABC):
    """
    A trajectory optimiser optimises a sequence of actions given a dynamics model
    that is used for virtual rollouts.
    """

    def __init__(
        self,
        policy: VariationalPolicy,
        dynamics: Dynamics,
        cost_fn: Callable,
        terminal_cost_fn: Callable,
        optimiser=None,  # has to be scipy?
        horizon: int = 10,
    ):
        self.policy = policy
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.terminal_cost_fn = terminal_cost_fn
        assert horizon > 0, "horizon has to be > 0"
        self.horizon = horizon

        if optimiser is None:
            self.optimiser = gpf.optimizers.Scipy()
        else:
            self.optimiser = optimiser

        self.state_dim = self.dynamics.state_dim
        self.control_dim = self.policy.control_dim

        self._objective_closure = None

    def optimise(
        self,
        start_state: ttf.Tensor2[Batch, StateDim],
        training_spec: TrajectoryOptimiserTrainingSpec,
        constraints=[],
    ):
        """Optimise trajectories starting from an initial state"""
        if training_spec.monitor and training_spec.manager:

            def callback(step, variables, values):
                training_spec.monitor(step)
                training_spec.manager.save()

        elif training_spec.monitor is not None:

            def callback(step, variables, values):
                training_spec.monitor(step)

        elif training_spec.manager is not None:

            def callback(step, variables, values):
                training_spec.manager.save()

        else:
            callback = None

        if self.objective_closure is None:
            self.build_objective(start_state, compile=training_spec.compile_loss_fn)

        policy_variables = [
            param.unconstrained_variable for param in self.policy.trainable_parameters
        ]

        optimisation_result = self.optimiser.minimize(
            self.objective_closure,
            policy_variables,
            method=training_spec.method,
            constraints=constraints,
            step_callback=callback,
            options={
                "disp": training_spec.disp,
                "maxiter": training_spec.max_iterations,
            },
        )
        print("Optimisation result:")
        print(optimisation_result)
        print("self.policy.trainable_variables")
        print(self.policy.trainable_variables)
        print("self.policy()")
        print(self.policy())
        print("self.policy.variational_dist.mean()")
        print(self.policy.variational_dist.mean())
        print(self.policy.variational_dist.variance())
        return optimisation_result

    def build_objective(
        self, start_state: ttf.Tensor2[Batch, StateDim], compile: bool = False
    ) -> Callable:
        def objective():
            return -self.objective(start_state)

        if compile:
            objective = tf.function(objective)
        self._objective_closure = objective
        return self._objective_closure

    @property
    def objective_closure(self) -> Callable:
        """Objective to optimise"""
        if self._objective_closure is not None:
            return self._objective_closure
        else:
            print("objective not built yet")
            return None

    @abc.abstractmethod
    def objective(self, start_state: ttf.Tensor2[Batch, StateDim]):
        """Objective to optimise"""
        raise NotImplementedError
