#!/usr/bin/env python3
import abc
from dataclasses import dataclass
from typing import Callable

import gpflow as gpf
import tensorflow as tf
from modeopt.policies import VariationalPolicy
from modeopt.rollouts import rollout_policy_in_dynamics
from modeopt.dynamics import Dynamics


@dataclass
class TrajectoryOptimiserTrainingSpec:
    """
    Specification data class for model training. Models that require additional parameters for
    training should create a subclass of this class and add additional properties.
    """

    max_iterations: int = 100
    method: str = "SLSQP"
    disp: bool = True


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

    @abc.abstractmethod
    def optimise(self, start_state, training_spec: TrajectoryOptimiserTrainingSpec):
        """Optimise trajectories starting from an initial state"""
        raise NotImplementedError

    # @abc.abstractmethod
    # def loss(self, start_state):
    #     raise NotImplementedError
