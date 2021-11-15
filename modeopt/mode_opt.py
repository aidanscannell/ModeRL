#!/usr/bin/env python3
from functools import partial
from typing import NewType, Optional, Tuple, Union

import gpflow as gpf
import numpy as np
import tensor_annotations.tensorflow as ttf
from gpflow import Module, default_float
from tensor_annotations import axes
from tensor_annotations.axes import Batch
from tf_agents.environments import tf_py_environment

from modeopt.constraints import build_mode_chance_constraints_scipy
from modeopt.cost_functions import (
    build_riemmanian_energy_cost_fn,
    build_state_control_riemannian_energy_quadratic_cost_fn,
    state_control_quadratic_cost_fn,
    terminal_state_cost_fn,
)
from modeopt.dynamics.multimodal import ModeOptDynamics, ModeOptDynamicsTrainingSpec
from modeopt.policies import VariationalGaussianPolicy, VariationalPolicy
from modeopt.rollouts import rollout_policy_in_dynamics, rollout_policy_in_env
from modeopt.trajectory_optimisers import (
    ExplorativeTrajectoryOptimiser,
    ExplorativeTrajectoryOptimiserTrainingSpec,
    ModeVariationalTrajectoryOptimiser,
    ModeVariationalTrajectoryOptimiserTrainingSpec,
    VariationalTrajectoryOptimiser,
    VariationalTrajectoryOptimiserTrainingSpec,
)

StateDim = NewType("StateDim", axes.Axis)
ControlDim = NewType("ControlDim", axes.Axis)


def init_variational_gaussian_policy(
    horizon, control_dim, mu_noise=0.1, var_noise=0.01
):
    control_means = (
        np.ones((horizon, control_dim)) * 0.5
        + np.random.random((horizon, control_dim)) * mu_noise
    )
    control_vars = (
        np.ones((horizon, control_dim)) * 0.2
        + np.random.random((horizon, control_dim)) * var_noise
    )
    return VariationalGaussianPolicy(means=control_means, vars=control_vars)


class ModeOpt(Module):
    def __init__(
        self,
        start_state: ttf.Tensor2[Batch, StateDim],
        target_state: ttf.Tensor2[Batch, StateDim],
        env,
        policy: VariationalPolicy,
        dynamics: ModeOptDynamics,
        dataset: Tuple,
        desired_mode: int = 1,
        mode_chance_constraint_lower=0.5,
        horizon: int = 10,
        Q: ttf.Tensor2[StateDim, StateDim] = None,
        R: ttf.Tensor2[ControlDim, ControlDim] = None,
        Q_terminal: ttf.Tensor2[StateDim, StateDim] = None,
        riemannian_metric_cost_weight: Optional[default_float()] = None,
        riemannian_metric_covariance_weight: Optional[default_float()] = 1.0,
    ):
        self.start_state = start_state
        self.target_state = target_state
        self.dynamics = dynamics
        self.dataset = dataset
        self.desired_mode = desired_mode
        self.mode_chance_constraint_lower = mode_chance_constraint_lower
        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal
        self.riemannian_metric_cost_weight = riemannian_metric_cost_weight

        # Set policy
        if policy is None:
            self.policy = init_variational_gaussian_policy(
                horizon, control_dim=self.dynamics.control_dim
            )
        else:
            self.policy = policy

        # # Init terminal quadratic costs on states (Euclidean distance)
        # self.terminal_cost_fn = partial(
        #     terminal_state_cost_fn, Q=Q_terminal, target_state=self.target_state
        # )

        # Init quadratic cost functions for state, control and Riemannian energy
        # if riemannian_metric_cost_weight is None:
        #     self.cost_fn = partial(state_control_quadratic_cost_fn, Q=self.Q, R=self.R)
        # else:
        #     # TODO does this need to be rebuilt after training dynamics?
        #     self.cost_fn = build_state_control_riemannian_energy_quadratic_cost_fn(
        #         Q=Q,
        #         R=R,
        #         gp=self.dynamics.gating_gp,
        #         riemannian_metric_cost_weight=riemannian_metric_cost_weight,
        #         riemannian_metric_covariance_weight=riemannian_metric_covariance_weight,
        #     )

        # Init tf environment
        self.env = env
        self.tf_env = tf_py_environment.TFPyEnvironment(env)

    def optimise_policy(
        self,
        start_state,
        training_spec: Union[
            VariationalTrajectoryOptimiserTrainingSpec,
            ModeVariationalTrajectoryOptimiserTrainingSpec,
            ExplorativeTrajectoryOptimiserTrainingSpec,
        ],
    ):
        # Init terminal quadratic costs on states (Euclidean distance)
        self.terminal_cost_fn = partial(
            terminal_state_cost_fn,
            Q=training_spec.Q_terminal,
            target_state=self.target_state,
        )
        # if isinstance(training_spec, ModeVariationalTrajectoryOptimiserTrainingSpec):
        if isinstance(
            training_spec, ModeVariationalTrajectoryOptimiserTrainingSpec
        ) or isinstance(training_spec, VariationalTrajectoryOptimiserTrainingSpec):
            # Init quadratic cost functions for state, control and Riemannian energy
            if training_spec.riemannian_metric_cost_weight is None:
                cost_fn = partial(
                    state_control_quadratic_cost_fn,
                    Q=training_spec.Q,
                    R=training_spec.R,
                )
            else:
                # TODO does this need to be rebuilt after training dynamics?
                cost_fn = build_state_control_riemannian_energy_quadratic_cost_fn(
                    Q=training_spec.Q,
                    R=training_spec.R,
                    gp=self.dynamics.gating_gp,
                    riemannian_metric_cost_weight=training_spec.riemannian_metric_cost_weight,
                    riemannian_metric_covariance_weight=training_spec.riemannian_metric_covariance_weight,
                )
            if isinstance(training_spec, VariationalTrajectoryOptimiserTrainingSpec):
                trajectory_optimiser = VariationalTrajectoryOptimiser(
                    self.policy,
                    self.dynamics,
                    cost_fn=cost_fn,
                    terminal_cost_fn=self.terminal_cost_fn,
                )
            elif isinstance(
                training_spec, ModeVariationalTrajectoryOptimiserTrainingSpec
            ):
                trajectory_optimiser = ModeVariationalTrajectoryOptimiser(
                    self.policy,
                    self.dynamics,
                    cost_fn=cost_fn,
                    terminal_cost_fn=self.terminal_cost_fn,
                )
        elif isinstance(training_spec, ExplorativeTrajectoryOptimiserTrainingSpec):
            cost_fn = partial(
                state_control_quadratic_cost_fn,
                Q=training_spec.Q,
                R=training_spec.R,
            )
            trajectory_optimiser = ExplorativeTrajectoryOptimiser(
                self.policy,
                self.dynamics,
                cost_fn=cost_fn,
                terminal_cost_fn=self.terminal_cost_fn,
            )

        gpf.set_trainable(self.dynamics, False)
        # print("param")
        # for param in self.policy.trainable_parameters:
        #     print(param)
        # print(self.policy.trainable_variables)
        # gpf.set_trainable(self.policy, True)
        gpf.utilities.print_summary(self)
        if (
            training_spec.mode_chance_constraint_lower is None
            or training_spec.mode_chance_constraint_lower <= 0.0
        ):
            mode_chance_constraints = []
            print(
                "Turning mode chance constraints off because training_spec.mode_chance_constraint_lower is None or <=0.0"
            )
        else:
            mode_chance_constraints = build_mode_chance_constraints_scipy(
                mode_opt_dynamics=self.dynamics,
                start_state=start_state,
                horizon=self.horizon,
                lower_bound=training_spec.mode_chance_constraint_lower,
                upper_bound=1.0,
                compile=training_spec.compile_mode_constraint_fn,
            )
        return trajectory_optimiser.optimise(
            start_state=start_state,
            training_spec=training_spec,
            constraints=mode_chance_constraints,
        )

    def dynamics_rollout(self, start_state, start_state_var=None):
        return rollout_policy_in_dynamics(
            self.policy, self.dynamics, start_state, start_state_var=start_state_var
        )

    def env_rollout(self, start_state):
        return rollout_policy_in_env(self.env, self.policy, start_state=start_state)

    def optimise_dynamics(
        self,
        dataset,
        training_spec: ModeOptDynamicsTrainingSpec,
        trainable_variables=None,
    ):
        # TODO make mosvgpe trainable?
        self.dynamics.set_trainable(True, trainable_variables=trainable_variables)
        for param in self.policy.trainable_parameters:
            gpf.set_trainable(param, False)
        # gpf.set_trainable(self.policy, False)
        gpf.utilities.print_summary(self)
        self.dynamics._train(dataset, training_spec)
