#!/usr/bin/env python3
from typing import Callable, Optional, Union
import numpy as np

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float

# from moderl.controllers.utils import (
#     build_mode_variational_objective,
#     initialise_deterministic_trajectory,
# )
from moderl.cost_functions import (
    ControlQuadraticCostFunction,
    CostFunction,
    TargetStateCostFunction,
)
from moderl.custom_types import (
    ControlDim,
    ControlTrajectory,
    ObjectiveFn,
    State,
    StateDim,
    Horizon,
)
from moderl.dynamics import ModeRLDynamics, SVGPDynamicsWrapper
from moderl.dynamics.conditionals import svgp_covariance_conditional
from gpflow.conditionals import base_conditional
from moderl.utils import combine_state_controls_to_input

# from moderl.rollouts import rollout_controller_in_dynamics

# from .trajectory_optimisation.objectives import build_explorative_objective

from .controller import ControllerInterface

# from .constraints import build_mode_chance_constraints_scipy
from .trajectory_optimiser import TrajectoryOptimiser

tfd = tfp.distributions


# def build_explorative_objective(
#     dynamics: SVGPDynamicsWrapper, cost_fn: CostFunction, start_state: State
# ) -> ObjectiveFn:
#     def explorative_objective(
#         initial_solution: ControlTrajectory,
#     ) -> ttf.Tensor0:
#         # Rollout controls in dynamics
#         control_means, control_vars = initial_solution(variance=True)
#         state_means, state_vars = rollout_controls_in_dynamics(
#             dynamics=dynamics,
#             start_state=start_state,
#             control_means=control_means,
#             #         control_vars=control_vars,
#         )

#         h_means_prior, h_vars_prior = dynamics.uncertain_predict_gating(
#             state_means[1:, :], control_means
#         )
#         gating_gp = dynamics.desired_mode_gating_gp

#         input_means, input_vars = combine_state_controls_to_input(
#             state_means[1:, :],
#             control_means,
#             state_vars[1:, :],
#             control_vars,
#         )

#         h_means, h_vars = h_means_prior[0:1, :], h_vars_prior[0:1, :]
#         for t in range(1, initial_solution.horizon):
#             Xnew = input_means[t : t + 1, :]
#             Xobs = input_means[0:t, :]
#             f = h_means_prior[0:t, :]

#             Knn = svgp_covariance_conditional(X1=Xnew, X2=Xnew, svgp=gating_gp)[0, 0, :]
#             Kmm = svgp_covariance_conditional(X1=Xobs, X2=Xobs, svgp=gating_gp)[0, :, :]
#             Kmn = svgp_covariance_conditional(X1=Xobs, X2=Xnew, svgp=gating_gp)[0, :, :]
#             Kmm += tf.eye(Kmm.shape[0], dtype=default_float()) * default_jitter()
#             # Lm = tf.linalg.cholesky(Kmm)
#             # A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]
#             h_mean, h_var = base_conditional(
#                 Kmn=Kmn,
#                 Kmm=Kmm,
#                 Knn=Knn,
#                 f=f,
#                 full_cov=False,
#                 q_sqrt=None,
#                 white=False,
#             )
#             h_means = tf.concat([h_means, h_mean], 0)
#             h_vars = tf.concat([h_vars, h_var], 0)
#         h_dist = tfd.MultivariateNormalDiag(h_means, h_vars)
#         gating_entropy = h_dist.entropy()

#         return -tf.reduce_sum(gating_entropy) + cost_fn(
#             state_means, control_means, state_vars, control_vars
#         )

#     return explorative_objective


class ExplorativeController(ControllerInterface):
    dynamics: ModeRLDynamics
    control_dim: int
    trajectory_optimiser: TrajectoryOptimiser
    callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]]

    def __init__(
        self,
        start_state: ttf.Tensor1[StateDim],
        dynamics: ModeRLDynamics,
        cost_fn: CostFunction,
        control_dim: int,
        # explorative_
        horizon: int = 10,
        max_iterations: int = 100,
        mode_satisfaction_prob: float = 0.8,
        keep_last_solution: bool = True,
        callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = None,
        method: Optional[str] = "SLSQP",
        name: str = "ExplorativeController",
    ):
        super().__init__(name=name)
        self.start_state = start_state
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.control_dim = control_dim
        self.callback = callback
        # TODO use setter to build constraints when mode_satisfaction_prob is set
        self.mode_satisfaction_prob = mode_satisfaction_prob

        initial_solution = ControlTrajectory(
            dist=tfd.Deterministic(
                tf.Variable(np.random.random((horizon, control_dim)) * 0.001)
            )
        )
        # initial_solution = self.find_initial_solution_in_desired_mode()

        # mode_chance_constraints = build_mode_chance_constraints_scipy(
        #     dynamics,
        #     start_state,
        #     horizon,
        #     control_dim=self.initial_solution.control_dim,
        #     lower_bound=mode_satisfaction_prob,
        # )

        # explorative_objective = build_explorative_objective(
        #     dynamics, cost_fn, start_state
        # )

        self.trajectory_optimiser = TrajectoryOptimiser(
            max_iterations=max_iterations,
            initial_solution=initial_solution,
            objective_fn=self.objective_fn,
            keep_last_solution=keep_last_solution,
            # constraints=[mode_chance_constraints],
            method=method,
        )

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
            return self.trajectory_optimiser.previous_solution(timestep=timestep)

    def optimise(self):
        optimisation_result = self.trajectory_optimiser.optimise(self.callback)
        return optimisation_result

    def objective_fn(self, initial_solution: ControlTrajectory) -> ttf.Tensor0:
        """Combines the greedy cost function"""

        # Rollout controls in dynamics
        # control_means, control_vars = initial_solution(variance=True)
        control_dists = initial_solution()
        print("blh")
        print("control_dists")
        print(control_dists)
        state_dists = self.dynamics.rollout_control_trajectory(
            control_trajectory=initial_solution, start_state=self.start_state
        )
        print("blh 2")
        print("state_dists")
        print(state_dists)

        gating_entropy = gating_function_entropy(
            dynamics=self.dynamics,
            initial_solution=initial_solution,
            start_state=self.start_state,
        )
        print("gating_entropy")
        print(gating_entropy)

        return -gating_entropy + self.cost_fn(
            state_dists.mean(),
            control_dists.mean(),
            state_dists.variance(),
            control_dists.variance(),
        )


def gating_function_entropy(
    dynamics: ModeRLDynamics, initial_solution: ControlTrajectory, start_state: State
) -> ttf.Tensor0:
    control_dists = initial_solution()
    state_dists = dynamics.rollout_control_trajectory(
        control_trajectory=initial_solution, start_state=start_state
    )
    input_dists = combine_state_controls_to_input(
        state=state_dists[1:], control=control_dists
    )
    print("input_dists")
    print(input_dists)
    h_means, h_vars = dynamics.mosvgpe.gating_network.gp.predict_f(
        input_dists.mean(), full_cov=True
    )
    h_vars = (
        h_vars + tf.eye(h_vars.shape[1], h_vars.shape[2], dtype=default_float()) * 1e-6
    )
    h_dist = tfd.MultivariateNormalFullCovariance(h_means, h_vars[0, :, :] ** 2)
    gating_entropy = h_dist.entropy()
    return tf.reduce_sum(gating_entropy)


# def find_solution_in_desired_mode(
#     self,
#     start_state: State,
#     fake_target_state: State,
#     # max_iterations: int = 1000,
#     # method: str = "SLSQP",
# ) -> ControlTrajectory:
#     # state_dim = self.start_state.shape[-1]
#     # terminal_state_cost_matrix =
#     # tf.eye(
#     #     self.dynamics.state_dim, dtype=default_float()
#     # )
#     terminal_cost_fn = TargetStateCostFunction(
#         weight_matrix=tf.eye(self.dynamics.state_dim, dtype=default_float()),
#         target_state=fake_target_state,
#     )
#     control_cost_fn = ControlQuadraticCostFunction(
#         weight_matrix=tf.eye(self.control_dim, dtype=default_float())
#     )
#     initial_cost_fn = terminal_cost_fn + control_cost_fn

#     # initial_solution = initialise_deterministic_trajectory(
#     #     self.horizon, control_dim
#     # )
#     objective_fn = build_mode_variational_objective(
#         self.dynamics, initial_cost_fn, self.start_state
#     )
#     explorative_controller = TrajectoryOptimiser(
#         max_iterations=1000,
#         initial_solution=initial_solution,
#         objective_fn=objective_fn,
#         method="SLSQP",
#     )
#     explorative_controller.optimise()
#     return explorative_controller.previous_solution
