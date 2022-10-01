#!/usr/bin/env python3
from typing import Callable, Optional, Union

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from moderl.constraints import build_mode_chance_constraints_scipy
from moderl.cost_functions import COST_FUNCTION_OBJECTS, CostFunction
from moderl.custom_types import ControlTrajectory, State, StateDim
from moderl.dynamics import ModeRLDynamics
from moderl.optimisers import TrajectoryOptimiser
from moderl.rollouts import rollout_ControlTrajectory_in_ModeRLDynamics

from .base import TrajectoryOptimisationController
from .utils import find_solution_in_desired_mode

tfd = tfp.distributions


class ExplorativeController(TrajectoryOptimisationController):
    def __init__(
        self,
        start_state: Union[ttf.Tensor1[StateDim], np.ndarray],
        dynamics: ModeRLDynamics,
        explorative_objective_fn: Callable[
            [ModeRLDynamics, ControlTrajectory, State], float
        ],
        cost_fn: CostFunction,
        control_dim: int,
        horizon: int = 10,
        max_iterations: int = 100,
        mode_satisfaction_prob: float = 0.8,
        keep_last_solution: bool = True,
        callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = None,
        method: Optional[str] = "SLSQP",
        # name: str = "ExplorativeController",
    ):
        if not isinstance(start_state, tf.Tensor):
            start_state = tf.constant(start_state, dtype=default_float())
        self.start_state = start_state
        self.dynamics = dynamics
        self.callback = callback
        # TODO use setter to build constraints when mode_satisfaction_prob is set
        self.mode_satisfaction_prob = mode_satisfaction_prob

        def augmentd_objective_fn(initial_solution: ControlTrajectory) -> ttf.Tensor0:
            """Augmented objective (expected cost over trajectory) with exploration objective"""
            state_dists = rollout_ControlTrajectory_in_ModeRLDynamics(
                dynamics=self.dynamics,
                control_trajectory=initial_solution,
                start_state=self.start_state,
            )
            control_dists = initial_solution()
            exploration_cost = explorative_objective_fn(
                dynamics=self.dynamics,
                initial_solution=initial_solution,
                start_state=self.start_state,
            )
            return -exploration_cost + cost_fn(
                # return cost_fn(
                state=state_dists,
                control=control_dists,
            )

        initial_solution = find_solution_in_desired_mode(
            dynamics=dynamics,
            horizon=horizon,
            control_dim=control_dim,
            start_state=start_state,
        )
        mode_chance_constraints = build_mode_chance_constraints_scipy(
            dynamics=dynamics,
            control_trajectory=initial_solution,
            start_state=start_state,
            lower_bound=mode_satisfaction_prob,
            upper_bound=1.0,  # max prob=1.0
            # compile=False,
            compile=True,
        )
        trajectory_optimiser = TrajectoryOptimiser(
            max_iterations=max_iterations,
            initial_solution=initial_solution,
            objective_fn=augmentd_objective_fn,
            keep_last_solution=keep_last_solution,
            constraints=[mode_chance_constraints],
            method=method,
        )

        super().__init__(trajectory_optimiser=trajectory_optimiser)

    def get_config(self):
        return {
            "start_state": self.start_state.numpy(),
            "dynamics": tf.keras.utils.serialize_keras_object(self.dynamics),
            # "explorative_objective_fn": tf.keras.utils.serialize_keras_object(self.explorative_objective_fn),
            "cost_fn": tf.keras.utils.serialize_keras_object(self.cost_fn),
            "control_dim": self.trajectory_optimiser.initial_solution.control_dim,
            "horizon": self.trajectory_optimiser.initial_solution.horizon,
            "max_iterations": self.trajectory_optimiser.max_iterations,
            "mode_satisfatction_prob": self.mode_satisfaction_prob,
            "keep_last_solution": self.trajectory_optimiser.keep_last_solution,
            "method": self.trajectory_optimiser.method,
        }

    @classmethod
    def from_config(cls, cfg: dict):
        dynamics = tf.keras.layers.deserialize(
            cfg["dynamics"], custom_objects={"ModeRLDynamics": ModeRLDynamics}
        )
        # TODO implement cost function serialisation
        cost_fn = tf.keras.layers.deserialize(
            cfg["cost_fn"], custom_objects=COST_FUNCTION_OBJECTS
        )
        return cls(
            # start_state=cfg["start_state"],
            start_state=np.array(cfg["start_state"]),
            dynamics=dynamics,
            # explorative_objective_fn=explorative_objective_fn,
            cost_fn=cost_fn,
            control_dim=cfg["control_dim"],
            horizon=cfg["horizon"],
            max_iterations=cfg["max_iterations"],
            mode_satisfaction_prob=cfg["mode_satisfaction_prob"],
            keep_last_solution=cfg["keep_last_solution"],
            method=cfg["method"],
        )


# class ExplorativeTrajectoryOptisationController(ControllerInterface):
#     # dynamics: ModeRLDynamics
#     # control_dim: int
#     # trajectory_optimiser: TrajectoryOptimiser
#     # callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]]

#     def __init__(
#         self,
#         start_state: ttf.Tensor1[StateDim],
#         dynamics: ModeRLDynamics,
#         cost_fn: CostFunction,
#         control_dim: int,
#         horizon: int = 10,
#         max_iterations: int = 100,
#         mode_satisfaction_prob: float = 0.8,
#         keep_last_solution: bool = True,
#         callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = None,
#         method: Optional[str] = "SLSQP",
#         name: str = "ExplorativeController",
#     ):
#         super().__init__(name=name)
#         self.start_state = start_state
#         self.dynamics = dynamics
#         self.cost_fn = cost_fn
#         self.control_dim = control_dim
#         self.callback = callback
#         # TODO use setter to build constraints when mode_satisfaction_prob is set
#         self.mode_satisfaction_prob = mode_satisfaction_prob

#         initial_solution = ControlTrajectory(
#             dist=tfd.Deterministic(
#                 tf.Variable(np.random.random((horizon, control_dim)) * 0.001)
#             )
#         )
#         # initial_solution = self.find_initial_solution_in_desired_mode()

#         # mode_chance_constraints = build_mode_chance_constraints_scipy(
#         #     dynamics,
#         #     start_state,
#         #     horizon,
#         #     control_dim=self.initial_solution.control_dim,
#         #     lower_bound=mode_satisfaction_prob,
#         # )

#         # explorative_objective = build_explorative_objective(
#         #     dynamics, cost_fn, start_state
#         # )

#         self.trajectory_optimiser = TrajectoryOptimiser(
#             max_iterations=max_iterations,
#             initial_solution=initial_solution,
#             objective_fn=self.objective_fn,
#             keep_last_solution=keep_last_solution,
#             # constraints=[mode_chance_constraints],
#             method=method,
#         )

#     def __call__(
#         self, timestep: Optional[int] = None, dist: bool = False
#     ) -> Union[
#         tfd.Distribution,
#         Union[ttf.Tensor1[ControlDim], ttf.Tensor2[Horizon, ControlDim]],
#     ]:
#         if dist:  # tfd.Distribution
#             if timestep is not None:
#                 return self.trajectory_optimiser.previous_solution.dist[timestep]
#             else:
#                 return self.trajectory_optimiser.previous_solution.dist
#         else:
#             return self.trajectory_optimiser.previous_solution(timestep=timestep)

#     def optimise(self):
#         optimisation_result = self.trajectory_optimiser.optimise(self.callback)
#         return optimisation_result

#     def objective_fn(self, initial_solution: ControlTrajectory) -> ttf.Tensor0:
#         """Combines the greedy cost function"""

#         # Rollout controls in dynamics
#         # control_means, control_vars = initial_solution(variance=True)
#         control_dists = initial_solution()
#         print("blh")
#         print("control_dists")
#         print(control_dists)
#         state_dists = self.dynamics.rollout_control_trajectory(
#             control_trajectory=initial_solution, start_state=self.start_state
#         )
#         print("blh 2")
#         print("state_dists")
#         print(state_dists)

#         gating_entropy = gating_function_entropy(
#             dynamics=self.dynamics,
#             initial_solution=initial_solution,
#             start_state=self.start_state,
#         )
#         print("gating_entropy")
#         print(gating_entropy)

#         return -gating_entropy + self.cost_fn(
#             state_dists.mean(),
#             control_dists.mean(),
#             state_dists.variance(),
#             control_dists.variance(),
#         )

#     def get_config(self):
#         return {
#             "start_state": self.trajectory_optimiser.initial_solution.start_state.numpy(),
#             # "target_state": self.initial_solution.target_state.numpy(),
#             "cost_fn": tf.keras.utils.serialize_keras_object(self.cost_fn),
#             "dynamics": tf.keras.utils.serialize_keras_object(self.dynamics),
#             "horizon": self.trajectory_optimiser.initial_solution.horizon,
#             "max_iterations": self.trajectory_optimiser.max_iterations,
#             "keep_last_solution": self.trajectory_optimiser.keep_last_solution,
#             "method": self.trajectory_optimiser.method,
#         }

#     @classmethod
#     def from_config(cls, cfg: dict):
#         dynamics = tf.keras.layers.deserialize(
#             cfg["dynamics"], custom_objects={"ModeRLDynamics": ModeRLDynamics}
#         )
#         controller = cls(
#             start_state=tf.constant(cfg["start_state"], dtype=default_float()),
#             # target_state=tf.constant(cfg["target_state"], dtype=default_float()),
#             dynamics=dynamics,
#             horizon=cfg["horizon"],
#             max_iterations=cfg["max_iterations"],
#             keep_last_solution=cfg["keep_last_solution"],
#             method=cfg["method"],
#         )
#         return controller
