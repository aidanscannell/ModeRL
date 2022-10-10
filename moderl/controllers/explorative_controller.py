#!/usr/bin/env python3
from typing import Callable, Optional, Union

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from moderl.constraints import build_mode_chance_constraints_scipy
from moderl.custom_types import ControlTrajectory, State, StateDim
from moderl.dynamics import ModeRLDynamics
from moderl.optimisers import TrajectoryOptimiser
from moderl.reward_functions import REWARD_FUNCTION_OBJECTS, RewardFunction
from moderl.rollouts import rollout_ControlTrajectory_in_ModeRLDynamics
from moderl.utils import save_json_config
from scipy.optimize import LinearConstraint

from .base import TrajectoryOptimisationController
from .utils import find_solution_in_desired_mode


tfd = tfp.distributions


class ExplorativeController(TrajectoryOptimisationController):
    def __init__(
        self,
        start_state: Union[ttf.Tensor1[StateDim], np.ndarray],
        dynamics: ModeRLDynamics,
        reward_fn: RewardFunction,
        control_dim: int,
        explorative_objective_fn: Optional[
            Callable[[ModeRLDynamics, ControlTrajectory, State], float]
        ] = None,
        horizon: int = 10,
        max_iterations: int = 100,
        mode_satisfaction_prob: float = 0.8,
        exploration_weight: float = 1.0,
        keep_last_solution: bool = True,
        callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = None,
        control_lower_bound: Optional[float] = None,
        control_upper_bound: Optional[float] = None,
        method: Optional[str] = "SLSQP",
        # name: str = "ExplorativeController",
        initial_solution: Optional[ControlTrajectory] = None,
    ):
        if not isinstance(start_state, tf.Tensor):
            start_state = tf.constant(start_state, dtype=default_float())
        self.start_state = start_state
        self.dynamics = dynamics
        self.reward_fn = reward_fn
        self.exploration_weight = exploration_weight
        self.control_lower_bound = control_lower_bound
        self.control_upper_bound = control_upper_bound
        self.callback = callback
        # TODO use setter to build constraints when mode_satisfaction_prob is set
        self.mode_satisfaction_prob = mode_satisfaction_prob

        if explorative_objective_fn is None:
            explorative_objective_fn = lambda *args, **kwargs: 0.0

        def augmentd_objective_fn(initial_solution: ControlTrajectory) -> ttf.Tensor0:
            """Adds explorative objective to expected reward over trajectory"""
            state_dists = rollout_ControlTrajectory_in_ModeRLDynamics(
                dynamics=self.dynamics,
                control_trajectory=initial_solution,
                start_state=self.start_state,
            )
            control_dists = initial_solution()
            exploration_reward = explorative_objective_fn(
                dynamics=self.dynamics,
                initial_solution=initial_solution,
                start_state=self.start_state,
            )
            reward = reward_fn(state=state_dists, control=control_dists)
            # logger.debug("Reward: {}".format(reward))
            # logger.debug("Exploration reward: {}".format(exploration_reward))
            return exploration_reward * exploration_weight + reward

        if initial_solution is None:
            initial_solution = find_solution_in_desired_mode(
                dynamics=dynamics,
                horizon=horizon,
                control_dim=control_dim,
                start_state=start_state,
            )
        constraints = [
            build_mode_chance_constraints_scipy(
                dynamics=dynamics,
                control_trajectory=initial_solution,
                start_state=start_state,
                lower_bound=mode_satisfaction_prob,
                upper_bound=1.0,  # max prob=1.0
                # compile=False,
                compile=True,
            )
        ]
        if control_lower_bound is not None and control_upper_bound is not None:
            constraints.append(
                LinearConstraint(
                    np.eye(horizon * control_dim),
                    control_lower_bound,
                    control_upper_bound,
                )
            )
        trajectory_optimiser = TrajectoryOptimiser(
            max_iterations=max_iterations,
            initial_solution=initial_solution,
            objective_fn=augmentd_objective_fn,
            keep_last_solution=keep_last_solution,
            constraints=constraints,
            method=method,
        )
        super().__init__(trajectory_optimiser=trajectory_optimiser)

    def previous_solution(self):
        return self.trajectory_optimiser.previous_solution

    def save(self, save_filename: str):
        save_json_config(self, filename=save_filename)

    @classmethod
    def load(cls, load_filename: str):
        with open(load_filename, "r") as read_file:
            json_cfg = read_file.read()
        return tf.keras.models.model_from_json(
            json_cfg, custom_objects={"ExplorativeController": ExplorativeController}
        )

    def get_config(self) -> dict:
        return {
            "start_state": self.start_state.numpy(),
            "dynamics": tf.keras.utils.serialize_keras_object(self.dynamics),
            "reward_fn": tf.keras.utils.serialize_keras_object(self.reward_fn),
            "control_dim": self.trajectory_optimiser.initial_solution.control_dim,
            "horizon": self.trajectory_optimiser.initial_solution.horizon,
            "max_iterations": self.trajectory_optimiser.max_iterations,
            "mode_satisfaction_prob": self.mode_satisfaction_prob,
            "exploration_weight": self.exploration_weight,
            "keep_last_solution": self.trajectory_optimiser.keep_last_solution,
            "control_lower_bound": self.control_lower_bound,
            "control_upper_bound": self.control_upper_bound,
            "method": self.trajectory_optimiser.method,
            "initial_solution": tf.keras.utils.serialize_keras_object(
                self.trajectory_optimiser.previous_solution
            ),
        }

    @classmethod
    def from_config(cls, cfg: dict):
        dynamics = tf.keras.layers.deserialize(
            cfg["dynamics"], custom_objects={"ModeRLDynamics": ModeRLDynamics}
        )
        # TODO implement reward function serialisation
        reward_fn = tf.keras.layers.deserialize(
            cfg["reward_fn"], custom_objects=REWARD_FUNCTION_OBJECTS
        )
        try:
            initial_solution = tf.keras.layers.deserialize(
                cfg["initial_solution"],
                custom_objects={"ControlTrajectory": ControlTrajectory},
            )
        except KeyError:
            initial_solution = None
        try:
            explorative_objective_fn = cfg["explorative_objective_fn"]
        except KeyError:
            explorative_objective_fn = None
        return cls(
            start_state=np.array(cfg["start_state"]),
            dynamics=dynamics,
            explorative_objective_fn=explorative_objective_fn,
            reward_fn=reward_fn,
            control_dim=cfg["control_dim"],
            horizon=cfg["horizon"],
            max_iterations=cfg["max_iterations"],
            mode_satisfaction_prob=cfg["mode_satisfaction_prob"],
            exploration_weight=cfg["exploration_weight"],
            keep_last_solution=cfg["keep_last_solution"],
            control_lower_bound=cfg["control_lower_bound"],
            control_upper_bound=cfg["control_upper_bound"],
            method=cfg["method"],
            initial_solution=initial_solution,
        )
