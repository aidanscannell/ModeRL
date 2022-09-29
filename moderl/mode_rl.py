#!/usr/bin/env python3
from builtins import NotImplementedError
import datetime
import os
from typing import Callable, List, Optional, Union

from tf_agents.environments import py_environment
import numpy as np
import simenvs
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import default_float

# from gpflow.utilities.keras import try_array_except_none, try_val_except_none
from tensor_annotations.axes import Batch

from moderl.controllers import ControllerInterface, ExplorativeController
from moderl.custom_types import Dataset, StateDim
from moderl.dynamics import ModeRLDynamics
from moderl.rollouts import collect_data_from_env

Callback = Callable


def mode_rl_loop(
    start_state: ttf.Tensor2[Batch, StateDim],
    dynamics: ModeRLDynamics,
    # mode_controller: Controller,
    env: py_environment.PyEnvironment,
    explorative_controller: ControllerInterface = None,
    desired_mode: int = 1,
    mode_satisfaction_probability: float = 0.7,  # Mode satisfaction probability (0, 1]
    # save_freq: Optional[Union[str, int]] = None,
    # log_dir: str = "./",
    # max_to_keep: int = None,
    num_explorative_trajectories: int = 6,
):
    converged = False
    while not converged:

        opt_result = explorative_controller.optimise()

        X, Y = [], []
        for i in range(num_explorative_trajectories):
            if isinstance(start_state, tf.Tensor):
                start_state = start_state.numpy()
            X_, Y_ = collect_data_from_env(
                env=env,
                start_state=start_state,
                controls=explorative_controller(),
            )
            X.append(X_)
            Y.append(Y_)
        X = np.concatenate(X, 0)
        Y = np.concatenate(Y, 0)
        new_dataset = (X, Y)

        # new_data = self.explore_env()
        dynamics.update_dataset(new_dataset)
        dynamics.optimise()

        # (trajectory, converged) = .find_trajectory_to_target()
        # if converged:
        #     in_desired_mode = .check_mode_remaining(trajectory)
        #     if in_desired_mode:
        #         print("Found delta mode remaining trajectory that's converged")
        #         return True
        #     else:
        #         converged = False

    print("opt_result['success']")
    print(opt_result["success"])
    if not opt_result["success"]:
        raise NotImplementedError
        # explorative_controller.reset()
        # return explore_env()
